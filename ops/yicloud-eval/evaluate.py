"""Read-only paired validation through native model assembly and NFC restore."""
def main():
    import hashlib
    import json
    import math
    import os
    import sys
    import time
    from pathlib import Path

    import torch
    import torch.distributed as dist

    root = Path(os.environ['RUN_DIR'])
    variant = os.environ['EVAL_VARIANT']
    expected_step = {'native': 0, 'trained': 796}[variant]
    rank = int(os.environ['RANK'])
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'yicloud-tuned'))
    from runtime import install
    install(os.environ['STREAM_ROOT'])
    from specforge.training.trainer import Trainer
    from specforge.training.backend import FSDPTrainingBackend
    from specforge.training.strategies.base import DSparkTrainStrategy
    from specforge.algorithms.common.dflash_family_model import OnlineDFlashModel

    totals = {}
    slots = []
    anchor_hash = hashlib.sha256()
    original_loss = DSparkTrainStrategy.forward_loss
    original_anchors = OnlineDFlashModel._sample_anchor_positions
    original_fit = Trainer.fit
    started = time.monotonic()

    def write(name, value):
        path = root / name
        temp = path.with_suffix('.partial')
        temp.write_text(json.dumps(value, indent=2))
        temp.replace(path)

    def anchors(self, *args, **kwargs):
        values = original_anchors(self, *args, **kwargs)
        for value in values:
            cpu = value.detach().cpu().contiguous()
            anchor_hash.update(str((str(cpu.dtype), tuple(cpu.shape))).encode())
            anchor_hash.update(cpu.view(torch.uint8).numpy().tobytes())
        return values

    def loss(self, batch, ctx=None):
        assert not self.trainable_module().training, 'evaluation must not train'
        assert len(batch.sample_ids) == 1
        # Per-slot seeds make the two independently loaded models see identical anchors.
        seed = int.from_bytes(hashlib.sha256(
            ('denovo-eval-20261105:' + batch.sample_ids[0]).encode()).digest()[:8], 'big') % (2**63)
        torch.manual_seed(seed)
        result = original_loss(self, batch, ctx)
        assert torch.isfinite(result.loss).all()
        for key, pair in result.ratio_metrics.items():
            numerator, denominator = [torch.as_tensor(v).detach().double().flatten() for v in pair]
            assert numerator.shape == denominator.shape
            assert torch.isfinite(numerator).all() and torch.isfinite(denominator).all()
            assert (denominator >= 0).all()
            if key not in totals:
                totals[key] = torch.zeros((2, numerator.numel()), dtype=torch.float64, device=numerator.device)
            totals[key][0] += numerator
            totals[key][1] += denominator
        slots.extend(batch.sample_ids)
        if len(slots) == 1 or len(slots) % 32 == 0:
            write(f'{variant}-progress-rank{rank}.json', {
                'variant': variant, 'rank': rank, 'batches': len(slots),
                'expected_batches': 737, 'elapsed_seconds': time.monotonic() - started,
            })
        return result

    def reject_step(*args, **kwargs):
        raise AssertionError('optimizer update forbidden in evaluation')

    def fit(self):
        assert self.global_step == expected_step
        assert self._controller.eval_data_factory is not None
        write(f'{variant}-loaded-rank{rank}.json', {
            'rank': rank, 'variant': variant, 'global_step': self.global_step,
            'native_checkpoint_restored': variant == 'trained',
        })
        # Skip only the training loop; retain native Trainer.fit cleanup. The
        # restored step is already durable, and step zero never triggers saving.
        self._controller.fit = lambda loader: self.global_step
        legacy_metrics = self.evaluate()
        assert len(slots) == len(set(slots)) == 737
        expected = [f'yicloud-denovo-full-stream-eval:{i:08d}' for i in range(rank, 5896, 8)]
        assert slots == expected, 'validation coverage/order changed'
        schemas = [None] * dist.get_world_size()
        dist.all_gather_object(schemas, {k: list(v.shape) for k, v in totals.items()})
        assert all(s == schemas[0] for s in schemas)
        ratios, counts = {}, {}
        for key in sorted(totals):
            value = totals[key]
            dist.all_reduce(value)
            num, den = value.cpu().tolist()
            assert all(d > 0 for d in den), (key, den)
            ratios[key] = [n / d for n, d in zip(num, den)]
            counts[key] = {'numerator': num, 'denominator': den}
            if len(ratios[key]) == 1:
                ratios[key] = ratios[key][0]
        assert len(ratios['accuracy_position']) == 5
        assert all(0 <= x <= 1 for x in ratios['accuracy_position'])
        objective = 0.1 * ratios['ce_loss'] + 0.9 * ratios['l1_loss'] + ratios['confidence_loss']
        assert math.isfinite(objective)
        value = original_fit(self)
        assert value == self.global_step == expected_step
        result = {
            'passed': True, 'variant': variant, 'rank': rank, 'global_step': value,
            'optimizer_updates': 0, 'validation_windows': 5896, 'local_windows': len(slots),
            'anchor_sha256': anchor_hash.hexdigest(), 'metrics': ratios,
            'weighted_objective': objective, 'ratio_counts': counts,
            'legacy_evaluator_metrics': legacy_metrics,
            'elapsed_seconds': time.monotonic() - started,
            'scope': 'teacher-forced full-context features; 32 sampled anchors/window; offline proxies, not serving speed or SWE solve rate',
        }
        write(f'{variant}-result-rank{rank}.json', result)
        if rank == 0:
            print(json.dumps({k: v for k, v in result.items() if k != 'ratio_counts'}), flush=True)
        return value

    OnlineDFlashModel._sample_anchor_positions = anchors
    DSparkTrainStrategy.forward_loss = loss
    FSDPTrainingBackend.step = reject_step
    Trainer.fit = fit
    from specforge.cli import main as cli
    assert cli(sys.argv[1:]) == 0


if __name__ == '__main__':
    main()
