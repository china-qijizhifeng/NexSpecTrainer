"""Check native sixteen-column accept/commit contracts before loading V4.1.

These synthetic logits exercise the real SGLang kernels. They do not establish
target-model output equivalence or target/MTP KV contents; full-model E2E owns
those checks.
"""

import argparse
import hashlib
import json
from pathlib import Path
import traceback


def accept_gate(torch, *, prefix_dtype, fused=False):
    from sglang.kernels.ops.speculative.dspark.dspark_accept import (
        AcceptGreedy,
        FinalizeAcceptLens,
    )
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import BuildOutTokens

    # Every first rejection position, including full acceptance, and every
    # effective verifier length, including the anchor-only path.
    scenarios = [(raw, cap) for raw in range(16) for cap in range(1, 17)]
    if fused:
        scenarios = [(0, 16), (5, 16), (15, 16), (15, 1)]
    rows = [[9 + column for column in range(16)] for _ in scenarios]
    predictions = [[10 + column if column < raw else 50 for column in range(16)]
                   for raw, _ in scenarios]
    candidates = torch.tensor(rows, dtype=torch.int64, device="cuda")
    prediction = torch.tensor(predictions, dtype=torch.int64, device="cuda")
    logits = torch.full((len(scenarios) * 16, 4096 if fused else 64), -10.0,
                        dtype=torch.float32, device="cuda")
    logits.scatter_(1, prediction.reshape(-1, 1), 10.0)
    cutoff = torch.tensor([cap for _, cap in scenarios], dtype=torch.int32, device="cuda")
    prefix_values = [127 + 128 * (index % 33) for index in range(len(scenarios))]
    prefix = torch.tensor(prefix_values, dtype=prefix_dtype, device="cuda")
    correct, bonus, trim = AcceptGreedy.execute(
        candidates=candidates, target_logits=logits, verify_num_draft_tokens=16,
        cutoff_verify_lens=cutoff, fused_argmax=fused)
    finalized = FinalizeAcceptLens.execute(correct_len=correct, cap_trim_lens=trim,
                                           prefix_lens=prefix)
    emitted = BuildOutTokens.execute(draft_tokens=candidates[:, 1:].contiguous(),
        correct_len=correct, bonus=bonus, gamma=15, verify_num_draft_tokens=16)
    accepted = [min(raw, cap - 1) for raw, cap in scenarios]
    expected_bonus = [predictions[index][count] for index, count in enumerate(accepted)]
    expected_trim = [raw - count for (raw, _), count in zip(scenarios, accepted)]
    assert correct.cpu().tolist() == accepted
    assert bonus.cpu().tolist() == expected_bonus
    assert trim.cpu().tolist() == expected_trim
    assert finalized.commit_lens.cpu().tolist() == [count + 1 for count in accepted]
    assert finalized.new_seq_lens.dtype == prefix_dtype
    assert finalized.new_seq_lens.cpu().tolist() == [
        size + count + 1 for size, count in zip(prefix_values, accepted)]
    assert finalized.cap_trim_lens.cpu().tolist() == expected_trim
    for index, output in enumerate(emitted.cpu().tolist()):
        count = accepted[index]
        assert output[:count + 1] == rows[index][1:count + 1] + [expected_bonus[index]]
    # The unlimited path must accept the full matching prefix independently of
    # the effective cap passed to the preceding invocation.
    unlimited, _, unlimited_trim = AcceptGreedy.execute(
        candidates=candidates, target_logits=logits, verify_num_draft_tokens=16,
        fused_argmax=fused)
    assert unlimited.cpu().tolist() == [raw for raw, _ in scenarios]
    assert unlimited_trim.cpu().tolist() == [0] * len(scenarios)
    return dict(scenarios=len(scenarios), prefix_dtype=str(prefix_dtype),
                fused_argmax=fused, maximum_commit=int(finalized.commit_lens.max().item()))


def ring_gate(torch):
    from sglang.kernels.ops.speculative.dspark.dspark_verify_window import (
        build_ring_commit_inject_layout,
    )

    # Include zero commit to prove rejected/padded hidden rows never receive
    # a writable ring address, and full sixteen-row commits across wraps.
    scenarios = [(prefix, count) for prefix in (127, 1023, 4095) for count in range(17)]
    slots = list(range(1, len(scenarios) + 1))
    device_slots = torch.tensor(slots, dtype=torch.int64, device="cuda")
    prefixes = torch.tensor([prefix for prefix, _ in scenarios], dtype=torch.int64, device="cuda")
    counts = torch.tensor([count for _, count in scenarios], dtype=torch.int32, device="cuda")
    offsets = torch.arange(16, dtype=torch.int64, device="cuda")
    for size in (128, 1024):
        layout = build_ring_commit_inject_layout(req_pool_indices=device_slots,
            prefix_lens=prefixes, block_pos_offsets=offsets, commit_lens=counts,
            stride=16, ring_stride=size)
        expected_positions, expected_locations = [], []
        for slot, (prefix, count) in zip(slots, scenarios):
            for column in range(16):
                position = prefix + column
                expected_positions.append(position)
                expected_locations.append(slot * size + position % size if column < count else -1)
        assert layout.positions.cpu().tolist() == expected_positions
        assert layout.swa_loc.cpu().tolist() == expected_locations
    return dict(scenarios=len(scenarios), ring_sizes=[128, 1024],
                scope="committed hidden injection address masks; no model weights")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = dict(passed=False, scope="native sixteen-column greedy accept and ring address contracts")
    try:
        import torch
        import sglang.kernels.ops.speculative.dspark.dspark_accept as accept_module
        import sglang.kernels.ops.speculative.dspark.dspark_verify_window as window_module

        assert torch.cuda.is_available(), "CUDA device required"
        report.update(torch=torch.__version__, cuda=torch.version.cuda,
                      gpu=torch.cuda.get_device_name(0), source_sha256={
                          Path(module.__file__).name: hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
                          for module in (accept_module, window_module)})
        report["accept"] = [accept_gate(torch, prefix_dtype=dtype)
                            for dtype in (torch.int32, torch.int64)]
        report["accept"].append(accept_gate(torch, prefix_dtype=torch.int64, fused=True))
        report["ring"] = ring_gate(torch)
        torch.cuda.synchronize()
        report["passed"] = True
    except Exception as error:
        report.update(error=repr(error), traceback=traceback.format_exc())
        raise
    finally:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report))


if __name__ == "__main__":
    main()
