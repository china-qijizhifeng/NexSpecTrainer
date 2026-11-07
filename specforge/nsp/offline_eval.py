"""Standalone model evaluation: no optimizer, scheduler, training loop or padded DP samples."""

from __future__ import annotations

from pathlib import Path

from specforge.nsp.metrics import MetricTotals, parse_bins
from specforge.nsp.store import atomic_json


def _number(value):
    return (
        float(value.detach().double().sum().cpu())
        if hasattr(value, "detach")
        else float(value)
    )


def output_terms(out, selected, strategy):
    terms = {}
    if "loss" in selected:
        if out.loss_terms is not None:
            terms["loss"] = tuple(_number(v) for v in out.loss_terms)
        elif "metric_losses" in out.metrics:
            denominators = [_number(v) for v in out.metrics["metric_loss_denoms"]]
            if len(set(denominators)) != 1:
                raise ValueError(
                    "per-position loss denominators differ; cannot pool this objective as one ratio"
                )
            terms["loss"] = (_number(out.loss) * denominators[0], denominators[0])
        else:
            raise ValueError(f"{strategy} does not expose additive objective terms")
    if "accuracy" in selected:
        selector_accuracy = "dflash2/selector/self_conditioned_marginal_accuracy"
        if selector_accuracy in out.ratio_metrics:
            terms["accuracy"] = tuple(
                _number(v) for v in out.ratio_metrics[selector_accuracy]
            )
        elif "acc" in out.ratio_metrics:
            terms["accuracy"] = tuple(_number(v) for v in out.ratio_metrics["acc"])
        elif "accuracy" in out.ratio_metrics:
            terms["accuracy"] = tuple(_number(v) for v in out.ratio_metrics["accuracy"])
        elif "acc_corrects" in out.metrics:
            terms["accuracy"] = (
                sum(_number(v) for v in out.metrics["acc_corrects"]),
                sum(_number(v) for v in out.metrics["acc_denoms"]),
            )
        else:
            raise ValueError(f"{strategy} does not expose exact accuracy counts")
    if "prefix-al" in selected:
        if "offline/prefix_al" not in out.ratio_metrics:
            raise ValueError(f"{strategy} did not produce prefix AL counts")
        terms["prefix-al"] = tuple(
            _number(v) for v in out.ratio_metrics["offline/prefix_al"]
        )
    return terms


def run_offline(spec, attempt):
    import torch
    from accelerate.utils import set_seed

    from specforge.application import resolve_run
    from specforge.config import Config
    from specforge.launch import _make_offline_eval_data_factory
    from specforge.nsp.eval_inputs import (
        checkpoint_metadata,
        feature_inventory,
        validate_model_identity,
    )
    from specforge.nsp.telemetry import emit_metrics
    from specforge.runtime.contracts import TrainBatch
    from specforge.training.assembly import build_model_bundle
    from specforge.training.strategies.base import StepContext

    set_seed(spec.seed)
    inventory = feature_inventory(spec.features)
    atomic_json(Path(attempt) / "data-manifest.json", inventory)
    cfg = Config.model_validate(spec.resolved_recipe)
    resolved = resolve_run(cfg)
    algorithm = resolved.algorithm
    bundle = build_model_bundle(cfg, algorithm=algorithm)
    bundle.model.eval()
    saved = checkpoint_metadata(cfg.model.draft_checkpoint_path)
    provenance = validate_model_identity(saved, bundle, attempt)
    global_step = saved.get("global_step", spec.global_step or 0)
    if spec.global_step is not None and spec.global_step != global_step:
        raise ValueError("global_step does not match the selected checkpoint")
    total_steps = spec.total_steps or saved.get("effective_total_steps")
    scheduled = (
        cfg.training.strategy == "domino"
        or cfg.training.dflash2_selector_warmup_ratio
        or cfg.training.dflash2_selector_ramp_ratio
    )
    if scheduled and total_steps is None:
        raise ValueError(
            "scheduled objective requires original total_steps in checkpoint or eval config"
        )
    strategy = algorithm.providers.step.build(
        bundle.model, target_head=bundle.target_head, **bundle.strategy_kwargs
    )
    loader = _make_offline_eval_data_factory(
        algorithm=algorithm,
        modality=cfg.model.input_modality,
        hidden_states_path=spec.features,
        run_id="nsp-eval",
        batch_size=1,
        max_len=cfg.data.max_length,
        ttt_length=cfg.training.ttt_length,
        use_usp_preprocess=False,
        dataloader_num_workers=0,
    )()
    normalize = loader.per_sample_transform

    def checked_normalize(raw):
        if raw["input_ids"].numel() > cfg.data.max_length:
            raise ValueError(
                "evaluation record exceeds data.max_length; increase it instead of silently truncating"
            )
        return normalize(raw)

    loader.per_sample_transform = checked_normalize
    selected = [m for m in spec.metrics if m != "al"]
    edges = parse_bins(spec.ctx_bins) if spec.group_by else None
    totals = MetricTotals(selected, edges)
    count = 0
    anchor_count = 0
    full_anchor_count = 0
    has_selector = getattr(bundle.draft_model, "candidate_selector", None) is not None
    ctx = StepContext(
        global_step=global_step,
        total_steps=total_steps,
        collect_detailed_metrics=(
            "prefix-al" in selected or ("accuracy" in selected and has_selector)
        ),
    )
    anchor_model = algorithm.name in {"dspark", "deepseek_v41", "dflash", "dflash2"}
    with torch.inference_mode():
        for batch in loader:
            count += 1
            if anchor_model:
                mask = batch.tensors["loss_mask"][0].bool()
                positions = torch.where(mask[:-1] & mask[1:])[0]
                # Exclude the supplied anchor in all accuracy counts. Context
                # length includes the anchor, so an anchor at index 0 has ctx=1.
                partitions = (
                    [(None, positions)]
                    if edges is None
                    else [
                        (i, positions[((positions + 1) >= lo) & ((positions + 1) < hi)])
                        for i, (lo, hi) in enumerate(zip(edges, edges[1:]))
                    ]
                )
                for index, anchors in partitions:
                    for chunk in anchors.split(spec.anchor_chunk_size):
                        if not chunk.numel():
                            continue
                        tensors = {
                            **batch.tensors,
                            "eval_anchor_positions": chunk.unsqueeze(0),
                        }
                        output = strategy.forward_loss(
                            TrainBatch(
                                batch.sample_ids,
                                batch.strategy,
                                tensors,
                                batch.metadata,
                            ),
                            ctx,
                        )
                        totals.add(
                            output_terms(output, selected, algorithm.name),
                            bin_index=index,
                        )
                        anchor_count += chunk.numel()
                        if "offline/prefix_al" in output.ratio_metrics:
                            full_anchor_count += int(
                                _number(output.ratio_metrics["offline/prefix_al"][1])
                            )
            else:
                output = strategy.forward_loss(batch, ctx)
                totals.add(output_terms(output, selected, algorithm.name))
            emit_metrics({"eval/samples": count, "eval/anchors": anchor_count}, count)
    if count == 0:
        raise ValueError("evaluation features contain no samples")
    if inventory != feature_inventory(spec.features):
        raise ValueError("evaluation features changed during the run")
    result = {
        "model_provenance": provenance,
        "data_fingerprint": inventory["fingerprint"],
        "global_step": global_step,
        "total_steps": total_steps,
        "schema_version": 1,
        "complete": True,
        "stage": "offline",
        "checkpoint": spec.checkpoint,
        "algorithm": algorithm.name,
        "reference": (
            "target-argmax" if algorithm.name == "eagle3" else "recorded-token"
        ),
        "accuracy_path": (
            "selector-self-conditioned"
            if getattr(bundle.draft_model, "candidate_selector", None) is not None
            else "unary-marginal"
        ),
        "prefix_path": "reference prefix through first mismatch",
        "ctx_basis": "record_anchor_prefix" if edges else None,
        "prefix_al_definition": "1 + leading matches; full-horizon anchors only",
        "sample_count": count,
        "anchor_count": anchor_count,
        "full_horizon_anchor_count": (
            full_anchor_count if "prefix-al" in selected else None
        ),
        "objective": {
            "strategy": algorithm.name,
            "training": cfg.training.model_dump(mode="json"),
        },
        **totals.result(),
    }
    if any(metric["value"] is None for metric in result["overall"].values()):
        result["complete"] = False
        atomic_json(Path(attempt) / "offline-results.json", result)
        raise ValueError("one or more requested metrics have zero valid observations")
    atomic_json(Path(attempt) / "offline-results.json", result)
    return result
