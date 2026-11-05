"""Native exhaustive anchors with independent FSDP forward/backward graphs."""

import torch

from specforge.algorithms.common.dflash_family_model import (
    compute_walk_accepted_length_terms,
)
from specforge.training.strategies.base import DSparkTrainStrategy, StepOutput


class NativeStrategy(DSparkTrainStrategy):
    name = "deepseek_v41"

    def stream_backward(self, batch, backward, ctx=None):
        model = self.dspark_model
        if model.anchor_sampling != "all":
            return None
        self.validate_batch(batch)
        device = self._device()
        inputs = {
            name: batch.tensors[name].to(device, non_blocking=True)
            for name in self.required_features
        }
        positions, denominator, counts, valid = model.prepare_anchor_chunks(
            inputs["input_ids"], inputs["loss_mask"]
        )
        detailed = ctx.collect_detailed_metrics if ctx is not None else True
        ratios = {}
        accepted = []
        total_loss = denominator.new_zeros(())
        # Root FSDP casts floating tensor inputs to its parameter dtype. Keep
        # the FP32 normalizer exact across that boundary, with one host sync per
        # microbatch rather than one per chunk.
        denominator_value = denominator.item()
        # Eligibility/exclusion describe the whole window. Count selection and
        # target exposure from the chunks actually executed, not only the plan.
        observed = {
            name: torch.zeros_like(value)
            for name, value in counts.items()
            if name not in ("anchors/eligible", "anchors/excluded")
        }
        width = model.anchor_forward_chunk_blocks
        for start in range(0, positions.shape[1], width):
            # This must call the root wrapper, not its unwrapped forward. The
            # callback finishes backward/reduce-scatter before the next call.
            loss, _, metrics = model(
                **inputs,
                anchor_positions=positions[:, start : start + width],
                global_loss_den=denominator_value,
                collect_detailed_metrics=detailed,
            )
            backward(loss)
            total_loss = total_loss + loss.detach()
            for name in observed:
                observed[name] = observed[name] + metrics["sum_metrics"][name]
            for name, (numerator, divisor) in metrics["ratio_metrics"].items():
                previous = ratios.get(name)
                ratios[name] = (
                    (numerator, divisor)
                    if previous is None
                    else (previous[0] + numerator, previous[1] + divisor)
                )
            if detailed:
                accepted.append(metrics["walk_accepted"])
            del loss, metrics
        if detailed:
            ratios["dspark/hard_label/walk_accepted_length"] = (
                compute_walk_accepted_length_terms(
                    torch.cat(accepted, dim=1), positions, valid
                )
            )
        correct, valid_targets = ratios["acc"]
        counts.update(observed)
        return StepOutput(
            loss=total_loss,
            metrics={
                "accuracy": correct / valid_targets.clamp_min(1),
                "accuracy_denom": valid_targets,
            },
            ratio_metrics=ratios,
            sum_metrics=counts,
        )
