"""Exhaustive legal anchors with identical distributed forward schedules."""

import torch
import torch.distributed as dist


def all_anchor_positions(loss_mask):
    """Enumerate each legal anchor once, padding to the largest rank's count.

    -1 is a masked slot, never a real position. Even an empty rank executes
    every chunk so nested FSDP collectives have identical schedules.
    """
    valid = (loss_mask[:, :-1] > 0.5) & (loss_mask[:, 1:] > 0.5)
    count = valid.sum(dim=1).max()
    if dist.is_available() and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.MAX)
    width = max(1, int(count.item()))
    positions = torch.arange(valid.shape[1], device=valid.device).expand_as(valid)
    positions = positions.masked_fill(~valid, loss_mask.shape[1]).sort(dim=1).values
    positions = positions[:, :width]
    positions = positions.masked_fill(positions >= loss_mask.shape[1], -1)
    return torch.nn.functional.pad(positions, (0, width - positions.shape[1]), value=-1)


def coverage_counts(positions, eval_mask, loss_mask):
    """Additive counts; repeated target exposures are not unique token counts."""
    counts = {
        "anchors/eligible": (
            (loss_mask[:, :-1] > 0.5) & (loss_mask[:, 1:] > 0.5)
        ).sum(),
        "anchors/selected": (positions >= 0).sum(),
        "anchors/padding": (positions < 0).sum(),
        "anchors/excluded": (
            (loss_mask[:, :-1] <= 0.5) | (loss_mask[:, 1:] <= 0.5)
        ).sum(),
        "anchors/duplicates": positions.new_zeros(()),
        "supervision/target_exposures": eval_mask.sum(),
    }
    for k in range(eval_mask.shape[-1]):
        counts[f"supervision/position_{k + 1}"] = eval_mask[..., k].sum()
    return {name: value.detach() for name, value in counts.items()}
