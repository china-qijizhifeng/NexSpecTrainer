"""Reuse DSpark's losses with the native pre-norm confidence features."""

import torch
import torch.distributed as dist

from specforge.algorithms.common.anchor_coverage import (
    all_anchor_positions,
    coverage_counts,
)
from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel


class OnlineDeepSeekV41Model(OnlineDSparkModel):
    def __init__(
        self, *args, anchor_sampling="uniform", anchor_forward_chunk_blocks=32, **kwargs
    ):
        super().__init__(*args, **kwargs)
        if (
            anchor_sampling not in ("uniform", "all")
            or anchor_forward_chunk_blocks <= 0
        ):
            raise ValueError("invalid native anchor coverage configuration")
        self.anchor_sampling = anchor_sampling
        self.anchor_forward_chunk_blocks = anchor_forward_chunk_blocks

    @property
    def fsdp_sync_each_microstep(self):
        # Retain only reduced gradient shards between chunks and microbatches;
        # the optimizer still updates at the same accumulation boundary.
        return self.anchor_sampling == "all"

    @property
    def fsdp_prefetch(self):
        # Many recomputed chunks revisit each expert group. Do not overlap a
        # second full parameter group with the current group's full gradients.
        return self.anchor_sampling != "all"

    def _attention_mask(self, anchor_positions, block_keep_mask, sequence_length):
        # The native model gathers a private context window for each block.
        return None

    def _project_draft_logits(self, hidden):
        return self._project_logits(self.draft_model.normalize_for_logits(hidden))

    @torch.no_grad()
    def prepare_anchor_chunks(self, input_ids, loss_mask):
        """Plan all ranks' chunks and one denominator without a parameter graph."""
        positions = all_anchor_positions(loss_mask)
        _, mask, _ = self._build_dspark_labels_and_mask(
            input_ids, loss_mask, positions.clamp_min(0), positions >= 0
        )
        denominator = self._dspark_loss_weight_mask(mask).sum()
        if dist.is_available() and dist.is_initialized():
            dist.all_reduce(denominator)
        torch._assert_async(
            denominator > 0, "DSpark objective has no supervised target tokens"
        )
        return (
            positions,
            denominator,
            coverage_counts(positions, mask, loss_mask),
            mask.any(dim=-1),
        )

    def forward(
        self,
        input_ids,
        hidden_states,
        loss_mask,
        target_last_hidden_states=None,
        max_valid_anchors=None,
        collect_detailed_metrics=True,
        anchor_positions=None,
        global_loss_den=None,
    ):
        if self.anchor_sampling == "uniform":
            return super().forward(
                input_ids,
                hidden_states,
                loss_mask,
                target_last_hidden_states,
                max_valid_anchors,
                collect_detailed_metrics,
            )
        positions = (
            all_anchor_positions(loss_mask)
            if anchor_positions is None
            else anchor_positions
        )
        keep = positions >= 0
        target_ids, eval_mask, indices = self._build_dspark_labels_and_mask(
            input_ids,
            loss_mask,
            positions.clamp_min(0),
            keep,
        )
        previous = torch.cat(
            [
                input_ids.gather(1, positions.clamp_min(0)).unsqueeze(-1),
                target_ids[:, :, :-1],
            ],
            dim=-1,
        )

        def build_hidden(chunk_positions):
            return self._forward_draft_blocks(
                input_ids,
                hidden_states,
                loss_mask,
                anchor_selection=(chunk_positions.clamp_min(0), chunk_positions >= 0),
            )[2]

        loss, metrics = self._compute_dspark_loss(
            output_hidden=None,
            target_ids=target_ids,
            eval_mask=eval_mask,
            prev_token_ids=previous,
            safe_label_indices=indices,
            target_last_hidden_states=target_last_hidden_states,
            anchor_positions=positions,
            collect_detailed_metrics=collect_detailed_metrics,
            hidden_builder=build_hidden,
            forward_chunk_blocks=self.anchor_forward_chunk_blocks,
            global_loss_den=global_loss_den,
            return_walk_accepted=anchor_positions is not None,
        )
        metrics["sum_metrics"] = coverage_counts(positions, eval_mask, loss_mask)
        return loss, metrics.pop("accuracy"), metrics
