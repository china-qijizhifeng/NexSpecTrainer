"""Reuse DSpark's losses with the native pre-norm confidence features."""

from specforge.algorithms.common.dflash_family_model import OnlineDSparkModel


class OnlineDeepSeekV41Model(OnlineDSparkModel):
    def _attention_mask(self, anchor_positions, block_keep_mask, sequence_length):
        # The native model gathers a private context window for each block.
        return None

    def _project_draft_logits(self, hidden):
        return self._project_logits(self.draft_model.normalize_for_logits(hidden))
