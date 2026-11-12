"""Native V4.1 adapter. Target/draft KV commits stay owned by SGLang.

The separate token-ID pool follows the same pool slots. Prefill reseeds complete
visible prefixes (including cache hits/retractions), and decode writes only
the current anchor and then the accepted verify inputs. Slot reuse therefore
cannot expose an earlier request's tokens.
"""

import hashlib
import json
from pathlib import Path

import torch

from .kernels import write_history
from .ngram import scan
from .selection import estimate_ngram, select_chains, verify_lengths


class HybridRuntime:
    def __init__(self, config, req_pool, device, model_path):
        self.config = config
        if config.calibrator:
            provenance = json.loads(Path(config.calibrator).read_text())["provenance"]
            for key, name in (("target", "config.json"), ("tokenizer", "tokenizer.json")):
                actual = hashlib.sha256((Path(model_path) / name).read_bytes()).hexdigest()
                if provenance.get(key) != actual:
                    raise ValueError(f"Hybrid calibrator {key} fingerprint differs from native serving model")
            if provenance.get("temperature") != 0:
                raise ValueError("Hybrid v1 calibrator must use greedy target trajectories")
        self.history = torch.empty(req_pool.req_to_token.shape, dtype=torch.int32, device=device)
        self.lengths = torch.zeros(self.history.shape[0], dtype=torch.int32, device=device)
        self.coefficients = {key: torch.as_tensor(value, dtype=torch.float32, device=device)
                             for key, value in config.coefficients().items()}
        self.last_selection = None

    def prefill(self, batch):
        # These are scheduler-owned CPU prefixes, not a synchronization of
        # model output. Seeding a cached prefix is necessary on request reuse.
        from sglang.srt.utils import async_h2d

        if batch.seq_lens_cpu is None:
            raise ValueError("Hybrid prefill requires scheduler CPU prefix lengths")
        for req, size in zip(batch.reqs, batch.seq_lens_cpu.tolist(), strict=True):
            slot = req.kv.req_pool_idx
            ids = req.get_fill_ids()[:size]
            if len(ids) != size or not 0 <= slot < self.history.shape[0] or size > self.history.shape[1]:
                raise ValueError("Hybrid prefill prefix/slot exceeds token history")
            self.history[slot, :size].copy_(async_h2d(ids, dtype=torch.int32, device=self.history.device))
            self.lengths[slot].fill_(size)

    def propose(self, batch, prefix_lens, anchor, proposal, slot_price=None):
        from sglang.srt.speculative.dspark_components.dspark_draft import DraftBlockResult, DraftProposal

        block = proposal.draft_block
        if batch.sampling_info is not None and not batch.sampling_info.is_all_greedy:
            raise ValueError("Hybrid Spark v1 currently requires target temperature=0; sampling calibration is pending")
        if not block.greedy_draft:
            raise ValueError("Hybrid selection requires a deterministic greedy MTP proposal")
        slots = batch.req_pool_indices
        anchors = anchor.reshape(-1, 1).contiguous()
        write_history(self.history, self.lengths, slots, prefix_lens, anchors,
                      torch.ones_like(prefix_lens, dtype=torch.int32))
        ngram = scan(self.history, self.lengths, slots, min_match=self.config.min_match,
                     max_match=self.config.max_match, max_draft=self.config.max_draft)
        if proposal.confidence is None:
            raise ValueError("native Hybrid Spark requires the original MTP confidence head")
        if self.config.extra_budget_per_request is not None:
            slot_price = None  # Fixed-budget runs use their configured price.
        selected = select_chains(block.draft_tokens, proposal.confidence, ngram,
                                 estimate_ngram(ngram, self.coefficients),
                                 slot_price=self.config.slot_price if slot_price is None else slot_price,
                                 contains=self.config.contains, mode=self.config.mode)
        self.last_selection = selected
        new_block = DraftBlockResult(draft_tokens=selected.tokens, corrected_logits=None,
                                     greedy_mask=block.greedy_mask, temperatures=block.temperatures,
                                     greedy_draft=True)
        # verify_ids from the folded native sampler has only six columns. The
        # new chain must rebuild it; native draft graph capture remains usable.
        return DraftProposal(draft_block_ids=proposal.draft_block_ids, draft_block=new_block,
                             confidence=selected.survival, confidence_raw=None, folded=False)

    def cap_layout(self, layout, *, budget=None):
        from sglang.srt.speculative.ragged_verify import RaggedVerifyLayout

        selected = self.last_selection
        budget = self.config.extra_budget(selected.lengths.shape[0], budget)
        lens = verify_lengths(selected.survival, selected.lengths, budget)
        if layout is not None:
            lens = torch.minimum(lens, layout.verify_lens)
            return RaggedVerifyLayout.from_verify_lens_device(verify_lens=lens,
                                                               graph_num_tokens=layout.graph_num_tokens)
        return RaggedVerifyLayout.from_verify_lens_device(verify_lens=lens,
                                                           graph_num_tokens=lens.shape[0] * self.config.verify_width)

    def commit(self, batch, prefix_lens, verify_ids, commit_lens):
        write_history(self.history, self.lengths, batch.req_pool_indices, prefix_lens,
                      verify_ids, commit_lens)
