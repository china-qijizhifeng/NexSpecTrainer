"""Opt-in TP8-compatible arithmetic for eager V4.1 CP8 capture on Hopper.

CP changes GEMM and collective geometry as well as attention kernels. Small
BF16 differences can change sparse keys or expert routes and amplify through
40 layers. This mode preserves the TP reference's operation order, using only
the current request's operands. It trades communication/workspace for parity;
it is not a long-context performance optimization.
"""

from contextvars import ContextVar
from functools import wraps

import torch

_BATCH = ContextVar("northgate_cp_numerics_batch", default=None)


def sequence_order(rank_major, size):
    rows = rank_major.shape[0] // size
    return (
        rank_major.reshape(size, rows, *rank_major.shape[1:])
        .transpose(0, 1)
        .flatten(0, 1)
        .contiguous()
    )


def local_rows(sequence, rank, size, rows):
    result = sequence[rank::size].contiguous()
    if result.shape[0] == rows:
        return result
    padded = result.new_zeros((rows, *result.shape[1:]))
    padded[: result.shape[0]] = result
    return padded


def gather_sequence(x, batch, parallel):
    """Equalize collective counts even when the indexer drops local pad rows."""
    metadata = batch.attn_cp_metadata
    rows = max(metadata.per_rank_actual_token)
    if x.shape[0] > rows:
        raise ValueError("CP operand exceeds its physical shard size")
    if x.shape[0] < rows:
        padded = x.new_zeros((rows, *x.shape[1:]))
        padded[: x.shape[0]] = x
        x = padded
    gathered = parallel.attn_cp_group.all_gather(x.contiguous(), dim=0)
    return sequence_order(gathered, parallel.attn_cp_size)[
        : int(metadata.total_seq_lens)
    ].contiguous()


def combine_norm_prefill(x, pre, weight, eps):
    """Keep the four-warp row reduction independent of CP-local row count."""
    from sglang.kernels.ops.layernorm.hc_combine_norm import _hc_combine_norm_prefill

    y = x.new_empty((x.shape[0], 5120))
    _hc_combine_norm_prefill[(x.shape[0],)](
        x.flatten(1), pre, weight, y, x.stride(0), pre.stride(0), eps, num_warps=4
    )
    return y


def _keys(cache, layout, page_size, indices, lengths):
    """Dequantize each live slot once, retaining the caller's key order."""
    from sglang.srt.layers.attention.deepseek_v4_backend import (
        dequantize_k_cache_paged_dsv4,
    )

    cols = torch.arange(indices.shape[1], device=indices.device)[None, :]
    keep = (cols < lengths[:, None]) & (indices >= 0)
    counts = keep.sum(1).to(torch.int32)
    if not torch.equal(keep, cols < counts[:, None]):
        raise ValueError("CP sparse indices must have a contiguous valid prefix")
    slots = torch.unique(indices[keep], sorted=True)
    if slots.numel() == 0:
        return (
            torch.empty((0, 1, 512), dtype=torch.bfloat16, device=indices.device),
            torch.full_like(indices, -1),
            counts,
        )
    kv = dequantize_k_cache_paged_dsv4(cache, slots, page_size=page_size, layout=layout)
    mapped = torch.searchsorted(slots, indices.clamp_min(0)).to(torch.int32)
    return kv, mapped.masked_fill(~keep, -1), counts


def sparse_attention(backend, q, layer_id, ratio, sink):
    from sglang.kernels.ops.attention.flash_mla_interface import flash_mla_sparse_fwd

    core, pool = backend.forward_metadata.core_attn_metadata, backend.token_to_kv_pool
    rows = q.shape[0]

    def metadata(indices, lengths):
        indices, lengths = indices[:rows], lengths[:rows]
        if indices.shape[0] < rows:
            padding = rows - indices.shape[0]
            indices = torch.cat(
                (indices, indices.new_full((padding, indices.shape[1]), -1))
            )
            lengths = torch.cat((lengths, lengths.new_zeros(padding)))
        return indices, lengths

    swa, si, sn = _keys(
        pool.get_swa_key_buffer_radix(layer_id),
        pool.get_swa_key_layout(),
        pool.get_swa_key_page_size(),
        *metadata(core.swa_page_indices, core.swa_topk_lengths),
    )
    # Paged SWA is newest first; TP sparse attention reduces oldest first.
    col = torch.arange(si.shape[1], device=q.device)[None, :].expand(rows, -1)
    si = si.gather(1, (sn[:, None] - 1 - col).clamp_min(0).long())
    si.masked_fill_(col >= sn[:, None], -1)
    kv, indices, lengths = swa, si, sn
    if ratio:
        extra, ei, en = _keys(
            pool.get_extra_key_buffer(layer_id),
            pool.get_extra_key_layout(layer_id),
            pool.get_extra_key_page_size(layer_id),
            *metadata(core.sparse_page_indices(ratio), core.sparse_topk_lengths(ratio)),
        )
        col = torch.arange(ei.shape[1] + si.shape[1], device=q.device)[None, :].expand(
            rows, -1
        )
        extra_ids = ei.gather(1, col.clamp_max(ei.shape[1] - 1))
        swa_ids = si.gather(
            1, (col - en[:, None]).clamp(0, si.shape[1] - 1).long()
        ) + len(extra)
        lengths = en + sn
        indices = torch.where(col < en[:, None], extra_ids, swa_ids)
        indices.masked_fill_(col >= lengths[:, None], -1)
        kv = torch.cat((extra, swa))
    return flash_mla_sparse_fwd(
        q=q.squeeze(1) if q.ndim == 4 else q,
        kv=kv,
        indices=indices.unsqueeze(1),
        sm_scale=backend.softmax_scale,
        d_v=backend.head_dim_v,
        attn_sink=sink,
        topk_length=lengths,
    )[0]


def install(model):
    """Return an undo callback; validate the supported deployment before patching."""
    from sglang.srt.layers.attention.deepseek_v4_backend import DeepseekV4AttnBackend
    from sglang.srt.layers.attention.dsv4.dsv41_sparse import DeepseekV41Indexer
    from sglang.srt.layers.cp.utils import is_cp_active
    from sglang.srt.models import deepseek_v4
    from sglang.srt.models import deepseek_v4_mhc as mhc
    from sglang.srt.models.deepseek_common.moe_transport import MoeRowTransport
    from sglang.srt.runtime_context import get_parallel, get_platform

    parallel = get_parallel()
    if (parallel.tp_size, parallel.attn_cp_size, parallel.attn_tp_size) != (8, 8, 1):
        raise ValueError("CP numerical parity mode requires TP8/CP8/attention-TP1")
    if not get_platform().is_sm90:
        raise ValueError("CP numerical parity mode is validated only on Hopper")
    shards = {}
    for module in model.modules():
        if isinstance(module, deepseek_v4.MQALayer):
            linear = module.wo_b
            if (
                tuple(linear.weight.shape) != (5120, 8192)
                or linear.quant_method.weight_block_size != [32, 32]
                or linear.bias is not None
            ):
                raise ValueError("unsupported V4.1 wo_b layout")
            start = parallel.tp_rank * 1024
            shards[id(module)] = (
                linear.weight[:, start : start + 1024].contiguous(),
                linear.weight_scale_inv[
                    :, start // 32 : (start + 1024) // 32
                ].contiguous(),
            )
    if len(shards) != 40:
        raise ValueError(f"expected 40 V4.1 attention layers, got {len(shards)}")
    undo = []

    def patch(owner, name, replacement):
        original = getattr(owner, name)
        undo.append((owner, name, original))
        setattr(owner, name, wraps(original)(replacement))
        return original

    def active():
        batch = _BATCH.get()
        return batch if batch is not None and is_cp_active(batch) else None

    def model_forward(self, input_ids, positions, forward_batch, *args, **kwargs):
        token = _BATCH.set(forward_batch)
        try:
            return original_model(
                self, input_ids, positions, forward_batch, *args, **kwargs
            )
        finally:
            _BATCH.reset(token)

    original_model = patch(deepseek_v4.DeepseekV4Model, "forward", model_forward)

    def attention(
        self,
        q,
        k,
        v,
        layer,
        forward_batch,
        compress_ratio,
        save_kv_cache=True,
        attn_sink=None,
        cos_sin_cache=None,
        **kwargs,
    ):
        if active() is None:
            return original_attention(
                self,
                q,
                k,
                v,
                layer,
                forward_batch,
                compress_ratio,
                save_kv_cache,
                attn_sink,
                cos_sin_cache,
                **kwargs,
            )
        if cos_sin_cache is not None or compress_ratio not in (0, 1, 2):
            raise ValueError("unsupported CP numerical parity attention path")
        if save_kv_cache:
            self.store_cache(layer.layer_id, k, forward_batch)
        return sparse_attention(self, q, layer.layer_id, compress_ratio, attn_sink)

    original_attention = patch(DeepseekV4AttnBackend, "_forward_attention", attention)

    def projection(self, o, defer_all_reduce=False):
        batch = active()
        if batch is None:
            return original_projection(self, o, defer_all_reduce)
        x = o.flatten(1)
        full = gather_sequence(x, batch, parallel)
        start = parallel.tp_rank * 1024
        weight, scale = shards[id(self)]
        partial = self.wo_b.quant_method.w8a8_block_fp8_linear(
            input=full[:, start : start + 1024].contiguous(),
            weight=weight,
            block_size=[32, 32],
            weight_scale=scale,
            input_scale=None,
            bias=None,
        )
        reduced = parallel.tp_group.all_reduce(partial)
        return local_rows(reduced, parallel.attn_cp_rank, 8, x.shape[0])

    original_projection = patch(deepseek_v4.MQALayer, "_project_wo_b", projection)

    def weights(self, x):
        batch = active()
        if batch is None:
            return original_weights(self, x)
        value = original_weights(self, gather_sequence(x, batch, parallel))
        return local_rows(value, parallel.attn_cp_rank, 8, x.shape[0])

    original_weights = patch(DeepseekV41Indexer, "head_weights_raw", weights)

    from sglang.kernels.ops.layernorm import mhc as mhc_kernels

    from ops.northgate_capture.cp_numerics_kernels import mix_stats_bf16x3

    def stats(*args, **kwargs):
        if active() is None:
            return original_stats(*args, **kwargs)
        return mix_stats_bf16x3(*args, **kwargs)

    original_stats = patch(mhc_kernels, "hc_mix_stats_sinkhorn_bf16x3", stats)

    def mix_stats(hc, x):
        batch = active()
        if batch is None or int(batch.attn_cp_metadata.total_seq_lens) <= 65536:
            return original_mix_stats(hc, x)
        # Hopper TP uses this path above 64K. CP-local row count must not
        # select the BF16x3 fused projection/Sinkhorn path instead.
        cfg = hc.cfg
        mixes = mhc_kernels.hc_mix_stats(x.flatten(1), hc.fn, cfg.rms_eps).unsqueeze(1)
        coefficients = mhc_kernels.hc_split_sinkhorn(
            mixes, hc.scale, hc.base, cfg.mult, cfg.sinkhorn_iters, cfg.eps
        )
        return tuple(value.squeeze(1) for value in coefficients)

    original_mix_stats = patch(mhc, "_mix_stats_impl", mix_stats)

    def combine(hc, state, quantized):
        batch = active()
        if batch is None or quantized is not None or state.pre is None:
            return original_combine(hc, state, quantized)
        from sglang.kernels.ops.layernorm.hc_combine_norm import hc_combine_norm
        from sglang.kernels.ops.layernorm.mhc import mhc_combine

        n = int(batch.attn_cp_metadata.total_seq_lens)
        cfg, norm = hc.cfg, hc.norm
        x, pre = state.residual, state.pre
        if cfg.mult != 4 or cfg.hidden != 5120:
            raise ValueError("unsupported mHC shape")
        if 4096 <= n <= 65536:
            return combine_norm_prefill(x, pre, norm.weight, norm.variance_epsilon)
        if 0 < n <= 96:
            return hc_combine_norm(
                x.flatten(1), pre, norm.weight, norm.variance_epsilon
            )
        return norm(mhc_combine(x.flatten(1), pre, cfg.mult, x.dtype))

    original_combine = patch(mhc, "_combine", combine)

    def returned(self, hidden_states):
        if not self._use_cp or active() is None:
            return original_returned(self, hidden_states)
        if not self.cp_return_contiguous:
            raise ValueError("CP parity requires contiguous MoE return")
        rows = hidden_states.shape[0] // 8
        sequence = sequence_order(hidden_states, 8)
        n = int(self.batch.attn_cp_metadata.total_seq_lens)
        reduced = parallel.tp_group.all_reduce(sequence[:n].contiguous())
        return local_rows(reduced, parallel.attn_cp_rank, 8, rows)

    original_returned = patch(MoeRowTransport, "return_output", returned)

    def restore():
        for owner, name, original in reversed(undo):
            setattr(owner, name, original)

    return restore
