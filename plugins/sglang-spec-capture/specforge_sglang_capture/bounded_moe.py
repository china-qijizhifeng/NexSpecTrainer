"""Experimental bounded MXFP4 MoE calls, inspired by Nextron's fixed workspace.

This is NOT a Nextron BF16 kernel port. Keep the pinned SGLang routing,
MXFP4 weights/scales, activation and EP collective; bound only the local
CUTLASS invocation's token count. Full attention/capture remain unchunked.
"""

import functools

_limit = 0
_original = None


def configure(limit):
    global _limit, _original
    if limit not in (0, 8192, 16384):
        raise ValueError(f"Unsupported MoE token bound: {limit}")
    from sglang.srt.layers.moe.moe_runner import flashinfer_cutlass as module

    if _original is None:
        _original = module._fused_experts_flashinfer_mxfp4_cutlass
        module._fused_experts_flashinfer_mxfp4_cutlass = functools.wraps(_original)(
            _bounded
        )
    _limit = limit


def _bounded(dispatch, quant, config):
    import torch
    from sglang.srt.layers.moe.token_dispatcher.standard import StandardCombineInput
    from sglang.srt.layers.moe.topk import StandardTopKOutput, TopKOutputChecker

    x = dispatch.hidden_states
    if not _limit or x.shape[0] <= _limit:
        return _original(dispatch, quant, config)
    if (
        x.dtype != torch.bfloat16
        or x.shape[1] != 5120
        or dispatch.hidden_states_scale is not None
        or dispatch.hidden_states_pre_quant is not None
        or quant.mxfp4_weight_global_scale is not None
        or quant.w13_humming_residual_scale is not None
        or quant.w2_humming_residual_scale is not None
        or quant.humming_fc2_act_scale is not None
        or quant.padded_hidden not in (None, 5120)
        or torch.cuda.is_current_stream_capturing()
    ):
        raise ValueError("Bounded MoE requires eager V4.1 SM90 W4A16 dispatch")
    topk = dispatch.topk_output
    if TopKOutputChecker.format_is_bypassed(topk):
        topk = topk.to_standard()
    if not TopKOutputChecker.format_is_standard(topk) or topk.topk_ids.shape != (
        x.shape[0],
        6,
    ):
        raise ValueError("Bounded MoE requires explicit V4.1 top-6 routes")
    # One full output, at most one bounded kernel workspace and chunk output.
    # All EP all-reduces remain outside this fused function, in their old order.
    output = torch.empty_like(x)
    for start in range(0, x.shape[0], _limit):
        end = min(start + _limit, x.shape[0])
        routes = StandardTopKOutput(
            topk.topk_weights[start:end],
            topk.topk_ids[start:end],
            None if topk.router_logits is None else topk.router_logits[start:end],
        )
        part = dispatch._replace(hidden_states=x[start:end], topk_output=routes)
        output[start:end].copy_(_original(part, quant, config).hidden_states)
    return StandardCombineInput(hidden_states=output)
