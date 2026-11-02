"""Native checkpoint identity and kernel preflight."""

import importlib

import torch
from packaging.version import Version


def validate_target(draft, target):
    raw = target.to_dict() if hasattr(target, "to_dict") else target
    if raw.get("model_type") != "deepseek_v41":
        raise ValueError(
            "deepseek_v41 strategy requires the native DeepSeek V4.1 target"
        )
    text = raw["text_config"]
    names = {
        "num_hidden_layers": "num_nextn_predict_layers",
        "n_routed_experts": "dspark_n_routed_experts",
        "num_experts_per_tok": "dspark_num_experts_per_tok",
        "block_size": "dspark_block_size",
        "mask_token_id": "dspark_noise_token_id",
        "target_layer_ids": "dspark_target_layer_ids",
        "markov_rank": "dspark_markov_rank",
    }
    for key in (
        "hidden_size",
        "vocab_size",
        "num_attention_heads",
        "head_dim",
        "qk_rope_head_dim",
        "q_lora_rank",
        "o_lora_rank",
        "o_groups",
        "moe_intermediate_size",
        "routed_scaling_factor",
        "hc_mult",
        "hc_sinkhorn_iters",
        "hc_eps",
        "rms_norm_eps",
        "rope_theta",
        "sliding_window",
        "swiglu_limit",
    ):
        names[key] = key
    for key, source in names.items():
        if getattr(draft, key) != text[source]:
            raise ValueError(
                f"native draft {key}={getattr(draft, key)!r} differs from target {source}={text[source]!r}"
            )


def preflight():
    from .kernels.elementwise import require_cuda

    if not torch.cuda.is_available():
        raise RuntimeError(
            "Native V4.1 requires CUDA Hopper; there is no eager fallback"
        )
    require_cuda(torch.empty(0, device="cuda"))
    apis = {
        "deep_gemm": (
            "m_grouped_bf16_gemm_nt_contiguous",
            "m_grouped_bf16_gemm_nn_contiguous",
            "k_grouped_bf16_gemm_tn_contiguous",
        ),
        "tile_kernels.quant": (
            "norm_forward",
            "norm_backward",
            "swiglu_forward",
            "swiglu_backward",
        ),
        "tile_kernels.mhc": (
            "mhc_sinkhorn_bwd",
            "mhc_pre_split_mixes_bwd",
            "mhc_pre_apply_mix_bwd",
            "mhc_post_bwd",
        ),
        "tile_kernels.moe": ("moe_topk_gate_forward", "moe_topk_gate_backward"),
        "tile_kernels.transform": ("apply_rotary",),
        # TE registers its extension search path when its PyTorch package loads.
        "transformer_engine.pytorch": (),
        "transformer_engine_torch": ("rmsnorm_fwd", "rmsnorm_bwd"),
        "megatron.core.transformer.moe.moe_utils": ("permute", "unpermute"),
        "megatron.core.transformer.experimental_attention_variant.csa_utils.packed_sparse_attention": (
            "csa_sparse_attn",
        ),
        "flash_mla": ("flash_mla_sparse_fwd",),
        "cudnn.deepseek_sparse_attention": ("sparse_attention_backward_wrapper",),
    }
    for name, required in apis.items():
        module = importlib.import_module(name)
        if name == "deep_gemm" and Version(module.__version__) < Version("2.8.2"):
            raise RuntimeError(
                "Native V4.1 uses Hopper prefix-sum grouped GEMM layouts; "
                "build the pinned DeepGEMM 2.8.2 against the existing Torch ABI. "
                "DeepGEMM 2.3 exposes these symbols but rejects this SM90 layout."
            )
        for symbol in required:
            if not callable(getattr(module, symbol, None)):
                raise TypeError(
                    f"{name}.{symbol} is required; install the pinned native training stack"
                )
