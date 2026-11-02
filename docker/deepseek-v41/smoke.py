"""CPU image import acceptance; GPU numerical acceptance is separate."""

import importlib
import json
import sys
from importlib import metadata
from pathlib import Path

import torch

importlib.import_module("transformer_engine.pytorch")  # Registers the extension path.

apis = {
    "deep_gemm": [
        "m_grouped_bf16_gemm_nt_contiguous",
        "m_grouped_bf16_gemm_nn_contiguous",
        "k_grouped_bf16_gemm_tn_contiguous",
    ],
    "tile_kernels.mhc": [
        "mhc_sinkhorn_fwd",
        "mhc_sinkhorn_bwd",
        "mhc_pre_split_mixes_bwd",
        "mhc_pre_apply_mix_bwd",
        "mhc_post_bwd",
    ],
    "tile_kernels.quant": [
        "norm_forward",
        "norm_backward",
        "swiglu_forward",
        "swiglu_backward",
    ],
    "tile_kernels.moe": ["moe_topk_gate_forward", "moe_topk_gate_backward"],
    "transformer_engine_torch": ["rmsnorm_fwd", "rmsnorm_bwd"],
    "flash_mla": ["flash_mla_sparse_fwd"],
    "cudnn.deepseek_sparse_attention": ["sparse_attention_backward_wrapper"],
    "megatron.core.transformer.experimental_attention_variant.csa_utils.packed_sparse_attention": [
        "csa_sparse_attn"
    ],
}
result = {
    "python": sys.version,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "gpu_available": torch.cuda.is_available(),
    "apis": {},
}
for name, symbols in apis.items():
    module = importlib.import_module(name)
    result["apis"][name] = {
        symbol: callable(getattr(module, symbol, None)) for symbol in symbols
    }
    assert all(result["apis"][name].values()), result["apis"][name]
result["packages"] = {
    name: metadata.version(name)
    for name in [
        "torch",
        "transformers",
        "tile-kernels",
        "tilelang",
        "deep-gemm",
        "transformer-engine",
        "flash-mla",
        "megatron-core",
        "nvidia-cudnn-frontend",
    ]
}
Path("/opt/nexspec/evidence/cpu-imports.json").write_text(json.dumps(result, indent=2))
print(json.dumps(result, indent=2))
