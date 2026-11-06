"""Check checkpoint and native-forward contracts in the actual trainer image."""

import hashlib
import importlib
import importlib.metadata as md
import inspect
import json
import os
import sys
from pathlib import Path

import google.protobuf
import nexflashcheckpoint
import torch
from google.protobuf.internal.api_implementation import Type
from megatron.core.dist_checkpointing import serialization
from megatron.core.dist_checkpointing.strategies.torch import (
    TorchDistSaveShardedStrategy,
)

root = Path(os.environ["NODE_ROOT"])
assert torch.__version__ == "2.10.0a0+b558c986e8.nv25.11"
assert md.version("transformer-engine") == "2.20.2+6ea2a74"
assert (
    "async_strategy"
    in inspect.signature(
        TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks
    ).parameters
)
core_sha = hashlib.sha256(Path(serialization.__file__).read_bytes()).hexdigest()
assert core_sha == "5f745e742b63b4d4f4c406edda9c09605746bd52faa24c1b5fb4446676799a65"
apis = {
    "transformer_engine.pytorch": (),
    "transformer_engine_torch": ("rmsnorm_fwd", "rmsnorm_bwd"),
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
    "megatron.core.transformer.moe.moe_utils": ("permute", "unpermute"),
    "megatron.core.transformer.experimental_attention_variant.csa_utils.packed_sparse_attention": (
        "csa_sparse_attn",
    ),
    "flash_mla": ("flash_mla_sparse_fwd",),
    "cudnn.deepseek_sparse_attention": ("sparse_attention_backward_wrapper",),
}
locations = {}
for name, symbols in apis.items():
    module = importlib.import_module(name)
    assert all(callable(getattr(module, symbol, None)) for symbol in symbols), name
    locations[name] = module.__file__
from megatron.core.transformer.moe.moe_utils import permute, unpermute

assert {"fused", "tokens_per_expert", "align_size"} <= set(
    inspect.signature(permute).parameters
)
assert {"fused", "pad_offsets"} <= set(inspect.signature(unpermute).parameters)
gpu_check = "--gpu" in sys.argv
if gpu_check:
    assert torch.cuda.device_count() == 8
    assert all("H200" in torch.cuda.get_device_name(i) for i in range(8))
    from specforge.modeling.draft.deepseek_v41.validation import preflight

    preflight()
result = {
    "passed": True,
    "gpu_preflight": gpu_check,
    "python": sys.version,
    "torch": torch.__version__,
    "te": md.version("transformer-engine"),
    "protobuf": google.protobuf.__version__,
    "protobuf_backend": Type(),
    "nfc": nexflashcheckpoint.__file__,
    "core": serialization.__file__,
    "core_sha256": core_sha,
    "module_locations": locations,
    "scope": "module and API compatibility; actual optimizer updates separately audited",
}
(
    root
    / ("training/training-environment.json" if gpu_check else "cpu-environment.json")
).write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result), flush=True)
