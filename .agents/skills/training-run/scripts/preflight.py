"""Check the temporary Weaver V4.1 environment; GPU execution is opt-in."""

import argparse
import hashlib
import importlib
import inspect
import json
import sys
from importlib import metadata
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Require eight H100 GPUs and native CUDA preflight",
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import google.protobuf
    import nexflashcheckpoint
    import torch
    from google.protobuf.internal.api_implementation import Type
    from megatron.core.dist_checkpointing import serialization
    from megatron.core.dist_checkpointing.strategies.torch import (
        TorchDistSaveShardedStrategy,
    )

    importlib.import_module("transformer_engine.pytorch")
    importlib.import_module("specforge.training.nfc_checkpoint")

    if (
        torch.__version__ != "2.10.0a0+b558c986e8.nv25.11"
        or metadata.version("transformer-engine") != "2.20.2+6ea2a74"
    ):
        raise RuntimeError(
            "Temporary workflow requires the validated Weaver Torch/TE binaries"
        )
    if (
        "async_strategy"
        not in inspect.signature(
            TorchDistSaveShardedStrategy._get_save_and_finalize_callbacks
        ).parameters
    ):
        raise RuntimeError("Loaded Core cannot run the NFC asynchronous pipeline")
    core_sha = hashlib.sha256(Path(serialization.__file__).read_bytes()).hexdigest()
    if core_sha != "5f745e742b63b4d4f4c406edda9c09605746bd52faa24c1b5fb4446676799a65":
        raise RuntimeError(
            "Unexpected checkpoint Core; audit the ABI before proceeding"
        )
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
        "transformer_engine_torch": ("rmsnorm_fwd", "rmsnorm_bwd"),
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
        if not all(callable(getattr(module, symbol, None)) for symbol in symbols):
            raise RuntimeError(f"Native forward API missing: {name}")
        locations[name] = module.__file__
    import deep_gemm
    from packaging.version import Version

    if Version(deep_gemm.__version__) < Version("2.8.2"):
        raise RuntimeError("SM90 prefix-sum grouped GEMM requires DeepGEMM >=2.8.2")
    from megatron.core.transformer.moe.moe_utils import permute, unpermute

    if (
        not {"fused", "tokens_per_expert", "align_size"}
        <= set(inspect.signature(permute).parameters)
        or "pad_offsets" not in inspect.signature(unpermute).parameters
    ):
        raise RuntimeError("MoE dispatch signatures do not match native forward")
    if args.gpu:
        if torch.cuda.device_count() != 8 or not all(
            "H100" in torch.cuda.get_device_name(i) for i in range(8)
        ):
            raise RuntimeError("This temporary GPU recipe requires eight H100 GPUs")
        from specforge.modeling.draft.deepseek_v41.validation import preflight

        preflight()
    result = {
        "passed": True,
        "gpu_preflight": args.gpu,
        "python": sys.version,
        "torch": torch.__version__,
        "te": metadata.version("transformer-engine"),
        "protobuf": google.protobuf.__version__,
        "protobuf_backend": Type(),
        "nfc": nexflashcheckpoint.__file__,
        "core": serialization.__file__,
        "core_sha256": core_sha,
        "module_locations": locations,
        "scope": "environment/API check; actual optimizer and checkpoint acceptance are separate",
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
