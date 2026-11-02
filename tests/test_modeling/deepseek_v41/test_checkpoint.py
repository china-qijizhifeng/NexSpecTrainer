import json

import pytest
import torch
from safetensors.torch import save_file

from specforge.modeling.draft.deepseek_v41 import (
    DeepSeekV41DraftConfig,
    DeepSeekV41DraftModel,
)
from specforge.modeling.draft.deepseek_v41.checkpoint import (
    FP4_VALUES,
    NativeCheckpoint,
    dequantize,
    initialize_native,
    native_parts,
    quantize_like,
)
from specforge.modeling.draft.deepseek_v41.export import export_native
from specforge.modeling.draft.deepseek_v41.validation import validate_target


def tiny_config():
    return DeepSeekV41DraftConfig(
        hidden_size=128,
        vocab_size=256,
        num_hidden_layers=2,
        num_attention_heads=64,
        head_dim=512,
        qk_rope_head_dim=64,
        q_lora_rank=128,
        o_lora_rank=64,
        o_groups=8,
        moe_intermediate_size=128,
        n_routed_experts=4,
        num_experts_per_tok=2,
        hc_mult=4,
        sliding_window=4,
        block_size=3,
        target_layer_ids=[1, 2],
        mask_token_id=255,
        markov_rank=32,
    )


def test_known_fp4_nibbles_and_e8m0_scales():
    packed = torch.arange(256, dtype=torch.uint8).view(8, 32)
    scales = torch.full((8, 2), 2.0).to(torch.float8_e8m0fnu)
    out = dequantize(packed.view(torch.int8), scales)
    for row in range(8):
        for col in range(32):
            value = row * 32 + col
            assert out[row, 2 * col] == 2 * FP4_VALUES[value & 15]
            assert out[row, 2 * col + 1] == 2 * FP4_VALUES[value >> 4]


@pytest.mark.parametrize("fp4", [True, False])
def test_quantization_roundtrip(fp4):
    torch.manual_seed(7)
    value = torch.randn(64, 128)
    raw = (
        torch.empty(64, 64, dtype=torch.int8)
        if fp4
        else torch.empty(64, 128, dtype=torch.float8_e4m3fn)
    )
    scale = torch.empty((64, 4) if fp4 else (2, 4), dtype=torch.float8_e8m0fnu)
    quantized, sf = quantize_like(value, raw, scale)
    assert quantized.shape == raw.shape and sf.shape == scale.shape
    error = (dequantize(quantized, sf) - value).square().mean().sqrt()
    assert error < (0.2 if fp4 else 0.04)
    with pytest.raises(ValueError, match="non-finite"):
        quantize_like(value.fill_(float("nan")), raw, scale)


def test_native_stacked_mapping_load_export(tmp_path):
    c = tiny_config()
    with torch.device("meta"):
        model = DeepSeekV41DraftModel(c)
    # Distinct expert/half contents catch swapped w1/w3, transposes and expert IDs.
    native = {}
    for n, (key, parameter) in enumerate(model.named_parameters()):
        value = torch.full(parameter.shape, n / 100, dtype=torch.float32)
        for i, (name, sl) in enumerate(native_parts(key, parameter.shape)):
            native[name] = value[sl].clone().fill_(n / 100 + i / 1000)
    native["embed.weight"] = torch.randn(c.vocab_size, c.hidden_size)
    source = tmp_path / "source"
    source.mkdir()
    save_file(native, source / "weights.safetensors")
    (source / "config.json").write_text("{}")
    (source / "model.safetensors.index.json").write_text(
        json.dumps({"weight_map": dict.fromkeys(native, "weights.safetensors")})
    )
    initialize_native(model, source)
    assert model.mtp[0].attn.sink.weight.dtype == torch.float32
    assert model.mtp[-1].confidence_head.proj.weight.dtype == torch.float32
    assert model.mtp[0].hc_attn.fn.dtype == torch.float32
    assert model.mtp[0].ffn.gate_up.dtype == torch.bfloat16
    trained = model.state_dict()
    output = export_native(trained, source, tmp_path / "export")
    exported = NativeCheckpoint(output)
    for key, tensor in trained.items():
        for name, sl in native_parts(key, tensor.shape):
            torch.testing.assert_close(exported.floating(name), tensor[sl].float())
    assert (output / "weights.safetensors").is_symlink()
    torch.testing.assert_close(
        NativeCheckpoint(source).tensor("embed.weight"), native["embed.weight"]
    )
    with pytest.raises(FileExistsError):
        export_native(trained, source, output)
    del trained["mtp.0.ffn.down"]
    with pytest.raises(ValueError, match="key mismatch"):
        export_native(trained, source, tmp_path / "bad")
    assert not (tmp_path / "bad").exists()


def test_cuda_required_no_cpu_fallback():
    from specforge.modeling.draft.deepseek_v41.ops import RMSNorm

    with pytest.raises(RuntimeError, match="CUDA kernels"):
        RMSNorm(128, 1e-6)(torch.randn(2, 128))


def test_native_target_validation():
    c = DeepSeekV41DraftConfig()
    raw = {"model_type": "wrong", "text_config": {}}
    with pytest.raises(ValueError, match="native DeepSeek"):
        validate_target(c, raw)
