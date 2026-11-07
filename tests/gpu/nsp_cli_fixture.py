"""Small deterministic real target weights/features for nsp CLI validation."""

import json
from pathlib import Path

import torch
import yaml
from transformers import Qwen3Config, Qwen3ForCausalLM

from tests.test_runtime import _fixtures as fx


def create_fixture(root):
    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(42)
    target = root / "target"
    target_config = Qwen3Config(
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=4,
        num_attention_heads=4,
        num_key_value_heads=2,
        head_dim=16,
        vocab_size=256,
        max_position_embeddings=512,
        tie_word_embeddings=False,
        bos_token_id=1,
        eos_token_id=2,
        pad_token_id=0,
    )
    Qwen3ForCausalLM(target_config).save_pretrained(target)
    draft = target_config.to_dict()
    draft.update(
        architectures=["DFlashDraftModel"],
        num_hidden_layers=1,
        layer_types=["full_attention"],
        num_target_layers=4,
        block_size=4,
        dflash_config={"mask_token_id": 0, "target_layer_ids": [1]},
    )
    (root / "draft.json").write_text(json.dumps(draft, indent=2))
    feature_dir = root / "features"
    fx.write_offline_files_dflash(
        str(feature_dir), n=6, seq=32, hidden=64, vocab=256, seed=42
    )
    # Long and short records exercise uneven final batches and ctx coverage.
    record = torch.load(feature_dir / "0000.ckpt", weights_only=False)
    record["input_ids"] = record["input_ids"][:16]
    record["loss_mask"] = record["loss_mask"][:16]
    record["hidden_states"] = record["hidden_states"][:, :16]
    torch.save(record, feature_dir / "0000.ckpt")
    config = {
        "model": {
            "target_model_path": str(target),
            "draft_model_config": str(root / "draft.json"),
            "mask_token_id": 0,
            "use_liger_kernel": False,
            "torch_dtype": "bfloat16",
        },
        "data": {
            "hidden_states_path": str(feature_dir),
            "max_length": 64,
            "dataloader_num_workers": 0,
        },
        "training": {
            "strategy": "dflash",
            "attention_backend": "sdpa",
            "batch_size": 1,
            "accumulation_steps": 1,
            "max_steps": 3,
            "num_epochs": 1,
            "learning_rate": 1e-3,
            "save_interval": 1,
            "log_interval": 1,
            "seed": 42,
            "num_anchors": 4,
            "objective_checkpointing": False,
            "dist_timeout": 2,
        },
        "run_id": "fixture",
        "output_dir": str(root / "unused-output"),
    }
    # Config validation should fail now rather than after submitting the GPU job.
    from specforge.application import resolve_run
    from specforge.config import Config

    resolve_run(Config.model_validate(config))
    (root / "train.yaml").write_text(yaml.safe_dump(config))
    return root / "train.yaml"


if __name__ == "__main__":
    import sys

    print(create_fixture(sys.argv[1]))
