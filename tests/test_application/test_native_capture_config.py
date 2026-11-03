"""Configuration-only regression for released native V4.1 capture."""

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from specforge.application import resolve_offline_capture
from specforge.config import Config
from specforge.training.capture_contract import resolve_server_capture_contract

DRAFT = Path(__file__).resolve().parents[2] / "configs/deepseek-v41-flash-native.json"
TARGET_TEXT = {
    "num_hidden_layers": 40,
    "num_nextn_predict_layers": 3,
    "hidden_size": 5120,
    "vocab_size": 129280,
    "num_attention_heads": 64,
    "head_dim": 512,
    "qk_rope_head_dim": 64,
    "q_lora_rank": 1280,
    "o_lora_rank": 1024,
    "o_groups": 8,
    "moe_intermediate_size": 2304,
    "routed_scaling_factor": 1.5,
    "hc_mult": 4,
    "hc_sinkhorn_iters": 20,
    "hc_eps": 1e-6,
    "rms_norm_eps": 1e-20,
    "rope_theta": 10000.0,
    "sliding_window": 128,
    "swiglu_limit": 10.0,
    "dspark_n_routed_experts": 128,
    "dspark_num_experts_per_tok": 3,
    "dspark_block_size": 5,
    "dspark_noise_token_id": 128799,
    "dspark_target_layer_ids": [37, 38, 39],
    "dspark_markov_rank": 256,
}


class NativeCaptureConfigTest(unittest.TestCase):
    def test_unregistered_target_resolves_both_capture_contracts(self):
        for overrides in ({}, {"dspark_block_size": 6}):
            with self.subTest(overrides=overrides), TemporaryDirectory() as directory:
                (Path(directory) / "config.json").write_text(
                    json.dumps(
                        {
                            "model_type": "deepseek_v41",
                            "text_config": {**TARGET_TEXT, **overrides},
                        }
                    )
                )
                cfg = Config.model_validate(
                    {
                        "model": {
                            "target_model_path": directory,
                            "draft_model_config": str(DRAFT),
                        },
                        "data": {"hidden_states_path": "features"},
                        "training": {
                            "strategy": "deepseek_v41",
                            "attention_backend": "megatron_csa",
                        },
                    }
                )
                with mock.patch(
                    "transformers.AutoConfig.from_pretrained",
                    side_effect=ValueError("unregistered model type deepseek_v41"),
                ):
                    if overrides:
                        with self.assertRaisesRegex(ValueError, "block_size"):
                            resolve_offline_capture(cfg)
                        # Resolve the registry without loading model weights.
                        from specforge.application import resolve_run

                        with self.assertRaisesRegex(ValueError, "block_size"):
                            resolve_server_capture_contract(
                                cfg, algorithm=resolve_run(cfg).algorithm
                            )
                    else:
                        plan = resolve_offline_capture(cfg)
                        contract = resolve_server_capture_contract(
                            cfg, algorithm=plan.run.algorithm
                        )
                        self.assertEqual(plan.capture_layers, (37, 38, 39))
                        self.assertEqual(contract.aux_layer_ids, plan.capture_layers)
                        self.assertEqual(contract.method, "dspark")
                        self.assertEqual(contract.target_hidden_size, 5120)
                        self.assertEqual(contract.target_vocab_size, 129280)
                        self.assertEqual(contract.draft_vocab_size, 129280)
