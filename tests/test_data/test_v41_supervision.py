"""Run against the same pinned official encoder/tokenizer used by capture."""

import importlib.util
import os
from pathlib import Path

import numpy as np
import pytest
from tokenizers import Tokenizer

from specforge.data.v41_supervision import encode_supervision


@pytest.fixture
def official():
    snapshot = os.environ.get("SPECFORGE_V41_SNAPSHOT")
    if not snapshot:
        pytest.skip("set SPECFORGE_V41_SNAPSHOT to the pinned official snapshot")
    root = Path(snapshot)
    spec = importlib.util.spec_from_file_location(
        "test_official_v41", root / "encoding/encoding.py"
    )
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    return encoder, Tokenizer.from_file(str(root / "tokenizer.json"))


def test_cleaning_flags_preserve_context_and_exclude_only_marked_spans(official):
    encoder, tokenizer = official
    calls = [
        {
            "id": f"call_{i}",
            "type": "function",
            "use_loss": i == 0,
            "function": {"name": "bash", "arguments": '{"command":"echo hello"}'},
        }
        for i in range(2)
    ]
    messages = [
        {"role": "user", "content": "Please use the tool."},
        {
            "role": "assistant",
            "reasoning_content": "PRIVATE REASONING TEXT",
            "use_thinking": False,
            "content": [
                {"type": "text", "text": "LEARN THIS SENTENCE", "use_loss": True},
                {"type": "text", "text": "HIDDEN CONTENT PART", "use_loss": False},
            ],
            "tool_calls": calls,
        },
        {"role": "tool", "tool_call_id": "call_0", "content": "TOOL OUTPUT ONE"},
        {
            "role": "user",
            "content_blocks": [
                {
                    "type": "tool_result",
                    "tool_use_id": "call_1",
                    "content": "TOOL OUTPUT TWO",
                }
            ],
        },
        {"role": "assistant", "content": "MASKED WHOLE MESSAGE", "use_loss": False},
        {"role": "user", "content": "Continue."},
        {"role": "assistant", "content": "VISIBLE FINAL ANSWER"},
    ]
    options = dict(thinking_mode="thinking", drop_thinking=False, reasoning_effort=75)
    ids, mask, call_ids = encode_supervision(
        encoder, tokenizer, messages, missing_use_loss=True, **options
    )
    text = encoder.encode_messages(messages, **options)
    encoded = tokenizer.encode(text, add_special_tokens=False)
    assert ids.tolist() == encoded.ids
    for fragment in (
        "PRIVATE REASONING TEXT",
        "HIDDEN CONTENT PART",
        "TOOL OUTPUT ONE",
        "TOOL OUTPUT TWO",
        "MASKED WHOLE MESSAGE",
    ):
        a = text.index(fragment)
        b = a + len(fragment)
        assert not any(
            mask[i] for i, (x, y) in enumerate(encoded.offsets) if x < b and y > a
        )
    assert mask[call_ids == 1].all() and not mask[call_ids == 2].any()
    assert (call_ids == 1).any() and (call_ids == 2).any()
    assert mask.sum() > 0
    # The encoder must be restored even after each annotation pass.
    assert encoder.encode_messages(messages, **options) == text


@pytest.mark.parametrize(
    "flag,value", [("use_loss", "false"), ("use_loss", 0), ("use_thinking", None)]
)
def test_malformed_cleaning_flag_rejected(official, flag, value):
    with pytest.raises(ValueError, match="strict boolean"):
        encode_supervision(
            *official,
            [{"role": "assistant", "content": "x", flag: value}],
            missing_use_loss=True,
        )


def test_explicit_false_wins_over_legacy_mask_and_missing_policy(official):
    encoder, tokenizer = official
    messages = [
        {"role": "user", "content": "hello"},
        {"role": "assistant", "content": "answer", "loss_mask": 1, "use_loss": False},
    ]
    _, mask, _ = encode_supervision(
        encoder,
        tokenizer,
        messages,
        missing_use_loss=True,
        allow_legacy_loss_mask=True,
        thinking_mode="thinking",
        drop_thinking=False,
    )
    assert not np.any(mask)
