"""Freeze complete-history generation boundaries before any test outputs exist."""

import bisect
import hashlib
import importlib.util
import json
import os
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
from tokenizers import Tokenizer

ROOT = Path(__file__).resolve().parent
DATA = Path("/Users/shom/Downloads/dspark-denovo-20261103")
SNAP = Path(
    "/Users/shom/Downloads/speculative-decoding-training-research/v41/rdma-realtrace-20261103/metadata-snapshot"
)
CAP = 64512


def digest(ids):
    return hashlib.sha256(np.asarray(ids, dtype="<i8").tobytes()).hexdigest()


def main():
    pins = json.loads((DATA / "evidence/tokenizer-template-pin.json").read_text())
    for name, sha in pins["files"].items():
        assert hashlib.sha256((SNAP / name).read_bytes()).hexdigest() == sha
    spec = importlib.util.spec_from_file_location(
        "official", SNAP / "encoding/encoding.py"
    )
    encoder = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(encoder)
    tok = Tokenizer.from_file(str(SNAP / "tokenizer.json"))
    split = json.loads((DATA / "evidence/instance-split-v1.json").read_text())[
        "instances"
    ]
    profiles = {
        x["trajectory_id"]: x
        for x in map(
            json.loads,
            (DATA / "evidence/trajectory-profile-full.jsonl").read_text().splitlines(),
        )
    }
    # Validation diagnostics chosen using IDs only; no test output is used for selection.
    val_ids = sorted(
        (k for k, v in profiles.items() if split[v["instance_id"]] == "validation"),
        key=lambda k: hashlib.sha256(k.encode()).hexdigest(),
    )[:8]
    rows = []
    for batch in pq.ParquetFile(DATA / "data/source.parquet").iter_batches(
        batch_size=8
    ):
        for row in batch.to_pylist():
            group = split[row["instance_id"]]
            if group != "test" and row["trajectory_id"] not in val_ids:
                continue
            messages = [
                {k: v for k, v in m.items() if v is not None} for m in row["messages"]
            ]
            assert (
                messages[0]["role"] == "system"
                and messages[0]["content"] == row["system"]
            )
            messages[0]["tools"] = json.loads(row["tools"])
            orig_assistant = [
                i for i, m in enumerate(messages) if m["role"] == "assistant"
            ]
            cursors, cursor, ordinal = [], len(encoder.bos_token), 0
            render = encoder.render_message

            def observe(index, processed, **kwargs):
                nonlocal cursor, ordinal
                if processed[index]["role"] == "assistant":
                    original_index = orig_assistant[ordinal]
                    if processed[index].get("loss_mask", 0) == 1:
                        cursors.append((cursor, original_index))
                    ordinal += 1
                out = render(index, processed, **kwargs)
                cursor += len(out)
                return out

            encoder.render_message = observe
            try:
                full = encoder.encode_messages(
                    messages,
                    thinking_mode="thinking",
                    drop_thinking=False,
                    reasoning_effort=75,
                )
            finally:
                encoder.render_message = render
            assert ordinal == len(orig_assistant) and cursor == len(full)
            encoded = tok.encode(full, add_special_tokens=False)
            profile = profiles[row["trajectory_id"]]
            assert (
                digest(encoded.ids) == profile["token_sha256"]
                and len(encoded.ids) == profile["tokens"]
            )
            ends = [b for a, b in encoded.offsets]
            candidates = [
                (c, i) for c, i in cursors if bisect.bisect_right(ends, c) <= CAP
            ]
            selected = None
            for c, i in reversed(candidates):
                prompt = encoder.encode_messages(
                    messages[:i],
                    thinking_mode="thinking",
                    drop_thinking=False,
                    reasoning_effort=75,
                )
                assert full.startswith(prompt), (row["trajectory_id"], i)
                ids = tok.encode(prompt, add_special_tokens=False).ids
                if len(ids) <= CAP and prompt.endswith(encoder.thinking_start_token):
                    selected = (i, ids)
                    break
            assert selected, ("no eligible complete boundary", row["trajectory_id"])
            i, ids = selected
            rows.append(
                {
                    "id": row["trajectory_id"],
                    "instance_id": row["instance_id"],
                    "split": group,
                    "assistant_message_index": i,
                    "input_ids": ids,
                    "input_tokens": len(ids),
                    "input_sha256": digest(ids),
                    "full_trajectory_tokens": profile["tokens"],
                    "full_trajectory_sha256": profile["token_sha256"],
                    "history_policy": "all original messages [0,assistant_index), no truncation; last eligible supervised assistant boundary",
                }
            )
            if len(rows) % 20 == 0:
                print(json.dumps({"prepared": len(rows)}), flush=True)
    rows.sort(key=lambda r: (r["split"], hashlib.sha256(r["id"].encode()).hexdigest()))
    for group in ("test", "validation"):
        selected = [r for r in rows if r["split"] == group]
        with (ROOT / (group + "-inputs.jsonl")).open("x") as f:
            for row in selected:
                f.write(
                    json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n"
                )
    test = [r for r in rows if r["split"] == "test"]
    assert len(test) == 247 and len({r["instance_id"] for r in test}) == 36
    assert {r["instance_id"] for r in rows if r["split"] == "validation"}.isdisjoint(
        {r["instance_id"] for r in test}
    )
    out = {
        "passed": True,
        "dataset_revision": "a9c33019adbcd00c93e4fdd47c9dc3e4b851d35f",
        "target_revision": "1bae0e1088993b9b82de08f59de5be5a34e42170",
        "input_cap": CAP,
        "max_new_tokens": 1024,
        "test_trajectories": 247,
        "test_instances": 36,
        "validation_diagnostic_prompts": 8,
        "encoding": pins,
        "selection": "one last eligible supervised assistant boundary per original test trajectory; complete prefix; before any output",
        "scope": "held-out prompt replay, not closed-loop SWE task success",
        "files": {
            g
            + "-inputs.jsonl": hashlib.sha256(
                (ROOT / (g + "-inputs.jsonl")).read_bytes()
            ).hexdigest()
            for g in ("test", "validation")
        },
        "samples": [{k: v for k, v in r.items() if k != "input_ids"} for r in rows],
    }
    (ROOT / "inputs-manifest.json").write_text(json.dumps(out, indent=2) + "\n")
    print(
        json.dumps({k: v for k, v in out.items() if k not in ("samples", "encoding")}),
        flush=True,
    )


if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    main()
