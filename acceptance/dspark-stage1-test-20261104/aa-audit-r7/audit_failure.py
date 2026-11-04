"""Replay captured r7 responses on CPU; never send requests or disclose token arrays."""

import hashlib
import json
import struct
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def pair_summary(left, right):
    a, b = left["output_ids"], right["output_ids"]
    assert left["id"] == right["id"]
    assert left["input_sha256"] == right["input_sha256"]
    first = next(
        (index for index, (x, y) in enumerate(zip(a, b)) if x != y),
        min(len(a), len(b)),
    )
    eos = [
        next((i for i, token in enumerate(ids) if token == 1), None) for ids in (a, b)
    ]
    different = a != b
    return {
        "id": left["id"],
        "input_sha256": left["input_sha256"],
        "input_tokens": left["input_tokens"],
        "output_tokens": [len(a), len(b)],
        "output_equal": not different,
        "finish_equal": left["meta_info"]["finish_reason"]
        == right["meta_info"]["finish_reason"],
        "finish_reasons": [
            item["meta_info"]["finish_reason"] for item in (left, right)
        ],
        "first_difference_zero_based": first if different else None,
        "first_eos_zero_based": eos,
        "difference_before_either_eos": different
        and all(index is None or first < index for index in eos),
        "cached_tokens": [item["meta_info"]["cached_tokens"] for item in (left, right)],
        "num_retractions": [
            item["meta_info"]["num_retractions"] for item in (left, right)
        ],
        "observed_weight_version_labels_equal": (
            left["meta_info"].get("weight_version")
            == right["meta_info"].get("weight_version")
            and [
                item["version"] for item in left["meta_info"].get("weight_versions", [])
            ]
            == [
                item["version"]
                for item in right["meta_info"].get("weight_versions", [])
            ]
        ),
    }


def replay_events(path, expected, evaluator):
    recovered = []
    last = None
    elapsed = -1
    count = 0
    for line in path.read_text().splitlines():
        record = json.loads(line)
        assert record["elapsed_seconds"] >= elapsed
        elapsed = record["elapsed_seconds"]
        last = record["response"]
        recovered = evaluator.merge_ids(recovered, last)
        count += 1
    assert count and recovered == expected["output_ids"]
    assert last["meta_info"] == expected["meta_info"]
    evaluator.validate_completion(
        recovered, last["meta_info"], 1024, expected["input_tokens"]
    )
    return count


def audit(root):
    sys.path.insert(0, str(root))
    import evaluate
    from common import verify_execution

    verify_execution()
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["generation"]["ignore_eos"] is False
    assert protocol["generation"]["max_new_tokens"] == 1024
    validation = {
        row["id"]: row
        for row in (
            json.loads(line)
            for line in (root / "validation-inputs.jsonl").read_text().splitlines()
        )
    }
    rows = []
    evidence = {}
    for path in sorted((root / "target-only").glob("diagnostic-[0-9].json")):
        repeat = path.with_name(path.stem + "-repeat.json")
        left, right = (json.loads(p.read_text()) for p in (path, repeat))
        for file, item in ((path, left), (repeat, right)):
            raw = file.with_suffix(".events.jsonl")
            replay_events(raw, item, evaluate)
            assert (
                item["requested_max_new_tokens"] == 1024 and item["ignore_eos"] is False
            )
            assert (
                item["meta_info"]["cached_tokens"]
                == item["meta_info"]["num_retractions"]
                == 0
            )
            for p in (file, raw):
                evidence[str(p.relative_to(root))] = sha(p)
        row = pair_summary(left, right)
        if row["id"] in validation:
            original = validation[row["id"]]["input_ids"]
            digest = hashlib.sha256(
                b"".join(struct.pack("<q", token) for token in original)
            ).hexdigest()
            assert (
                digest == row["input_sha256"] and len(original) == row["input_tokens"]
            )
        rows.append({"diagnostic": path.stem, **row})
    mismatches = [
        row for row in rows if not row["output_equal"] or not row["finish_equal"]
    ]
    assert len(rows) == 4 and len(mismatches) == 1
    assert mismatches[0]["first_difference_zero_based"] == 19
    assert mismatches[0]["difference_before_either_eos"]
    assert mismatches[0]["output_tokens"] == [669, 403]
    repro = json.loads((root / "target-only/r3-eos-repro.json").read_text())
    assert repro["passed"] and repro["tokens"] == 61
    tests = {
        group: len(list((root / group).glob("test-*.json")))
        for group in ("target-only", "official-mtp", "step25", "step118")
    }
    assert not any(tests.values())
    assert not (root / "target-only/aa-acceptance.json").exists()
    deleted = json.loads((root / "eval-delete.json").read_text())
    assert (
        deleted["name"] == "job-6b4dd801c01d-20261104173837"
        and deleted["response"]["success"]
    )
    from controller import single

    absent = single("denovo-dspark-test-eval-20261104-r7") is None
    assert absent
    return {
        "audit_passed": True,
        "gpu_aa_passed": False,
        "execution_complete": False,
        "raw_sse_matches_saved_outputs": True,
        "real_server_eos_contract_passed_for_all_captured_requests": True,
        "mismatch_before_eos_confirmed": True,
        "diagnostics_completed": len(rows),
        "diagnostics_planned": 10,
        "diagnostics": rows,
        "r3_frozen_61_token_reproduction_passed": True,
        "test_requests": tests,
        "job_alias_absent": absent,
        "evidence_sha256": evidence,
        "execution_manifest_sha256": sha(root / "execution-sha256.json"),
        "numerical_root_cause_confirmed": False,
        "scope": "Audit reproduces recorded A/A failure on CPU. No new inference, no kernel fix, no passing A/A or test score.",
    }


if __name__ == "__main__":
    print(json.dumps(audit(Path(sys.argv[1])), indent=2))
