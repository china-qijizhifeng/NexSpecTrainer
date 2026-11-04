"""Audit captured A/A responses; never generate requests or read weight shards."""

import hashlib
import json
import struct
import sys
from pathlib import Path


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def first_difference(left, right):
    return next(
        (i for i, (a, b) in enumerate(zip(left, right)) if a != b),
        min(len(left), len(right)),
    )


def through_eos(ids, eos_ids):
    stop = next((i + 1 for i, token in enumerate(ids) if token in eos_ids), len(ids))
    return ids[:stop]


def verify_events(path, expected):
    previous = []
    last = None
    for line in path.read_text().splitlines():
        event = json.loads(line)["response"]
        ids = event["output_ids"]
        total = event["meta_info"]["completion_tokens"]
        if len(ids) == total:
            assert ids[: len(previous)] == previous, "non-prefix raw stream"
            previous = ids
        else:
            assert len(previous) + len(ids) == total, "invalid delta count"
            previous += ids
        last = event
    assert last is not None and previous == expected["output_ids"]
    assert last["meta_info"] == expected["meta_info"]
    return len(previous)


def audit(root, model):
    config_path = model / "config.json"
    config = json.loads(config_path.read_text())
    eos = config["eos_token_id"]
    eos_ids = {eos} if isinstance(eos, int) else set(eos)
    protocol = json.loads((root / "protocol.json").read_text())
    assert protocol["generation"]["ignore_eos"] is True
    results = []
    evidence = {}
    for path in sorted((root / "target-only").glob("diagnostic-[0-9].json")):
        repeat = path.with_name(path.stem + "-repeat.json")
        assert repeat.exists()
        a, b = (json.loads(p.read_text()) for p in (path, repeat))
        assert a["id"] == b["id"] and a["input_sha256"] == b["input_sha256"]
        for p, item in ((path, a), (repeat, b)):
            raw = p.with_suffix(".events.jsonl")
            assert verify_events(raw, item) == item["meta_info"]["completion_tokens"]
            assert item["meta_info"]["cached_tokens"] == 0
            assert item["meta_info"]["num_retractions"] == 0
            evidence[str(p.relative_to(root))] = sha(p)
            evidence[str(raw.relative_to(root))] = sha(raw)
        left, right = a["output_ids"], b["output_ids"]
        ae, be = through_eos(left, eos_ids), through_eos(right, eos_ids)
        k = first_difference(left, right)
        results.append(
            {
                "diagnostic": path.stem,
                "input_sha256": a["input_sha256"],
                "input_tokens": a["input_tokens"],
                "output_tokens": [len(left), len(right)],
                "full_equal": left == right,
                "first_difference_zero_based": None if left == right else k,
                "first_eos_zero_based": [
                    next((i for i, t in enumerate(x) if t in eos_ids), None)
                    for x in (left, right)
                ],
                "equal_through_first_eos": ae == be,
                "difference_strictly_after_eos": left != right
                and len(ae) <= k
                and len(be) <= k,
            }
        )
    mismatches = [item for item in results if not item["full_equal"]]
    assert mismatches, "expected the recorded failure"
    validation = [
        json.loads(line)
        for line in (root / "validation-inputs.jsonl").read_text().splitlines()
    ]
    for item in mismatches:
        (row,) = [
            row
            for row in validation
            if hashlib.sha256(
                b"".join(struct.pack("<q", token) for token in row["input_ids"])
            ).hexdigest()
            == item["input_sha256"]
        ]
        assert len(row["input_ids"]) == item["input_tokens"]
    tests = {
        group: len(list((root / group).glob("test-*.json")))
        for group in ("target-only", "official-mtp", "step25", "step118")
    }
    assert not any(
        tests.values()
    ), "test requests exist; stop before protocol amendment"
    return {
        "audit_passed": True,
        "original_fixed_length_aa_passed": False,
        "eos_token_ids": sorted(eos_ids),
        "target_config_sha256": sha(config_path),
        "raw_sse_matches_saved_outputs": True,
        "all_observed_mismatches_strictly_after_eos": all(
            item["difference_strictly_after_eos"] for item in mismatches
        ),
        "diagnostics": results,
        "test_requests": tests,
        "evidence_sha256": evidence,
        "numerical_root_cause_confirmed": False,
        "scope": "Recorded failure only. Equality before EOS is not a passing rerun, deterministic-kernel proof, or test score.",
    }


if __name__ == "__main__":
    root, model = (Path(arg) for arg in sys.argv[1:])
    print(json.dumps(audit(root, model), indent=2))
