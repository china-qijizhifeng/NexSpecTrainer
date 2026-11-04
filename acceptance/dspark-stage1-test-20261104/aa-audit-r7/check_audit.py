"""Exercise the failure auditor against real SSE/EOS functions without model execution."""

import copy
import importlib.util
import json
import sys
import tempfile
from pathlib import Path

from audit_failure import pair_summary, replay_events


def row(tokens):
    return {
        "id": "fixture",
        "input_sha256": "same",
        "input_tokens": 4,
        "output_ids": tokens,
        "meta_info": {
            "completion_tokens": len(tokens),
            "prompt_tokens": 4,
            "finish_reason": {"type": "stop", "matched": 1},
            "cached_tokens": 0,
            "num_retractions": 0,
        },
    }


def rejected(call):
    try:
        call()
    except AssertionError:
        return
    raise AssertionError("corrupt evidence accepted")


def main():
    sys.path.insert(0, sys.argv[1])
    path = Path(sys.argv[1]) / "evaluate.py"
    spec = importlib.util.spec_from_file_location("recorded_evaluator", path)
    evaluator = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(evaluator)
    left, right = row([7, 8, 1]), row([7, 9, 10, 1])
    mismatch = pair_summary(left, right)
    assert mismatch["first_difference_zero_based"] == 1
    assert mismatch["difference_before_either_eos"]
    assert pair_summary(left, left)["output_equal"]
    rejected(lambda: pair_summary(left, {**right, "input_sha256": "different"}))
    with tempfile.TemporaryDirectory() as directory:
        p = Path(directory) / "events.jsonl"
        first = {
            "elapsed_seconds": 0.1,
            "response": {"output_ids": [7], "meta_info": {"completion_tokens": 1}},
        }
        last = {
            "elapsed_seconds": 0.2,
            "response": {"output_ids": [8, 1], "meta_info": left["meta_info"]},
        }
        p.write_text("\n".join(json.dumps(event) for event in [first, last]) + "\n")
        assert replay_events(p, left, evaluator) == 2
        corrupt = copy.deepcopy(last)
        corrupt["response"]["output_ids"] = [9, 1]
        p.write_text("\n".join(json.dumps(event) for event in [first, corrupt]) + "\n")
        rejected(lambda: replay_events(p, left, evaluator))
        post_eos = row([7, 1, 9])
        p.write_text(
            json.dumps(
                {
                    "elapsed_seconds": 0.1,
                    "response": {
                        "output_ids": post_eos["output_ids"],
                        "meta_info": post_eos["meta_info"],
                    },
                }
            )
            + "\n"
        )
        rejected(lambda: replay_events(p, post_eos, evaluator))
    print(
        json.dumps(
            {
                "passed": True,
                "captured_sse_replay_contract": True,
                "pre_eos_failure_detected": True,
                "corrupt_input_or_stream_rejected": True,
                "post_eos_rejected": True,
                "scope": "CPU evidence audit; no GPU determinism claim",
            }
        )
    )


if __name__ == "__main__":
    main()
