"""CPU checks for EOS-boundary and saved-event audit contracts, not GPU parity."""

import json
import tempfile
from pathlib import Path

from audit_failure import first_difference, through_eos, verify_events


def main():
    assert through_eos([10, 1, 14], {1}) == [10, 1]
    assert through_eos([1, 14], {1}) == [1]
    assert through_eos([10, 14], {1}) == [10, 14]
    assert through_eos([10, 2, 14], {1, 2}) == [10, 2]
    assert through_eos([], {1}) == []
    left, right = [10, 1, 14], [10, 1, 15]
    assert through_eos(left, {1}) == through_eos(right, {1})
    assert first_difference(left, right) == 2
    assert through_eos([11, 1], {1}) != through_eos([10, 1], {1})
    assert first_difference([10, 1], [10, 1, 14]) == 2
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp) / "events.jsonl"
        final = {"completion_tokens": 3, "finish_reason": {"type": "length"}}
        expected = {"output_ids": [10, 1, 14], "meta_info": final}
        events = [
            {"output_ids": [10], "meta_info": {"completion_tokens": 1}},
            {"output_ids": [1], "meta_info": {"completion_tokens": 2}},
            {"output_ids": [10, 1, 14], "meta_info": final},
        ]
        path.write_text("".join(json.dumps({"response": e}) + "\n" for e in events))
        assert verify_events(path, expected) == 3
        events[1]["output_ids"] = [15]
        path.write_text("".join(json.dumps({"response": e}) + "\n" for e in events))
        try:
            verify_events(path, expected)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt raw prefix accepted")
    print(
        json.dumps(
            {
                "passed": True,
                "scope": "EOS boundary, pre-EOS mismatch, no-EOS and multiple-EOS handling, raw delta/cumulative replay and corruption rejection; no GPU numerical claim",
            }
        )
    )


if __name__ == "__main__":
    main()
