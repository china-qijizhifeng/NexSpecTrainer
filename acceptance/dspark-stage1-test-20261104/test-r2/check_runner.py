"""Exercise real SSE framing, delta/cumulative counts and pooled metric algebra."""

import io
import json
import tempfile
from pathlib import Path

from common import checkpoint_metadata, sha
from evaluate import aggregate, merge_ids, sse_events, validate_stats


def main():
    fixture = b': ping\r\n\r\ndata: {"output_ids":[9,4],\r\ndata: "meta_info":{"completion_tokens":2}}\r\n\r\ndata: {"output_ids":[9,4,5],"meta_info":{"completion_tokens":3}}\n\ndata: [DONE]\n\n'
    events = list(sse_events(io.BytesIO(fixture)))
    assert len(events) == 2
    ids = []
    for e in events:
        ids = merge_ids(ids, e)
    assert ids == [9, 4, 5]
    assert merge_ids(
        [9, 4], {"output_ids": [5], "meta_info": {"completion_tokens": 3}}
    ) == [9, 4, 5]
    assert (
        merge_ids(ids, {"output_ids": [], "meta_info": {"completion_tokens": 3}}) == ids
    )
    try:
        merge_ids(
            [1, 2], {"output_ids": [1, 3, 4], "meta_info": {"completion_tokens": 3}}
        )
    except AssertionError:
        pass
    else:
        raise AssertionError("corrupt cumulative prefix accepted")
    rows = []
    for n, v, c in ((12, 3, 5), (24, 12, 4)):
        m = {
            "spec_verify_ct": v,
            "spec_num_proposed_drafts": v * 5,
            "spec_num_correct_drafts": c,
            "spec_accept_length": n / v,
            "spec_accept_rate": c / (v * 5),
        }
        row = {
            "output_ids": [0] * n,
            "duration_seconds": n / 2,
            "ttft_seconds": 1,
            "tpot_seconds": 0.1,
            "meta_info": m,
        }
        validate_stats(row, True)
        rows.append(row)
    pooled = aggregate(rows, True)
    assert (
        pooled["verify_weighted_accept_length"] == 36 / 15
        and pooled["accept_rate"] == 9 / 75
    )
    assert (
        pooled["verify_weighted_accept_length"]
        != sum(x["meta_info"]["spec_accept_length"] for x in rows) / 2
    )
    with tempfile.TemporaryDirectory() as temp:
        p = Path(temp)
        (p / "nfc").mkdir()
        (p / "nfc/common.pt").write_bytes(b"fake metadata, never torch.load")
        for i in range(32):
            (p / "nfc" / f"{i}.distcp").touch()
        marker = {
            "format": "specforge-nfc-fsdp-v1",
            "global_step": 25,
            "common_sha256": sha(p / "nfc/common.pt"),
        }
        (p / "training_state.json").write_text(json.dumps(marker))
        assert checkpoint_metadata(p, 25)["nfc/common.pt"] == marker["common_sha256"]
        (p / "nfc/common.pt").write_bytes(b"changed")
        try:
            checkpoint_metadata(p, 25)
        except AssertionError:
            pass
        else:
            raise AssertionError("corrupt common metadata accepted")
        (p / "nfc/common.pt").unlink()
        (p / "common.pt").write_bytes(b"fake metadata, never torch.load")
        try:
            checkpoint_metadata(p, 25)
        except FileNotFoundError:
            pass
        else:
            raise AssertionError("wrong root-level common layout accepted")
    print(
        json.dumps(
            {
                "passed": True,
                "scope": "multi-line SSE, comments, CRLF, DONE, cumulative/delta/final streams, corrupt-prefix rejection, verify-weighted counters",
            }
        )
    )


if __name__ == "__main__":
    main()
