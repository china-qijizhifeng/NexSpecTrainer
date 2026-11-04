"""Enable real decode replay and account for prefix hits without altering inputs."""

import re

from common import write


def validate_cache_counts(meta, input_tokens):
    cached = meta.get("cached_tokens", 0)
    assert type(cached) is int and 0 <= cached <= input_tokens
    assert meta["prompt_tokens"] == input_tokens


def graph_receipt(folder):
    lines = (folder / "server.log").read_text().splitlines()
    decode = [line for line in lines if "Decode batch" in line]
    replayed = [line for line in decode if re.search(r"cuda graph: True", line)]
    assert replayed, "decode CUDA graph requested but no actual replay observed"
    write(
        folder / "decode-graph-evidence.json",
        {
            "passed": True,
            "actual_decode_graph_observed": True,
            "decode_log_samples": len(decode),
            "graph_replay_log_samples": len(replayed),
            "non_graph_decode_log_samples": len(decode) - len(replayed),
            "scope": "Pinned scheduler logs report actual forward graph execution",
        },
    )
