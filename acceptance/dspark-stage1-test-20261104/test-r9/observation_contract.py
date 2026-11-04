"""User-authorized numerical tolerance; execution and counting guards stay strict."""

from common import write


def private_stream(path):
    import os

    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "w")


def write_private(path, value):
    import json
    import os

    with private_stream(path) as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def validate_amendment(protocol):
    amendment = protocol["observational_amendment"]
    assert amendment["allow_aa_output_mismatch"] is True
    assert amendment["allow_historical_reference_mismatch"] is True
    assert amendment["prior_held_out_requests"] == 0
    assert amendment["lossless_claim_allowed"] is False
    assert amendment["retain_eos_http_counts_finite_input_identity_guards"] is True
    assert protocol["variants"] == ["target-only", "official-mtp", "step25", "step118"]
    assert protocol["generation"] == {
        "temperature": 0,
        "ignore_eos": False,
        "max_new_tokens": 1024,
        "concurrency": 1,
        "gamma": 5,
        "eos_token_ids": [1],
    }


def comparison(left, right):
    assert left["id"] == right["id"] and left["input_sha256"] == right["input_sha256"]
    a, b = left["output_ids"], right["output_ids"]
    common = next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )
    return {
        "id": left["id"],
        "input_sha256": left["input_sha256"],
        "output_tokens": [len(a), len(b)],
        "match_tokens": common,
        "outputs_equal": a == b,
        "finish_reasons_equal": left["meta_info"]["finish_reason"]
        == right["meta_info"]["finish_reason"],
        "first_difference_zero_based": None if a == b else common,
    }


def record_aa(folder, rows, reference):
    assert len(rows) == 10 and len({r["id"] for r in rows}) == 10
    same = all(row["outputs_equal"] and row["finish_reasons_equal"] for row in rows)
    write(
        folder / "aa-observation.json",
        {
            "diagnostics": 10,
            "all_equal": same,
            "samples": rows,
            "historical_reference_matches": reference["passed"],
            "output_difference_permitted": True,
            "user_authorized": True,
            "claim": "observational measurement only; equality is recorded, not assumed",
        },
    )
    # A failed equality gate must never be rewritten as acceptance.
    if same and reference["passed"]:
        write(
            folder / "aa-acceptance.json",
            {
                "passed": True,
                "diagnostics": 10,
                "complete_outputs_and_finish_reasons_match": True,
                "historical_reference_matches": True,
            },
        )


def validate_resolved(info, group):
    assert info["attention_backend"] == "dsv4"
    assert info["moe_runner_backend"] == "flashinfer_mxfp4"
    assert info["tp_size"] == info["ep_size"] == 8
    assert info["disable_radix_cache"] is True and info["max_running_requests"] == 1
    assert info["chunked_prefill_size"] == -1
    assert (
        info["cuda_graph_backend_prefill"]
        == info["cuda_graph_backend_decode"]
        == "disabled"
    )
    assert info["enable_deterministic_inference"] is False
    algorithm = info.get("speculative_algorithm")
    assert (algorithm == "DSPARK") if group != "target-only" else not algorithm
    if group != "target-only":
        assert info["speculative_dspark_block_size"] == 5
    fields = (
        "attention_backend",
        "moe_runner_backend",
        "tp_size",
        "ep_size",
        "disable_radix_cache",
        "max_running_requests",
        "chunked_prefill_size",
        "cuda_graph_backend_prefill",
        "cuda_graph_backend_decode",
        "enable_deterministic_inference",
        "speculative_algorithm",
        "speculative_dspark_block_size",
        "flashinfer_allreduce_fusion_backend",
        "enforce_disable_flashinfer_allreduce_fusion",
    )
    return {name: info.get(name) for name in fields}
