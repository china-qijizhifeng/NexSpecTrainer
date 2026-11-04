"""Frozen validation-only requests and bounded numeric evidence contracts."""

import hashlib
import json
import math
import struct
from pathlib import Path

SOURCE = Path("/gpfs/users/linlizhi/dspark-test-20261104-r7")
INPUT_ID = "podhmo__alchemyjsonschema__239822ad0685__rollout_0005"
INPUT_SHA = "38db48ae0a9e0cd58315b31edae50f04631466b0557d3d21d26ee7d6ba009296"
SOURCE_MANIFEST_SHA = "38a080efa6a37603725d8e0f0cbce67dc884ee8cd97b03ecf0950922880ed4e2"
MODES = ("baseline", "no-fused-allreduce")
DISABLE = "--enforce-disable-flashinfer-allreduce-fusion"
SERVER_FIELDS = (
    "attention_backend",
    "moe_runner_backend",
    "tp_size",
    "ep_size",
    "disable_radix_cache",
    "max_running_requests",
    "chunked_prefill_size",
    "cuda_graph_backend_prefill",
    "cuda_graph_backend_decode",
    "flashinfer_allreduce_fusion_backend",
    "enforce_disable_flashinfer_allreduce_fusion",
    "disable_custom_all_reduce",
    "enable_torch_symm_mem",
    "enable_deterministic_inference",
    "disable_overlap_schedule",
    "dsa_topk_backend",
)


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def ids_sha(ids):
    return hashlib.sha256(b"".join(struct.pack("<q", i) for i in ids)).hexdigest()


def verify_source():
    assert sha(SOURCE / "execution-sha256.json") == SOURCE_MANIFEST_SHA
    manifest = json.loads((SOURCE / "execution-sha256.json").read_text())
    assert len(manifest) == 860
    for name, digest in manifest.items():
        assert sha(SOURCE / name) == digest, name
    deleted = json.loads((SOURCE / "eval-delete.json").read_text())
    assert deleted["name"] == "job-6b4dd801c01d-20261104173837"
    assert deleted["response"]["success"]
    assert not (SOURCE / "target-only/aa-acceptance.json").exists()
    audit = json.loads((SOURCE / "diagnosis/aa-failure-audit.json").read_text())
    assert audit["audit_passed"] and audit["mismatch_before_eos_confirmed"]
    for name, digest in audit["evidence_sha256"].items():
        assert sha(SOURCE / name) == digest, name
    references = json.loads(
        (Path(__file__).resolve().parent / "reference-sha256.json").read_text()
    )
    for name, digest in references.items():
        assert sha(SOURCE / name) == digest, name
    for group in ("target-only", "official-mtp", "step25", "step118"):
        assert not any((SOURCE / group).glob("test-*.json"))


def selected_input(source=SOURCE):
    matches = [
        json.loads(line)
        for line in (source / "validation-inputs.jsonl").read_text().splitlines()
        if json.loads(line)["id"] == INPUT_ID
    ]
    (row,) = matches
    assert len(row["input_ids"]) == 50527 and ids_sha(row["input_ids"]) == INPUT_SHA
    left, right = [
        json.loads((source / "target-only" / name).read_text())
        for name in ("diagnostic-3.json", "diagnostic-3-repeat.json")
    ]
    assert left["input_sha256"] == right["input_sha256"] == INPUT_SHA
    assert left["output_ids"][:19] == right["output_ids"][:19]
    assert left["output_ids"][19] != right["output_ids"][19]
    prefix = left["output_ids"][:19]
    assert 1 not in prefix
    return row, prefix, [left["output_ids"][19], right["output_ids"][19]]


def request_plan(row, prefix, source=SOURCE):
    plan = []
    for diagnostic in range(3):
        for repeat in ("", "-repeat"):
            recorded = json.loads(
                (
                    source / "target-only" / f"diagnostic-{diagnostic}{repeat}.json"
                ).read_text()
            )
            if diagnostic < 2:
                # Same fixed old prompt IDs, obtained from frozen evaluator source at runtime.
                inputs = None
            else:
                inputs = next(
                    json.loads(line)["input_ids"]
                    for line in (source / "validation-inputs.jsonl")
                    .read_text()
                    .splitlines()
                    if json.loads(line)["id"] == recorded["id"]
                )
            plan.append(
                {
                    "kind": "history",
                    "diagnostic": diagnostic,
                    "repeat": repeat,
                    "recorded": recorded,
                    "input_ids": inputs,
                    "budget": 1024,
                    "logprobs": False,
                }
            )
    for _ in range(4):
        plan.append(
            {
                "kind": "original-decode",
                "input_ids": row["input_ids"],
                "budget": 32,
                "logprobs": True,
            }
        )
    for _ in range(2):
        plan.append(
            {
                "kind": "common-prefix-prefill",
                "input_ids": row["input_ids"] + prefix,
                "budget": 1,
                "logprobs": True,
            }
        )
    assert len(plan) == 12
    return plan


def body_for(entry, candidates):
    body = {
        "input_ids": entry["input_ids"],
        "stream": True,
        "sampling_params": {
            "temperature": 0,
            "ignore_eos": False,
            "max_new_tokens": entry["budget"],
        },
    }
    assert body["input_ids"] and 1 <= entry["budget"] <= 1024
    if entry["logprobs"]:
        body.update(
            return_logprob=True,
            logprob_start_len=-1,
            top_logprobs_num=10,
            token_ids_logprob=candidates,
            return_text_in_logprobs=False,
        )
    return body


def numeric_rows(response, candidates):
    ids, meta = response["output_ids"], response["meta_info"]
    top = meta["output_top_logprobs"]
    selected = meta["output_token_logprobs"]
    specific = meta["output_token_ids_logprobs"]
    assert len(top) == len(selected) == len(specific) == len(ids)
    rows = []
    for token, entries, chosen, pair in zip(ids, top, selected, specific):
        assert len(entries) == 10 and len(pair) == 2
        assert len({item[1] for item in entries}) == 10
        assert {item[1] for item in pair} == set(candidates)
        assert chosen[1] == token
        for item in [*entries, chosen, *pair]:
            assert len(item) == 3 and item[2] is None
            assert math.isfinite(item[0]) and item[0] <= 1e-5
        values = {item[1]: item[0] for item in entries}
        assert token in values and math.isclose(
            values[token], max(values.values()), abs_tol=1e-5
        )
        assert math.isclose(chosen[0], values[token], abs_tol=1e-5)
        ranked = sorted(values.values(), reverse=True)
        pair_values = {item[1]: item[0] for item in pair}
        rows.append(
            {
                "top1_top2_margin": ranked[0] - ranked[1],
                "historical_candidate_margin": pair_values[candidates[0]]
                - pair_values[candidates[1]],
            }
        )
    return rows


def compare(left, right):
    a, b = left["output_ids"], right["output_ids"]
    divergence = next(
        (i for i, (x, y) in enumerate(zip(a, b)) if x != y), min(len(a), len(b))
    )
    equal = a == b
    comparable = min(len(a), len(b), divergence + (not equal))
    differences = []
    for aa, bb in zip(
        left["meta_info"]["output_token_ids_logprobs"][:comparable],
        right["meta_info"]["output_token_ids_logprobs"][:comparable],
    ):
        va, vb = ({x[1]: x[0] for x in rows} for rows in (aa, bb))
        assert va.keys() == vb.keys()
        differences.extend(abs(va[t] - vb[t]) for t in va)
    return {
        "outputs_equal": equal,
        "output_tokens": [len(a), len(b)],
        "first_difference_zero_based": None if equal else divergence,
        "same_input_prefix_logprob_steps": comparable,
        "max_abs_candidate_logprob_difference": max(differences, default=None),
        "scope": "Compare logprobs only through first output divergence; later prefixes differ.",
    }


def validate_job(job):
    from node_contract import validate_job as check

    check(job)
    alias = "denovo-dspark-aa-numerics-20261104-r1"
    assert (
        job["metadata"]["name"] == job["metadata"]["annotations"]["JobAlias"] == alias
    )
    assert job["metadata"]["namespace"] == "qiji"
    pod = job["spec"]["tasks"][0]["template"]["spec"]
    root = "/gpfs/users/linlizhi/dspark-aa-numerics-20261104-r1"
    worker = pod["containers"][0]
    assert worker["command"] == ["/bin/bash", root + "/run_eval.sh"]
    assert (
        worker["image"]
        == "lmsysorg/sglang@sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa"
    )
    assert worker["imagePullPolicy"] == "Never"
    assert worker["volumeMounts"] == [
        {"name": "gpfs", "mountPath": "/gpfs"},
        {"name": "shm", "mountPath": "/dev/shm"},
    ]
    (init,) = pod["initContainers"]
    assert init["command"][:3] == ["/bin/bash", "-c", init["command"][2]]
    assert root + "/image-import/invocation-started" in init["command"][2]
    assert root + "/import_worker_image.py" in init["command"][2]
    task = next(v for v in pod["volumes"] if v["name"] == "task")
    assert task["hostPath"]["path"] == root
    assert not any("dspark-test-20261104-r7" in str(value) for value in pod.values())


def validate_resolved(info, mode):
    assert info["attention_backend"] == "dsv4"
    assert info["moe_runner_backend"] == "flashinfer_mxfp4"
    assert info["tp_size"] == info["ep_size"] == 8
    assert info["disable_radix_cache"] is True
    assert info["max_running_requests"] == 1
    assert info["chunked_prefill_size"] == -1
    assert info["enable_deterministic_inference"] is False
    assert not info.get("speculative_algorithm")
    assert (
        info["cuda_graph_backend_prefill"]
        == info["cuda_graph_backend_decode"]
        == "disabled"
    )
    if mode == "no-fused-allreduce":
        assert info["enforce_disable_flashinfer_allreduce_fusion"] is True
        assert info["flashinfer_allreduce_fusion_backend"] is None
    else:
        assert not info["enforce_disable_flashinfer_allreduce_fusion"]
        assert info["flashinfer_allreduce_fusion_backend"] == "auto"
    return {name: info.get(name) for name in SERVER_FIELDS}
