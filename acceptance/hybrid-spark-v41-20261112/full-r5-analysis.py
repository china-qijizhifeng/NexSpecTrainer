"""Summarize r5 GPU evidence, retaining the reused-baseline limitations."""
import hashlib
import itertools
import json
from pathlib import Path
import re
import shutil


BASE = Path(__file__).resolve().parent
ARTIFACTS = BASE / "full-r5-artifacts"
FULL = ARTIFACTS / "full-model"
SOURCE = "2b6a293637efc40d2834a21af6c87580b13f9498"
RUN_UUID = "195ac64f-c389-47e4-99c5-b546632fde5b"
IDS = dict(gpu="j-1794486677-162329", control="j-1794486635-862619", observer="j-1794486784-536689")


def read(path):
    return json.loads(path.read_text())


def rows(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def digest(value):
    return hashlib.sha256(json.dumps(value, separators=(",", ":")).encode()).hexdigest()


def scheduler_failure():
    path = FULL / "hybrid/server.log"
    text = path.read_text() if path.exists() else ""
    match = re.search(r"^\[([^\]]+)\] Scheduler hit an exception: (Traceback.*?)\n([A-Za-z]\w*(?:Error|Exception)): ([^\n]+)", text, re.M | re.S)
    if match:
        return dict(log_header=match[1], error_type=match[3], message=match[4], stack=match[2] + "\n" + match[3] + ": " + match[4])
    return None


def gate_failure():
    path = BASE / "full-r5-gpu-final-GetLogs.json"
    if not path.exists():
        return None
    items = sorted(read(path)["Data"]["Items"], key=lambda row: int(row["Timestamp"]))
    lines = [row["Line"].split(" >> ", 1)[-1] for row in items]
    begin = next((i for i, line in enumerate(lines) if line.startswith("Traceback (most recent call last):")), None)
    end = next((i for i, line in enumerate(lines) if line.startswith("RuntimeError: T=0 output divergence recorded;")), None)
    if begin is not None and end is not None:
        return dict(scope="strict output-equality gate after all five new nsp eval Runs succeeded",
            timestamp_ns=items[end]["Timestamp"], error_type="RuntimeError", message=lines[end].split(": ", 1)[1],
            stack="\n".join(lines[begin:end + 1]), source=str(path.relative_to(BASE)), source_sha256=sha(path))
    return None


record = read(ARTIFACTS / "run.json")
assert record["uuid"] == RUN_UUID
prompts = {row["id"]: row for row in rows(FULL / "measured-prompts.jsonl")}
runs, variant_data = {}, {}
for variant in ("target", "dspark"):
    path = FULL / "baseline-receipts" / variant / "requests.jsonl"
    if path.exists():
        variant_data[variant] = {row["sample_id"]: row for row in rows(path)}
for path in sorted((FULL / "nsp/runs").glob("*/run.json")):
    run = read(path)
    attempt = path.parent / "attempts" / f'{run["attempt"]:04d}'
    status = read(attempt / "status.json")
    runs[run["name"]] = dict(id=run["id"], kind=run["kind"], state=status["state"],
        exit_code=status.get("exit_code"), error=status.get("error"))
    if run["name"] == "hybrid-measured" and (attempt / "requests.jsonl").exists():
        variant_data["hybrid"] = {row["sample_id"]: row for row in rows(attempt / "requests.jsonl")}

comparisons = {}
for left, right in itertools.combinations(variant_data, 2):
    a, b = variant_data[left], variant_data[right]
    suites, differences = {}, []
    same_coverage = a.keys() == b.keys() == prompts.keys()
    for sample_id in sorted(a.keys() & b.keys()):
        assert a[sample_id]["input_ids"] == b[sample_id]["input_ids"] == prompts[sample_id]["input_ids"]
        x, y = a[sample_id]["output_ids"], b[sample_id]["output_ids"]
        count = suites.setdefault(prompts[sample_id]["suite"], dict(equal=0, total=0))
        count["total"] += 1
        count["equal"] += x == y
        if x != y:
            first = next((i for i, pair in enumerate(zip(x, y)) if pair[0] != pair[1]), min(len(x), len(y)))
            differences.append(dict(sample_id=sample_id, first_divergence=first,
                left_length=len(x), right_length=len(y), left_tail=x[first:], right_tail=y[first:]))
    comparisons[left + "-vs-" + right] = dict(equal=same_coverage and not differences,
        complete_coverage=same_coverage, requests=len(a.keys() & b.keys()), suites=suites,
        differences=sorted(differences, key=lambda row: row["sample_id"]))

diagnostics, cycle_summaries = {}, {}
for variant, data in variant_data.items():
    directory = FULL / ("baseline-receipts/" + variant if variant != "hybrid" else variant)
    path = directory / "diagnostics/summary.json"
    if path.exists():
        d = read(path)
        original = [row["output_ids"] for row in d["receipts"]["original"]]
        prefix = [row["output_ids"] for row in d["receipts"]["fixed-prefix"]]
        primary = data["reasoning-no-match"]["output_ids"]
        diagnostics[variant] = dict(original_repeatable=d["original_repeatable"], prefix_repeatable=d["prefix_repeatable"],
            original_outputs=original, prefix_outputs=prefix, primary_reasoning_output=primary,
            original_matches_primary=[value == primary for value in original],
            fixed_prefix_input_sha256=digest(d["receipts"]["fixed-prefix"][0]["input_ids"]),
            fixed_prefix_input_tokens=len(d["receipts"]["fixed-prefix"][0]["input_ids"]))
    path = directory / "observed-measured-cycles.json"
    if path.exists():
        observed = read(path)
        assert {row["sample_id"] for row in observed} == data.keys()
        by_suite = {}
        for row in observed:
            output = data[row["sample_id"]]["output_ids"]
            assert row["prefill_tokens"] + sum(c["committed_output_tokens"] for c in row["cycles"]) == len(output)
            entry = by_suite.setdefault(prompts[row["sample_id"]]["suite"], dict(requests=0, output_tokens=0, verify_cycles=0))
            entry["requests"] += 1
            entry["output_tokens"] += len(output)
            entry["verify_cycles"] += len(row["cycles"])
        for value in by_suite.values():
            value["al"] = value["output_tokens"] / value["verify_cycles"] if value["verify_cycles"] else None
        cycle_summaries[variant] = dict(requests=len(observed), total_cycles=sum(len(r["cycles"]) for r in observed),
            maximum_commit=max(c["committed_output_tokens"] for r in observed for c in r["cycles"]), suites=by_suite)

terminals = {}
for kind, job_id in IDS.items():
    parent = read(BASE / f"full-r5-{kind}-final-GetJob.json")["Data"]
    replicas = read(BASE / f"full-r5-{kind}-final-ListJobReplicas.json")["Data"]["Items"]
    terminals[kind] = dict(job_id=job_id, phase=parent["Phase"], replicas=[
        {key: row.get(key) for key in ("Name", "ReplicaIndex", "Phase", "NodeName", "EndTime")} for row in replicas])

main = read(FULL / "summary.json") if (FULL / "summary.json").exists() else None
commits = read(FULL / "long-commits.json") if (FULL / "long-commits.json").exists() else None
reuse = read(FULL / "baseline-reuse.json")
main_summary = dict(main) if main else None
if main_summary:
    main_summary["results"] = {variant: {key: serving[key] for key in
        ("complete", "request_count", "overall", "sampling", "prompts_sha256") if key in serving}
        for variant, serving in main["results"].items()}
    main_summary["original_receipt"] = dict(relative_path="full-model/summary.json", sha256=sha(FULL / "summary.json"))
inventory = {str(path.relative_to(ARTIFACTS)): dict(bytes=path.stat().st_size, sha256=sha(path))
             for path in sorted(ARTIFACTS.rglob("*")) if path.is_file()}
report = dict(run_uuid=RUN_UUID, source_commit=SOURCE, run_dir="/mnt/work/shom/training_runs/hybrid-v41-full-2b6a293-r5",
    gpu_job_id=IDS["gpu"], terminal_phase=record["yicloud"].get("terminal_phase"),
    terminal_reconciled_at=record["yicloud"].get("terminal_reconciled_at"),
    gate_passed=bool(main and main["passed"]), throughput_comparison_valid=False,
    first_failure=gate_failure(), scheduler_failure=scheduler_failure(), main_summary=main_summary,
    readiness_seconds=read(FULL / "hybrid/readiness.json")["seconds"], nsp_runs=runs, comparisons=comparisons,
    diagnostics=diagnostics, observed_cycles=cycle_summaries,
    long_commits=dict(total_cycles=commits["total_cycles"], beyond_mtp_commits=len(commits["beyond_mtp_commits"]),
        followed_by_decode=len(commits["long_commits_followed_by_decode"]),
        maximum=max((r["committed_output_tokens"] for r in commits["beyond_mtp_commits"]), default=None)) if commits else None,
    baseline_reuse=dict(relative_path="full-model/baseline-reuse.json", sha256=sha(FULL / "baseline-reuse.json"),
        baseline_commit=reuse["baseline_commit"], baseline_runtime_manifest_sha256=reuse["baseline_runtime_manifest_sha256"],
        current_runtime_manifest_sha256=reuse["current_runtime_manifest_sha256"], evaluated_runs=reuse["evaluated_runs"],
        calibrator_sha256=reuse["calibrator_sha256"], calibration_provenance=reuse["calibration_provenance"], scope=reuse["scope"]),
    resource_terminals=terminals, artifact_count=len(inventory), artifact_bytes=sum(v["bytes"] for v in inventory.values()),
    claim_limit="Target/DSPARK correctness receipts are reused from r4; different GPU allocation prohibits throughput comparison. Strict three-way token equality and a measured long commit followed by decode remain required. Serial/prefix probes do not establish logit margins or numerical parity.")
assert all(row["phase"] in {"Failed", "Succeeded"} and len(row["replicas"]) == 1 and
           row["replicas"][0]["Phase"] in {"Failed", "Succeeded"} for row in terminals.values())
resources = dict(run_uuid=RUN_UUID, source_commit=SOURCE, terminal_reconciled_at=report["terminal_reconciled_at"],
    runtime_finished=read(ARTIFACTS / "logs/runtime-finished-replica-0.json"),
    expected_replicas_per_job=1, active_replicas=0, resource_terminals=terminals,
    source_receipts={f"full-r5-{kind}-final-{action}.json": sha(BASE / f"full-r5-{kind}-final-{action}.json")
        for kind in IDS for action in ("GetJob", "ListJobReplicas")},
    managed_run_receipt=dict(relative_path="full-r5-artifacts/run.json", sha256=sha(ARTIFACTS / "run.json")))
(BASE / "full-r5-resource-terminals.json").write_text(json.dumps(resources, indent=2) + "\n")
for name in ("target-vs-dspark.json", "target-vs-hybrid.json", "dspark-vs-hybrid.json", "summary.json"):
    shutil.copy2(FULL / name, BASE / ("full-r5-" + name))
if commits:
    assert commits["total_cycles"] == cycle_summaries["hybrid"]["total_cycles"]
    boundary = [row for row in commits["long_commits_followed_by_decode"]
        if row["commit"]["ctx_len"] < 512 <= row["commit"]["ctx_len"] + row["commit"]["committed_output_tokens"]]
    for row in commits["long_commits_followed_by_decode"]:
        assert row["next_cycle"]["cycle_id"] == row["commit"]["cycle_id"] + 1
        assert row["next_cycle"]["ctx_len"] == row["commit"]["ctx_len"] + row["commit"]["committed_output_tokens"]
    compact = dict(run_uuid=RUN_UUID, source_commit=SOURCE, nsp_run=runs["hybrid-measured"],
        telemetry=commits["telemetry"], **report["long_commits"], long_commit_followed_by_decode_gate_passed=bool(commits["long_commits_followed_by_decode"]),
        strict_three_way_equality_gate_passed=report["gate_passed"],
        first_continued_long_commit=commits["long_commits_followed_by_decode"][0],
        continued_long_commits_crossing_ctx512=len(boundary), ctx512_examples=boundary[:2],
        source_receipts={"full-r5-artifacts/full-model/" + name: sha(FULL / name) for name in
            ("long-commits.json", "hybrid/observed-measured-cycles.json")},
        scope="Actual measured greedy verify commits; fixed budget 15 extra slots per request; no throughput or calibration-quality claim")
    (BASE / "full-r5-long-commit-evidence.json").write_text(json.dumps(compact, indent=2) + "\n")
(BASE / "full-r5-artifact-index.json").write_text(json.dumps(inventory, indent=2) + "\n")
(BASE / "full-r5-analysis.json").write_text(json.dumps(report, indent=2) + "\n")
(BASE / ("full-r5-result.json" if report["gate_passed"] else "full-r5-failure.json")).write_text(json.dumps(report, indent=2) + "\n")
print(json.dumps({key: report[key] for key in ("run_uuid", "gate_passed", "first_failure", "comparisons", "long_commits", "resource_terminals")}, indent=2))
