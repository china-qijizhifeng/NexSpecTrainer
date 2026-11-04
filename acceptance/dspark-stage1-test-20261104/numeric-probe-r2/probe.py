"""Single finite validation probe: paired logprobs, then one supported reduction switch."""

import ast
import importlib.metadata
import importlib.util
import json
import os
import platform
import signal
import subprocess
import sys
import time
import urllib.request
from pathlib import Path

from common import ROOT, phase, public_stream, verify_execution, write
from evaluate import OPENER, URL, merge_ids, sse_events, stop, validate_completion
from probe_contract import (
    DISABLE,
    MODES,
    SOURCE,
    body_for,
    compare,
    ids_sha,
    numeric_rows,
    request_plan,
    selected_input,
    sha,
    validate_resolved,
    verify_source,
)
from worker_lifecycle import evaluation_signal_handler


def private_stream(path):
    descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    return os.fdopen(descriptor, "w")


def write_private(path, value):
    with private_stream(path) as stream:
        json.dump(value, stream, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())


def capture(entry, candidates, path):
    assert not path.exists()
    body = body_for(entry, candidates)
    req = urllib.request.Request(
        URL + "/generate",
        data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"},
    )
    begin = time.monotonic()
    ids, last, received = [], None, 0
    with private_stream(path.with_suffix(".events.jsonl")) as log:
        with OPENER.open(req, timeout=2400) as response:
            for event in sse_events(response):
                assert "error" not in event, "server returned an error"
                ids = merge_ids(ids, event)
                assert len(ids) <= entry["budget"]
                line = (
                    json.dumps(
                        {"elapsed_seconds": time.monotonic() - begin, "response": event}
                    )
                    + "\n"
                )
                received += len(line.encode())
                assert received <= 32 << 20, "bounded request evidence exceeded32MiB"
                log.write(line)
                log.flush()
                last = event
    assert last is not None
    validate_completion(ids, last["meta_info"], entry["budget"], len(body["input_ids"]))
    assert (
        last["meta_info"]["cached_tokens"] == last["meta_info"]["num_retractions"] == 0
    )
    result = {
        "output_ids": ids,
        "meta_info": last["meta_info"],
        "input_sha256": ids_sha(body["input_ids"]),
        "input_tokens": len(body["input_ids"]),
        "requested_max_new_tokens": entry["budget"],
        "duration_seconds": time.monotonic() - begin,
    }
    if entry["logprobs"]:
        numeric_rows(result, candidates)
    write_private(path, result)
    return result


def launch(mode, folder):
    folder.mkdir()
    folder.chmod(0o755)
    assert not any(
        "SIMULATE" in key and value not in ("", "0", "false", "False")
        for key, value in os.environ.items()
    ), "simulation environment present"
    old = json.loads((SOURCE / "target-only/launch.json").read_text())
    command = old["command"]
    assert command[:2] == ["sglang", "serve"] and DISABLE not in command
    assert "--speculative-algorithm" not in command
    if mode == "no-fused-allreduce":
        command = command + [DISABLE]
    write(
        folder / "launch.json",
        {
            "command": command,
            "single_variable": DISABLE if mode != "baseline" else None,
        },
    )
    log = public_stream(folder / "server.log", "x")
    process = subprocess.Popen(
        command, stdout=log, stderr=subprocess.STDOUT, start_new_session=True
    )
    write(
        folder / "server-process.json",
        {
            "pid": process.pid,
            "pgid": process.pid,
            "start_ticks": Path(f"/proc/{process.pid}/stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()[19],
        },
    )
    try:
        begin = time.monotonic()
        while time.monotonic() - begin < 5400:
            if process.poll() is not None:
                raise RuntimeError("owned server exited during readiness")
            try:
                with OPENER.open(URL + "/health", timeout=3) as response:
                    assert response.status == 200
            except OSError:
                time.sleep(5)
                continue
            with OPENER.open(URL + "/get_server_info", timeout=30) as response:
                # The full response can contain credentials: persist a strict whitelist only.
                info = validate_resolved(json.load(response), mode)
            write(folder / "resolved.json", info)
            return process, log
        raise TimeoutError("bounded readiness wait expired")
    except BaseException:
        stop(process, log)
        raise


def old_prompts():
    tree = ast.parse((SOURCE / "evaluate.py").read_text())
    main = next(
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and node.name == "main"
    )
    assignment = next(
        node
        for node in main.body
        if isinstance(node, ast.Assign)
        and any(
            isinstance(target, ast.Name) and target.id == "old"
            for target in node.targets
        )
    )
    rows = ast.literal_eval(assignment.value)
    assert [row["id"] for row in rows] == ["old-prompt-0", "old-prompt-1"]
    return rows


def environment():
    import torch
    from google.protobuf.internal.api_implementation import Type

    imported = json.loads((ROOT / "image-import/result.json").read_text())
    assert imported["passed"] and imported["node"] == os.environ["NODE_NAME"]
    assert (
        imported["manifest_digest"]
        == "sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa"
    )
    assert len(imported["gpus"]) == 8 and all("H100" in x for x in imported["gpus"])
    assert torch.cuda.device_count() == 8
    assert all(
        torch.cuda.get_device_capability(i) == (9, 0)
        and "H100" in torch.cuda.get_device_name(i)
        for i in range(8)
    )
    versions = {
        name: importlib.metadata.version(name)
        for name in ("sglang", "torch", "transformers", "protobuf", "flashinfer-python")
    }
    assert "38d9e4a3f" in versions["sglang"]
    source = Path(importlib.util.find_spec("sglang").origin).parent
    pins = json.loads((ROOT / "serving-source-pins.json").read_text())
    pins.update(json.loads((ROOT / "numeric-source-pins.json").read_text()))
    for name, digest in pins.items():
        assert sha(source / name) == digest, name
    for script, extra in (
        ("check_runner.py", []),
        ("check_eos.py", ["--serving-source", str(source)]),
    ):
        result = json.loads(
            subprocess.check_output(
                [sys.executable, str(ROOT / script), *extra], text=True
            )
        )
        assert result["passed"]
    write(
        ROOT / "environment.json",
        {
            "passed": True,
            "node": os.environ["NODE_NAME"],
            "hostname": platform.node(),
            "gpus": [torch.cuda.get_device_name(i) for i in range(8)],
            "versions": versions,
            "protobuf_backend": Type(),
            "source_pins": pins,
        },
    )


def run():
    verify_execution()
    verify_source()
    environment()
    row, prefix, candidates = selected_input()
    plan = request_plan(row, prefix)
    old = old_prompts()
    for entry in plan:
        if entry["kind"] == "history" and entry["input_ids"] is None:
            entry["input_ids"] = old[entry["diagnostic"]]["input_ids"]
        if entry["kind"] == "history":
            assert ids_sha(entry["input_ids"]) == entry["recorded"]["input_sha256"]
    summaries, modes = {}, {}
    for mode in MODES:
        folder = ROOT / mode
        phase("probe_server_starting", mode=mode)
        process, log = launch(mode, folder)
        try:
            outputs = []
            records = []
            for index, entry in enumerate(plan):
                phase(
                    "validation_numerical_probe",
                    mode=mode,
                    request_index=index,
                    kind=entry["kind"],
                )
                out = capture(entry, candidates, folder / f"request-{index:02d}.json")
                outputs.append(out)
                record = {
                    "index": index,
                    "kind": entry["kind"],
                    "input_sha256": out["input_sha256"],
                    "input_tokens": out["input_tokens"],
                    "output_tokens": len(out["output_ids"]),
                    "output_sha256": ids_sha(out["output_ids"]),
                    "finish_reason": out["meta_info"]["finish_reason"],
                }
                if entry["logprobs"]:
                    record["numeric_steps"] = numeric_rows(out, candidates)
                else:
                    original = entry["recorded"]
                    record["same_as_r7_recording"] = (
                        out["output_ids"] == original["output_ids"]
                        and out["meta_info"]["finish_reason"]
                        == original["meta_info"]["finish_reason"]
                    )
                records.append(record)
            summaries[mode] = {
                "requests": records,
                "original_decode_comparisons": [
                    compare(outputs[6], outputs[i]) for i in range(7, 10)
                ],
                "common_prefix_prefill_comparison": compare(outputs[10], outputs[11]),
            }
            modes[mode] = outputs
            write(folder / "summary.json", summaries[mode])
        finally:
            stop(process, log)
    result = {
        "probe_complete": True,
        "held_out_requests": 0,
        "gpu_aa_passed": False,
        "numerical_fix_validated": False,
        "request_audit": {
            "passed": True,
            "requests": sum(len(x) for x in modes.values()),
        },
        "modes": summaries,
        "cross_mode_original_decode": compare(modes[MODES[0]][6], modes[MODES[1]][6]),
        "cross_mode_common_prefix_prefill": compare(
            modes[MODES[0]][10], modes[MODES[1]][10]
        ),
        "scope": "Finite validation-only top-logprob probe; intervention and instrumentation can change execution. Prefill-prefix and incremental decode are distinct paths. Observations do not establish determinism or a kernel fix.",
    }
    assert result["request_audit"]["requests"] == 24
    write(ROOT / "result.json", result)
    phase("probe_complete", held_out_requests=0)


if __name__ == "__main__":
    os.umask(0o022)
    for signum in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
        signal.signal(signum, evaluation_signal_handler)
    try:
        run()
    except BaseException as error:
        write(
            ROOT / "eval-failure.json",
            {
                "type": type(error).__name__,
                "reason": str(error),
                "automatic_retry": False,
            },
        )
        phase("numerical_probe_failed", error_type=type(error).__name__)
        raise
