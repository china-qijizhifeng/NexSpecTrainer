"""Start one official Flash/dSpark worker; publish readiness only after native checks."""

import hashlib
import importlib.metadata
import importlib.util
import json
import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
OUT = ROOT / os.environ["WORKER_ID"]
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
KEY = (ROOT / "private" / "worker-key").read_text().strip()


def sanitize(value):
    if isinstance(value, dict):
        return {
            k: ("[REDACTED]" if "key" in k.lower() else sanitize(v))
            for k, v in value.items()
        }
    if isinstance(value, list):
        return [sanitize(v) for v in value]
    if isinstance(value, str):
        return value.replace(KEY, "[REDACTED]")
    return value


def write(name, value):
    target = OUT / name
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(sanitize(value), indent=2) + "\n")
    temporary.replace(target)


def phase(name, **details):
    write("phase.json", {"stage": name, "time": time.time(), **details})


def request(path, body=None, authenticated=True):
    headers = {"Content-Type": "application/json"}
    if authenticated:
        headers["Authorization"] = "Bearer " + KEY
    req = urllib.request.Request(
        "http://127.0.0.1:8000" + path,
        data=None if body is None else json.dumps(body).encode(),
        headers=headers,
    )
    return OPENER.open(req, timeout=300)


def inspect():
    assert os.getuid() == 1006
    assert os.environ["NODE_NAME"] == os.environ["EXPECTED_NODE"]
    assert os.environ["NODE_NAME"] in ("node205", "node207")
    imported = json.loads((OUT / "image-import" / "result.json").read_text())
    profile = json.loads((ROOT / "candidate-profile.json").read_text())
    assert imported["passed"] and imported["node"] == os.environ["NODE_NAME"]
    assert imported["source"].removeprefix("docker.io/") == profile["image"]
    import torch
    from google.protobuf.internal.api_implementation import Type

    assert torch.cuda.device_count() == 8
    assert all(torch.cuda.get_device_capability(i) == (9, 0) for i in range(8))
    assert all("H100" in torch.cuda.get_device_name(i) for i in range(8))
    versions = {
        n: importlib.metadata.version(n)
        for n in ("sglang", "torch", "protobuf", "flashinfer-python", "deepseek-recipe")
    }
    assert "38d9e4a3f" in versions["sglang"]
    assert versions["deepseek-recipe"] == "0.2.0"
    package = Path(importlib.util.find_spec("sglang").origin).parent
    pins = json.loads((ROOT / "serving-source-pins.json").read_text())
    for name, digest in pins.items():
        assert hashlib.sha256((package / name).read_bytes()).hexdigest() == digest, name
    target = Path(profile["model_path"])
    provenance = json.loads((ROOT / "model-provenance.json").read_text())
    for name, digest in provenance["sha256"].items():
        assert hashlib.sha256((target / name).read_bytes()).hexdigest() == digest, name
    download = target / ".download"
    complete = json.loads((download / "complete.json").read_text())
    identity = json.loads((download / "identity.json").read_text())
    assert complete["weight_sha256_verified"]
    assert complete["revision"] == identity["revision"] == profile["model_revision"]
    assert complete["repo"] == identity["repo"] == profile["model"]
    receipts = json.loads((download / "verified-files.json").read_text())
    weights = {name for name in receipts if name.endswith(".safetensors")}
    assert len(weights) == 48
    assert all(
        (target / name).stat().st_size == receipts[name]["bytes"] for name in weights
    )
    assert json.loads((target / "config.json").read_text())["eos_token_id"] == 1
    assert not any(
        "SIMULATE" in k or "SIM_DRAW_AHEAD" in k or "TP_SYNC_REDUCTION" in k
        for k in os.environ
    )
    router = importlib.util.find_spec("sglang_router")
    if router is not None:
        folder = Path(router.origin).parent
        for source in folder.rglob("*.py"):
            destination = OUT / "router-source" / source.relative_to(folder)
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(source.read_bytes())
        help_result = subprocess.run(
            [sys.executable, "-m", "sglang_router.launch_router", "--help"],
            capture_output=True,
            text=True,
            timeout=120,
        )
        (OUT / "router-help.txt").write_text(help_result.stdout + help_result.stderr)
        assert help_result.returncode == 0
    write(
        "environment.json",
        {
            "node": os.environ["NODE_NAME"],
            "gpus": [torch.cuda.get_device_name(i) for i in range(8)],
            "versions": versions,
            "protobuf_backend": Type(),
            "source_sha256": pins,
            "model_identity": identity,
            "weight_identity_scope": "existing full-SHA download receipt plus current 48 shard sizes; no new full weight hash",
            "router_installed": router is not None,
        },
    )


def smoke(profile):
    try:
        with request("/v1/models", authenticated=False):
            raise AssertionError("unauthenticated model API was allowed")
    except urllib.error.HTTPError as exc:
        assert exc.code == 401, exc.code
    with request("/v1/models") as response:
        models = json.load(response)
    assert profile["northgate_model"] in {m["id"] for m in models["data"]}
    with request("/get_server_info") as response:
        info = json.load(response)
    write("server-info.json", info)
    body = {
        "model": profile["northgate_model"],
        "messages": [{"role": "user", "content": "Reply with a short greeting."}],
        "max_tokens": 256,
        "temperature": 0,
        "chat_template_kwargs": {"thinking": False},
    }
    with request("/v1/chat/completions", body) as response:
        chat = json.load(response)
    choice = chat["choices"][0]
    assert choice["finish_reason"] in ("stop", "length")
    assert choice["message"].get("content")
    assert chat["usage"]["completion_tokens"] > 0
    events, done, text = 0, False, []
    with request("/v1/chat/completions", {**body, "stream": True}) as response:
        for raw in response:
            line = raw.decode().strip()
            if not line.startswith("data:"):
                continue
            payload = line[5:].strip()
            if payload == "[DONE]":
                done = True
                break
            event = json.loads(payload)
            assert "error" not in event
            events += 1
            for item in event.get("choices", []):
                text.append(item.get("delta", {}).get("content") or "")
    assert done and events > 0 and "".join(text)
    write(
        "smoke.json",
        {
            "passed": True,
            "scope": "worker authentication/models/chat/nonstream/SSE smoke only; tools, reasoning, concurrency, EOS/count and gateway acceptance still required",
            "chat_finish_reason": choice["finish_reason"],
            "chat_usage": chat["usage"],
            "stream_events": events,
            "stream_done": done,
            "lossless_greedy_verified": False,
        },
    )


def main():
    from execution_contract import verify_execution

    verify_execution(ROOT)
    OUT.mkdir(exist_ok=True)
    with (OUT / "worker-started.json").open("x") as file:
        json.dump({"time": time.time(), "node": os.environ["NODE_NAME"]}, file)
    phase("environment_preflight")
    inspect()
    from with_kernel_cache import cache_environment

    env = dict(os.environ)
    settings, directories = cache_environment(
        OUT / "cache", "sglang-c6da252c6837-sm90", {}
    )
    for path in set(directories.values()):
        Path(path).mkdir(parents=True, exist_ok=True)
    env.update(settings)
    phase("server_starting")
    child = subprocess.Popen(
        [sys.executable, str(ROOT / "launch_private.py")],
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        env=env,
        start_new_session=True,
    )
    write(
        "server-process.json",
        {
            "pid": child.pid,
            "start_ticks": Path(f"/proc/{child.pid}/stat")
            .read_text()
            .rsplit(")", 1)[1]
            .split()[19],
        },
    )

    def log_output():
        with (OUT / "server.log").open("a") as log:
            for line in child.stdout:
                log.write(line.replace(KEY, "[REDACTED]"))
                log.flush()

    thread = threading.Thread(target=log_output, daemon=True)
    thread.start()
    stopped = False

    def stop(_signum, _frame):
        nonlocal stopped
        stopped = True
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    try:
        deadline = time.monotonic() + 2400
        ready = False
        while time.monotonic() < deadline and not stopped:
            assert child.poll() is None, f"server exited {child.returncode}"
            try:
                with request("/health", authenticated=False) as response:
                    ready = response.status == 200
            except (urllib.error.URLError, TimeoutError):
                pass
            if ready:
                break
            time.sleep(5)
        assert ready and not stopped, "server readiness deadline or cancellation"
        phase("worker_smoke")
        smoke(json.loads((ROOT / "candidate-profile.json").read_text()))
        phase("worker_ready", northgate_published=False, full_acceptance=False)
        while not stopped:
            assert child.poll() is None, f"server exited {child.returncode}"
            time.sleep(10)
    finally:
        if child.poll() is None:
            os.killpg(child.pid, signal.SIGTERM)
            try:
                child.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid, signal.SIGKILL)
                child.wait()
        thread.join(timeout=5)
        write(
            "server-exit.json",
            {"returncode": child.returncode, "time": time.time(), "cancelled": stopped},
        )


if __name__ == "__main__":
    os.umask(0o077)
    try:
        main()
    except BaseException as error:
        write(
            "failure.json",
            {
                "time": time.time(),
                "type": type(error).__name__,
                "error": str(error),
                "automatic_retry": False,
            },
        )
        phase("worker_failed")
        raise
