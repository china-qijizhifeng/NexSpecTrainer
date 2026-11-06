"""Isolated full-model 128K capture and non-CP parity acceptance."""

import hashlib
import json
import os
import signal
import socket
import subprocess
import time
import traceback
from pathlib import Path

import torch

from ops.northgate_capture import retest
from specforge.data.feature_file import load_hybrid_features

CAPACITY = 135168
RUN = Path(os.environ["NORTHGATE_CAPTURE_RUN_DIR"])
ROOT = retest.ROOT
RANK = int(os.environ["KUBEBRAIN_REPLICA"])
INPUTS = retest.DATA / "cp128k-inputs"


def wait_file(path, seconds=1800):
    deadline = time.monotonic() + seconds
    while not path.exists():
        failures = list(RUN.glob("worker-*/REJECTED.json"))
        if failures:
            raise RuntimeError(f"Peer failed: {failures[0]}")
        if time.monotonic() > deadline:
            raise TimeoutError(str(path))
        time.sleep(2)
    return json.loads(path.read_text())


def probe(sequence):
    data = Path(__file__).with_name("cp128k_probe.py").read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    control = ROOT / "control"
    (control / "modules").mkdir(exist_ok=True)
    target = control / "modules" / f"{digest}.py"
    if not target.exists():
        target.write_bytes(data)
    retest.write(
        control / "command.json",
        {"sequence": sequence, "source": f"modules/{digest}.py", "sha256": digest},
    )
    deadline = time.monotonic() + 180
    while time.monotonic() < deadline:
        retest.request("/server_info", timeout=120)
        paths = list((control / "commands" / sequence).glob("rank*.json"))
        if len(paths) == 8:
            ranks = [json.loads(p.read_text()) for p in paths]
            assert {r["rank"] for r in ranks} == set(range(8))
            assert all(
                r["success"] and r["parameter_pointers_versions_unchanged"]
                for r in ranks
            ), ranks
            return ranks
        time.sleep(0.2)
    raise TimeoutError("all-rank probe receipts")


def gate(n, repeat):
    name = f"n{n}-r{repeat}"
    own = ROOT / "captures" / f"{name}.json"
    other = RUN / f"worker-{1 - RANK}/captures/{name}.json"
    a, b = wait_file(own), wait_file(other)
    decision = RUN / f"GATE-{name}.json"
    if RANK == 0:
        exact = a["tensors"] == b["tensors"]
        if exact:
            metrics = {"passed": True, "exact": True, "relative_l2": 0.0, "cosine": 1.0}
        else:
            ref = load_hybrid_features(Path(a["file"]), workers=8)
            actual = load_hybrid_features(Path(b["file"]), workers=8)
            metrics = retest.compare(actual, ref)
            del ref, actual
        retest.write(
            decision,
            {
                "tokens": n,
                "repeat": repeat,
                "exact": exact,
                "passed": metrics["passed"],
                "metrics": metrics,
            },
        )
    result = wait_file(decision)
    assert result["passed"], f"CP vs non-CP failed: {decision}"


def capture(payload, n, repeat):
    retest.state("capture", tokens=n, repeat=repeat, cp=RANK == 1)
    values = retest.capture(STORE, ROOT / "captures", payload, n, repeat)
    del values
    receipt = json.loads((ROOT / "captures" / f"n{n}-r{repeat}.json").read_text())
    if repeat:
        first = json.loads((ROOT / "captures" / f"n{n}-r0.json").read_text())
        assert first["tensors"] == receipt["tensors"], (
            "self-repeat is not bitwise exact"
        )
    retest.write(
        ROOT / "CLIENT_RESULT.json",
        {"success": True, "last_tokens": n, "last_repeat": repeat},
    )
    gate(n, repeat)


STORE = None


def main():
    global STORE
    from mooncake.store import MooncakeDistributedStore

    assert RANK in (0, 1) and int(os.environ["KUBEBRAIN_REPLICA_TOTAL"]) == 2
    assert os.environ["NORTHGATE_LONG_CAPTURE"] == "0"
    torch.set_num_threads(8)
    selection = json.loads((INPUTS / "selection.json").read_text())
    payloads = {}
    for entry in selection["inputs"]:
        p = INPUTS / entry["file"]
        assert hashlib.sha256(p.read_bytes()).hexdigest() == entry["file_sha256"]
        row = json.loads(p.read_text())
        n = entry["tokens"]
        assert len(row["input_ids"]) == len(row["loss_mask"]) == n
        payloads[n] = row
    config = {
        "context_length": CAPACITY,
        "max_total_tokens": CAPACITY,
        "exact_test_tokens": 131072,
        "real_complete_tokens": 130999,
        "tp": 8,
        "ep": 8,
        "cp": 8 if RANK else 1,
        "seed": 20261106,
        "model": retest.MODEL,
        "inputs": selection,
        "long_helpers": False,
        "cp_numerics": RANK == 1,
        "wide_offsets_int64": True,
        "codec": "native-brotli10-zstd",
        "failure_policy": "stop both layouts after first failed gate",
    }
    retest.write(ROOT / "config.yaml", config)
    if RANK == 0:
        retest.write(
            RUN / "config.yaml", {**config, "layouts": ["tp8", "cp8"], "total_gpus": 16}
        )
    for name in ["control", "captures", "server"]:
        (ROOT / name).mkdir(exist_ok=True)
    os.environ["NORTHGATE_RESIDENT_DEBUG"] = "1"
    os.environ["NORTHGATE_RESIDENT_CONTROL"] = str(ROOT / "control")
    os.environ.pop("NORTHGATE_CP_NUMERICAL_PARITY", None)
    master_log = (ROOT / "mooncake.log").open("x")
    master = subprocess.Popen(
        [
            "mooncake_master",
            "--enable_http_metadata_server=true",
            "--http_metadata_server_host=127.0.0.1",
            "--rpc_port=35551",
            "--http_metadata_server_port=35880",
            "--metrics_port=35900",
            "--default_kv_lease_ttl=30m",
        ],
        stdout=master_log,
        stderr=subprocess.STDOUT,
        start_new_session=True,
    )
    server = log = None
    try:
        for _ in range(30):
            assert master.poll() is None
            try:
                with socket.create_connection(("127.0.0.1", 35551), timeout=1):
                    break
            except OSError:
                time.sleep(1)
        else:
            raise TimeoutError("Mooncake readiness")
        STORE = MooncakeDistributedStore()
        assert STORE.setup(
            local_hostname="127.0.0.1",
            metadata_server="http://127.0.0.1:35880/metadata",
            global_segment_size=0,
            local_buffer_size=8 << 30,
            protocol="tcp",
            rdma_devices="",
            master_server_addr="127.0.0.1:35551",
        ) in (0, None)
        retest.state("starting-server", cp=RANK == 1)
        server, log = retest.launch(RANK == 1, ROOT / "server", context_length=CAPACITY)
        ranks = probe("001-install-and-budget")
        retest.write(ROOT / "READY.json", {"success": True, "ranks": ranks})
        wait_file(RUN / f"worker-{1 - RANK}/READY.json", 3600)
        capture(payloads[131072], 131072, 0)
        capture(payloads[131072], 131072, 1)
        capture(payloads[130999], 130999, 0)
        assert retest.request("/flush_cache?timeout=10", timeout=30).startswith(
            "Cache flushed."
        )
        probe("002-final-parameters")
        retest.write(
            ROOT / "RESULT.json",
            {
                "passed": True,
                "captures": 3,
                "lengths": [131072, 130999],
                "repeat_exact": True,
            },
        )
        retest.state("complete")
    except BaseException:
        retest.write(
            ROOT / "REJECTED.json", {"passed": False, "error": traceback.format_exc()}
        )
        raise
    finally:
        retest.stop(server)
        if log:
            log.close()
        if STORE is not None and hasattr(STORE, "close"):
            STORE.close()
        retest.stop(master)
        master_log.close()


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, interrupted)
    main()
