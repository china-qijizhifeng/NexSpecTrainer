"""One-change 128K mHC dispatch ablation against immutable TP fingerprints."""

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
from ops.northgate_capture.cp128k import probe
from specforge.data.feature_file import load_hybrid_features, save_hybrid_features

ROOT = retest.ROOT
RUN = Path(os.environ["NORTHGATE_CAPTURE_RUN_DIR"])
DATA = retest.DATA
PRIOR = DATA / "runs/northgate-cp128k-parity-16h200-20261106-v1"


def tensor_info(values):
    return {
        k: {
            "shape": list(v.shape),
            "sha256": hashlib.sha256(
                v.contiguous().view(torch.uint8).numpy()
            ).hexdigest(),
        }
        for k, v in values.items()
    }


def main():
    from mooncake.store import MooncakeDistributedStore

    torch.set_num_threads(8)
    assert int(os.environ["KUBEBRAIN_REPLICA_TOTAL"]) == 1
    payload = json.loads((DATA / "cp128k-inputs/n131072.json").read_text())
    baseline = json.loads((PRIOR / "worker-0/captures/n131072-r0.json").read_text())
    assert (
        hashlib.sha256(json.dumps(payload["input_ids"]).encode()).hexdigest()
        == baseline["input_sha256"]
    )
    config = {
        "tokens": 131072,
        "context_length": 135168,
        "tp": 8,
        "cp": 8,
        "ep": 8,
        "seed": 20261106,
        "ablation": "only global-length mHC stats dispatch differs from cp128k-v1",
        "baseline": str(PRIOR / "worker-0/captures/n131072-r0.json"),
        "wide_offsets_int64": True,
        "bounded_moe": False,
        "fused_indexer": False,
    }
    for p in [RUN / "config.yaml", ROOT / "config.yaml"]:
        retest.write(p, config)
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
    store = server = log = None
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
        store = MooncakeDistributedStore()
        assert store.setup(
            local_hostname="127.0.0.1",
            metadata_server="http://127.0.0.1:35880/metadata",
            global_segment_size=0,
            local_buffer_size=8 << 30,
            protocol="tcp",
            rdma_devices="",
            master_server_addr="127.0.0.1:35551",
        ) in (0, None)
        retest.state("starting-cp-ablation")
        server, log = retest.launch(True, ROOT / "server", context_length=135168)
        ranks = probe("001-install-and-budget")
        retest.write(ROOT / "READY.json", {"success": True, "ranks": ranks})
        receipts = []
        for repeat in range(2):
            retest.state("capture", tokens=131072, repeat=repeat)
            values, timing = retest.fetch_capture(
                store, ROOT / "captures", payload, 131072, repeat
            )
            tensors = tensor_info(values)
            same = {k: tensors[k] == v for k, v in baseline["tensors"].items()}
            receipt = {
                "tokens": 131072,
                "repeat": repeat,
                "tensors": tensors,
                "baseline_exact": same,
                "timing": timing,
                "finite": True,
            }
            retest.write(ROOT / f"RAW-{repeat}.json", receipt)
            receipts.append(receipt)
            if not all(same.values()):
                torch.save(
                    values["target_last_hidden_states"], ROOT / "failed-final.pt"
                )
                raise AssertionError(f"128K ablation not exact: {same}")
            if repeat == 0:
                del values
        assert receipts[0]["tensors"] == receipts[1]["tensors"]
        assert retest.request("/flush_cache?timeout=10", timeout=30).startswith(
            "Cache flushed."
        )
        probe("002-final-parameters")
        # Release GPU resources/processes before the CPU codec verification.
        retest.stop(server)
        server = None
        log.close()
        log = None
        retest.state("native-codec-roundtrip")
        path = ROOT / "captures/n131072-fixed.ckpt.hybrid"
        started = time.monotonic()
        save_hybrid_features(
            values,
            path,
            quality=10,
            workers=16,
            interleave_layers={"hidden_states": [37, 38, 39]},
        )
        restored = load_hybrid_features(path, workers=8)
        assert all(torch.equal(values[k], restored[k]) for k in values)
        retest.write(
            ROOT / "RESULT.json",
            {
                "passed": True,
                "tokens": 131072,
                "repeats": 2,
                "all_tensors_match_original_tp": True,
                "repeat_exact": True,
                "native_roundtrip_exact": True,
                "codec_seconds": time.monotonic() - started,
                "file": str(path),
                "bytes": path.stat().st_size,
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
        if store is not None and hasattr(store, "close"):
            store.close()
        retest.stop(master)
        master_log.close()


if __name__ == "__main__":

    def interrupted(signum, frame):
        raise SystemExit(128 + signum)

    for sig in [signal.SIGINT, signal.SIGTERM]:
        signal.signal(sig, interrupted)
    main()
