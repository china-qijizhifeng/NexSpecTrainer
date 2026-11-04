"""Add only the verified serving image to the assigned training worker's containerd cache."""

import hashlib
import json
import os
import shutil
import subprocess
import time
from pathlib import Path

from ctr_contract import overlayfs_roots
from execution_contract import verify_execution

root = Path("/gpfs/users/linlizhi/nexspec-v41-capture-20261103")
out = Path(__file__).resolve().parent / "image-import"
out.mkdir(exist_ok=True)
ctr = [
    "/host-ctr",
    "--address",
    "/run/nexspec-runtime/containerd.sock",
    "--namespace",
    "k8s.io",
]
cri = [
    "/host-crictl",
    "--runtime-endpoint",
    "unix:///run/nexspec-runtime/containerd.sock",
    "--image-endpoint",
    "unix:///run/nexspec-runtime/containerd.sock",
]
source = "docker.io/lmsysorg/sglang@sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa"
expected_config = (
    "sha256:31d58781877bdcd322d5fcc7a858d427db54f5cbf5a9e6342c5fafc01bc67c6e"
)


def run(command, timeout=60):
    return subprocess.check_output(command, text=True, timeout=timeout)


try:
    verify_execution(Path(__file__).resolve().parent)
    import os

    assert (
        os.environ["NODE_NAME"] == os.environ["EXPECTED_NODE"]
        and os.environ["NODE_NAME"] == "node090"
    )
    print(
        json.dumps(
            {
                "node": os.environ["NODE_NAME"],
                "scope": "import only this verified image, no pruning",
            }
        ),
        flush=True,
    )
    gpu_text = run(
        ["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader"]
    )
    gpu_rows = [line.strip() for line in gpu_text.splitlines() if line.strip()]
    assert len(gpu_rows) == 8 and all("H100" in line for line in gpu_rows), gpu_rows
    (out / "gpu-preflight.json").write_text(
        json.dumps(
            {"node": os.environ["NODE_NAME"], "gpus": gpu_rows, "passed": True},
            indent=2,
        )
        + "\n"
    )
    receipt = json.loads((root / "offline-image/source-receipt.json").read_text())
    complete = json.loads((root / "offline-image/complete.json").read_text())
    assert complete["verified"] and complete["image_config_digest"] == expected_config
    assert receipt["manifest_digest"] == source.rsplit("@", 1)[1]
    assert receipt["image_config_digest"] == expected_config
    assert complete["sha256"] == receipt["archive_sha256"]
    archive = root / "offline-image" / complete["archive"]
    assert archive.stat().st_size == receipt["archive_bytes"] == complete["bytes"]
    digest = hashlib.sha256()
    with archive.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    assert digest.hexdigest() == receipt["archive_sha256"]
    print(
        json.dumps({"phase": "archive_reverified", "bytes": archive.stat().st_size}),
        flush=True,
    )
    plugin_text = run(ctr + ["plugins", "list", "--detailed"])
    (out / "plugins.txt").write_text(plugin_text)
    roots = overlayfs_roots(plugin_text)
    expected_storage = "/data/var/lib/containerd/io.containerd.snapshotter.v1.overlayfs"
    assert roots == [
        expected_storage
    ], "node snapshotter differs from the exact mounted preflight path"
    capacities = {expected_storage: shutil.disk_usage("/host-snapshotter").free}
    assert min(capacities.values()) > receipt["uncompressed_image_size"] * 2 + (
        20 << 30
    ), "node cache space insufficient"
    print(json.dumps({"phase": "import", "storage_free_bytes": capacities}), flush=True)
    started = time.monotonic()
    with (out / "import.log").open("w") as log:
        subprocess.run(
            ctr
            + [
                "images",
                "import",
                "--local",
                "--snapshotter",
                "overlayfs",
                "--digests",
                "--base-name",
                "docker.io/lmsysorg/sglang",
                "--platform",
                "linux/amd64",
                str(archive),
            ],
            stdout=log,
            stderr=subprocess.STDOUT,
            check=True,
            timeout=1500,
        )
    names = set(run(ctr + ["images", "list", "--quiet"]).splitlines())
    if source not in names:
        assert receipt["cache_tag"] in names, "expected imported reference is absent"
        run(ctr + ["images", "tag", receipt["cache_tag"], source])
    manifest_blob = subprocess.check_output(
        ctr + ["content", "get", receipt["manifest_digest"]], timeout=30
    )
    assert (
        "sha256:" + hashlib.sha256(manifest_blob).hexdigest()
        == receipt["manifest_digest"]
    )
    assert json.loads(manifest_blob)["config"]["digest"] == expected_config
    config_blob = subprocess.check_output(
        ctr + ["content", "get", expected_config], timeout=30
    )
    assert "sha256:" + hashlib.sha256(config_blob).hexdigest() == expected_config
    assert json.loads(config_blob)["rootfs"]["diff_ids"] == receipt["rootfs_diff_ids"]
    # Register only this task's newly imported references with the CRI image manager.
    for reference in [source, receipt["cache_tag"]]:
        if reference in set(run(ctr + ["images", "list", "--quiet"]).splitlines()):
            run(ctr + ["images", "label", reference, "io.cri-containerd.image=managed"])
    for attempt in range(20):
        check = subprocess.run(
            cri + ["inspecti", source], capture_output=True, text=True, timeout=15
        )
        if check.returncode == 0:
            break
        if attempt == 19:
            raise RuntimeError("CRI cannot inspect the imported exact image")
        time.sleep(1)
    status = json.loads(check.stdout)["status"]
    assert status["id"] == expected_config, (status["id"], expected_config)
    assert source in status["repoDigests"], status["repoDigests"]
    result = {
        "passed": True,
        "node": os.environ["NODE_NAME"],
        "gpus": gpu_rows,
        "scope": "verified OCI archive imported into the assigned training worker containerd/CRI cache; no serving or model forward",
        "source": source,
        "manifest_digest": receipt["manifest_digest"],
        "image_config_digest": expected_config,
        "archive_sha256": receipt["archive_sha256"],
        "cache_tag": receipt["cache_tag"],
        "storage_free_bytes_before": capacities,
        "import_seconds": time.monotonic() - started,
        "cri_image_id": status["id"],
        "repo_digests": status["repoDigests"],
    }
    (out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    (out / "exit.txt").write_text("0\n")
    print(json.dumps(result), flush=True)
except BaseException:
    (out / "exit.txt").write_text("1\n")
    raise
