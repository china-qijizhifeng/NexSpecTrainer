import hashlib
import json
import os
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TARGET = Path(
    "/gpfs/users/linlizhi/models/huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017"
)
TRAIN = Path("/gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6")


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


def public_stream(path, mode):
    """Control receipts must be readable by the separate SSH controller UID."""
    assert mode in ("w", "x")
    flags = os.O_WRONLY | os.O_CREAT | (os.O_EXCL if mode == "x" else os.O_TRUNC)
    descriptor = os.open(path, flags, 0o644)
    try:
        os.fchmod(descriptor, 0o644)
        return os.fdopen(descriptor, "w")
    except BaseException:
        os.close(descriptor)
        raise


def write(path, value):
    path = Path(path)
    tmp = path.with_name(path.name + ".partial")
    with public_stream(tmp, "w") as f:
        json.dump(value, f, indent=2)
        f.write("\n")
        f.flush()
        os.fsync(f.fileno())
    tmp.replace(path)


def phase(stage, **kw):
    write(
        ROOT / "phase.json",
        {
            "stage": stage,
            "time_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            **kw,
        },
    )


def verify_execution():
    for name, digest in json.loads(
        (ROOT / "execution-sha256.json").read_text()
    ).items():
        assert sha(ROOT / name) == digest, name


def checkpoint_metadata(checkpoint, step):
    checkpoint = Path(checkpoint)
    marker = json.loads((checkpoint / "training_state.json").read_text())
    assert marker["format"] == "specforge-nfc-fsdp-v1" and marker["global_step"] == step
    files = ("training_state.json", "nfc/common.pt")
    hashes = {name: sha(checkpoint / name) for name in files}
    assert hashes["nfc/common.pt"] == marker["common_sha256"], "NFC common SHA mismatch"
    assert (
        len(list((checkpoint / "nfc").glob("*.distcp"))) == 32
    ), "incomplete tensor shards"
    return hashes


def verify_export(root, step, read_payload):
    root = Path(root)
    receipt = json.loads((root / f"export-step{step}-verification.json").read_text())
    assert receipt["passed"] and receipt["step"] == step
    for name, entry in receipt["overlays"].items():
        path = root / f"export-step{step}" / name
        assert path.stat().st_size == entry["bytes"]
        if read_payload:
            assert sha(path) == entry["sha256"]
    return receipt
