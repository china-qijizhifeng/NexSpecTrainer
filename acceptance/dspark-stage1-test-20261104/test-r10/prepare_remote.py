"""Prepare one new r10 directory, preserving all prior manifests and permissions."""

import json
import os
import shutil

from common import ROOT, sha, write


def copy_dependency(source, destination, digest, mode):
    assert not destination.is_symlink()
    if destination.exists():
        assert destination.is_file() and sha(destination) == digest
        return False
    assert sha(source) == digest
    shutil.copyfile(source, destination)
    assert sha(destination) == digest
    destination.chmod(mode)
    return True


def main():
    assert str(ROOT) == "/gpfs/users/linlizhi/dspark-test-20261104-r10"
    with (ROOT / "dependency-prepare-intent.json").open("x") as stream:
        json.dump({"single_use": True, "mutation_scope": "new r10 only"}, stream)
        stream.flush()
        os.fsync(stream.fileno())
    (ROOT / "image-import").mkdir()
    previous = ROOT.parent / "dspark-test-20261104-r9"
    pins = json.loads((previous / "execution-sha256.json").read_text())
    assert len(pins) == 872
    assert sha(previous / "execution-sha256.json") == (
        "330b7037d9b0f17b56754c5a72e9f0cacfbc66026089be1dc42327a6cd7d839d"
    )
    for name, digest in pins.items():
        assert sha(previous / name) == digest, name
    cancellation = json.loads((previous / "performance-cancellation.json").read_text())
    assert cancellation["passed"] and cancellation["alias_absent"]
    assert cancellation["job"] == "job-d1f6f211048d-20261104220148"
    assert cancellation["controller_stopped"]
    assert sum(cancellation["test_request_counts"].values()) == 36
    for name in ("source", "export-step25", "export-step118"):
        (ROOT / name).symlink_to((previous / name).resolve(), target_is_directory=True)
    private = ("test-inputs.jsonl", "validation-inputs.jsonl", "r3-eos-reference.json")
    for name in (
        *private,
        "export-step25-verification.json",
        "export-step118-verification.json",
    ):
        copy_dependency(
            previous / name,
            ROOT / name,
            pins[name],
            0o600 if name in private else 0o644,
        )
    r7_manifest = json.loads((previous / "r7-execution-sha256.json").read_text())
    write(ROOT / "r7-execution-sha256.json", r7_manifest)
    write(ROOT / "predecessor-cancellation.json", cancellation)
    write(
        ROOT / "dependency-preparation.json",
        {
            "passed": True,
            "all872_predecessor_files_verified": True,
            "private_inputs_unchanged": True,
            "r9_partial_test_requests": 36,
            "old_permissions_or_contents_modified": False,
            "no_old_job_restarted": True,
        },
    )
    print(json.dumps({"passed": True, "prior_test_requests": 36, "launched": False}))


if __name__ == "__main__":
    main()
