"""Single-use dependency preparation on SSH host; no launch, submit or deletion."""

import json
import os
import shutil

from common import ROOT, sha, write


def copy_dependency(source, destination, digest, mode):
    assert not destination.is_symlink(), "dependency destination may not be a symlink"
    if destination.exists():
        assert (
            destination.is_file() and sha(destination) == digest
        ), "existing dependency differs"
        return False
    assert sha(source) == digest
    shutil.copyfile(source, destination)
    assert sha(destination) == digest
    destination.chmod(mode)
    return True


def main():
    assert str(ROOT) == "/gpfs/users/linlizhi/dspark-test-20261104-r9"
    (ROOT / "image-import").mkdir()
    with (ROOT / "dependency-prepare-intent.json").open("x") as stream:
        json.dump(
            {"single_use": True, "mutation_scope": "new r9 directory only"}, stream
        )
        stream.flush()
        os.fsync(stream.fileno())
    previous = ROOT.parent / "dspark-test-20261104-r7"
    manifest = json.loads((previous / "execution-sha256.json").read_text())
    assert (
        len(manifest) == 860
        and sha(previous / "execution-sha256.json")
        == "38a080efa6a37603725d8e0f0cbce67dc884ee8cd97b03ecf0950922880ed4e2"
    )
    for name, digest in manifest.items():
        assert sha(previous / name) == digest, name
    (ROOT / "source").symlink_to(
        (previous / "source").resolve(), target_is_directory=True
    )
    for step in (25, 118):
        name = f"export-step{step}"
        (ROOT / name).symlink_to((previous / name).resolve(), target_is_directory=True)
    for name in (
        "test-inputs.jsonl",
        "validation-inputs.jsonl",
        "r3-eos-reference.json",
        "export-step25-verification.json",
        "export-step118-verification.json",
    ):
        copy_dependency(
            previous / name,
            ROOT / name,
            manifest[name],
            (
                0o600
                if name
                in (
                    "test-inputs.jsonl",
                    "validation-inputs.jsonl",
                    "r3-eos-reference.json",
                )
                else 0o644
            ),
        )
    finalize_preparation(previous, manifest)


def finalize_preparation(previous, manifest):
    for group in ("target-only", "official-mtp", "step25", "step118"):
        assert not any((previous / group).glob("test-*.json"))
    deleted = json.loads((previous / "eval-delete.json").read_text())
    assert (
        deleted["name"] == "job-6b4dd801c01d-20261104173837"
        and deleted["response"]["success"]
    )
    audit = json.loads((previous / "diagnosis/aa-failure-audit.json").read_text())
    assert audit["audit_passed"] and audit["mismatch_before_eos_confirmed"]
    write(
        ROOT / "predecessor-cancellation.json",
        {
            "r7_job": deleted["name"],
            "r7_delete_success": True,
            "r7_aa_remains_failed": True,
            "prior_held_out_requests": 0,
            "probe_r2_complete_and_cleaned": True,
            "no_old_controller_or_job_restarted": True,
            "source_manifest_sha256": sha(previous / "execution-sha256.json"),
        },
    )
    write(ROOT / "r7-execution-sha256.json", manifest)
    write(
        ROOT / "dependency-preparation.json",
        {
            "passed": True,
            "source_manifest_all860_files_verified": True,
            "private_input_and_reference_bytes_unchanged": True,
            "source_symlink": str((ROOT / "source").resolve()),
            "export_symlinks": {
                str(step): str((ROOT / f"export-step{step}").resolve())
                for step in (25, 118)
            },
            "old_permissions_or_contents_modified": False,
        },
    )
    print(json.dumps({"passed": True, "prior_held_out_requests": 0, "launched": False}))


if __name__ == "__main__":
    main()
