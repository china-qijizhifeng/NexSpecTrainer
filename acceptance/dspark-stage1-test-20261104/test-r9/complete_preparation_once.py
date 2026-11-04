"""Finish an unsubmitted preparation after a duplicate identical metadata receipt."""

import json
import os

from common import ROOT, public_stream, sha, write
from controller import ALIASES, api
from prepare_remote import copy_dependency, finalize_preparation


def main():
    assert str(ROOT) == "/gpfs/users/linlizhi/dspark-test-20261104-r9"
    assert (ROOT / "dependency-prepare-intent.json").exists()
    assert not (ROOT / "dependency-preparation.json").exists()
    for name in (
        "execution-sha256.json",
        "controller-launch-intent.json",
        "controller-started.json",
        "eval-submit-intent.json",
        "worker-started.json",
    ):
        assert not (ROOT / name).exists(), name
    assert api("jobs", alias=ALIASES["eval"]) == []
    with public_stream(ROOT / "dependency-completion-intent.json", "x") as stream:
        json.dump(
            {
                "single_use": True,
                "reason": "initial import found pre-copied identical export verification metadata",
                "no_job_submitted": True,
            },
            stream,
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
    for name in ("source", "export-step25", "export-step118"):
        assert (ROOT / name).is_symlink() and (ROOT / name).resolve() == (
            previous / name
        ).resolve()
    for name in (
        "test-inputs.jsonl",
        "validation-inputs.jsonl",
        "r3-eos-reference.json",
        "export-step25-verification.json",
        "export-step118-verification.json",
    ):
        assert (ROOT / name).is_file() and not (ROOT / name).is_symlink()
        assert (
            copy_dependency(previous / name, ROOT / name, manifest[name], 0o600)
            is False
        )
    finalize_preparation(previous, manifest)
    write(
        ROOT / "dependency-interruption-audit.json",
        {
            "passed": True,
            "initial_error": "exclusive-copy assertion on existing identical export verification metadata",
            "all_five_existing_dependencies_sha_verified": True,
            "original_prepare_intent_preserved": True,
            "no_old_or_new_dependency_bytes_permissions_modified": True,
            "new_completion_intent_single_use": True,
            "gpu_requests": 0,
            "job_submitted": False,
            "same_class_regressions": "copy absent; reuse exact existing; reject different bytes; reject symlink; preserve existing mode/mtime",
        },
    )


if __name__ == "__main__":
    main()
