"""Freeze only after CPU batch and review; refuse rewriting an existing manifest."""

import json

from common import ROOT, sha, write


def main():
    assert str(ROOT) == "/gpfs/users/linlizhi/dspark-test-20261104-r10"
    assert json.loads((ROOT / "cpu-regression.json").read_text())["passed"]
    assert json.loads((ROOT / "dependency-preparation.json").read_text())["passed"]
    assert not (ROOT / "execution-sha256.json").exists()
    source = json.loads((ROOT / "r7-execution-sha256.json").read_text())
    names = [name for name in source if name.startswith("source/")]
    names += [
        p.name
        for p in ROOT.iterdir()
        if p.is_file()
        and (
            p.suffix in (".py", ".sh", ".yaml")
            or p.name
            in (
                "RUN.md",
                "protocol.json",
                "resource-plan.json",
                "serving-source-pins.json",
                "inputs-manifest.json",
                "cached-image-evidence.json",
                "export-result.json",
                "export-step25-verification.json",
                "export-step118-verification.json",
                "test-inputs.jsonl",
                "validation-inputs.jsonl",
                "r3-eos-reference.json",
                "predecessor-cancellation.json",
                "cpu-regression.json",
                "static-contract-review.json",
            )
        )
    ]
    names += [
        str(p.relative_to(ROOT)) for p in (ROOT / "contract-source").rglob("*.py")
    ]
    files = {name: sha(ROOT / name) for name in sorted(set(names))}
    write(ROOT / "execution-sha256.json", files)
    write(
        ROOT / "source-provenance.json",
        {
            "execution_manifest_entries": len(files),
            "execution_manifest_sha256": sha(ROOT / "execution-sha256.json"),
            "source_attempt": "r7; source/r2 exports reused unchanged",
            "prior_r7_manifest_sha256": "38a080efa6a37603725d8e0f0cbce67dc884ee8cd97b03ecf0950922880ed4e2",
            "source_commit": "e925d251e9eece36339b3bc2a888ad19e15727ef",
            "sglang_commit": "38d9e4a3fe4a493b0ae468477a274e83defda81a",
            "observational_amendment_user_authorized": True,
            "prior_held_out_requests": 36,
            "old_manifests_and_failed_gate_unchanged": True,
            "automation_remains_paused": True,
            "private_runtime_omitted_from_git": [
                "test-inputs.jsonl",
                "validation-inputs.jsonl",
                "r3-eos-reference.json",
            ],
            "upstream_fixtures_sha256": {
                n: files[n] for n in files if n.startswith("contract-source/")
            },
            "public_execution_bytes_match_archive": True,
            "pre_commit_passed": True,
            "performance_amendment_user_authorized": True,
            "deepinfer_reference_commit": "a679bb86ce36e883a98a97733492b686e5b9a08e",
        },
    )
    print(
        json.dumps(
            {
                "passed": True,
                "manifest_entries": len(files),
                "manifest_sha256": sha(ROOT / "execution-sha256.json"),
            }
        )
    )


if __name__ == "__main__":
    main()
