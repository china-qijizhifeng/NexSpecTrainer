"""Validate real r4 receipts and reject changed inputs before GPU allocation.

Model inventory reads are replaced by the recorded r4 identity because this CPU
host does not mount the model. All source, prompt, nsp and calibrator files are
real. The GPU harness independently reads the mounted model inventory.
"""

import argparse
import copy
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
from unittest.mock import patch

import baseline_reuse as reuse


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    current = Path(__file__).resolve().parents[2]
    baseline = args.baseline_dir.resolve()
    status = reuse.read(baseline / "target/measured-show.stdout")
    identity = status["completion"]["stages"]["serving"]["checkpoint_identity"]
    model = Path(identity["path"])
    hashes = {model / row["name"]: row["sha256"] for row in identity["files"] if "sha256" in row}
    real_sha = reuse.sha
    def cpu_sha(path):
        return hashes[Path(path)] if Path(path) in hashes else real_sha(path)

    outcomes = []
    with tempfile.TemporaryDirectory(prefix="hybrid-r4-reuse-") as temporary:
        root = Path(temporary)
        source = root / "source"
        subprocess.run(["git", "clone", "--shared", "--no-checkout", str(current), str(source)],
                       check=True, capture_output=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", reuse.BASELINE_COMMIT],
                       check=True, capture_output=True)
        bundle = current / reuse.BUNDLE

        def exercise(name, *, prompt_change=False, identity_change=False, calibrator_change=False,
                     helper_change=False):
            output = root / name
            output.mkdir()
            for filename in ("calibration-prompts.jsonl", "measured-prompts.jsonl", "inputs.json"):
                shutil.copy2(baseline / filename, output / filename)
            if prompt_change:
                with (output / "measured-prompts.jsonl").open("a") as stream:
                    stream.write("\n")
            candidate_identity = copy.deepcopy(identity)
            if identity_change:
                candidate_identity["files"][0]["size"] += 1
            candidate = baseline
            if calibrator_change:
                candidate = root / "changed-baseline"
                shutil.copytree(baseline, candidate)
                artifact, = list(candidate.rglob("ngram-calibrator.json"))
                artifact.write_text(artifact.read_text() + "\n")
            helper = source / "specforge/hybrid/sglang_runtime.py"
            original = helper.read_bytes()
            if helper_change:
                helper.write_bytes(original + b"\n# Changed baseline source\n")
            try:
                with patch.object(reuse, "serving_checkpoint_identity", return_value=candidate_identity), patch.object(reuse, "sha", side_effect=cpu_sha):
                    outputs, _, calibrator = reuse.reuse_baseline(candidate, source, output,
                        current, bundle, model)
                assert name == "valid", name + " should have been rejected"
                assert set(outputs) == {"target", "dspark"}
                assert all(len(rows) == 17 for rows in outputs.values())
                assert reuse.sha(calibrator) == "0f481ca190c4312613fc191ffc94af292407b2f6c3e69f0a4a09bc8b2b8a7d09"
                assert reuse.read(output / "baseline-reuse.json")["calibration_provenance"]["runtime"] == reuse.BASELINE_MANIFEST
                outcomes.append(dict(case=name, passed=True, requests_per_mode=17))
            except ValueError as error:
                assert name != "valid", str(error)
                outcomes.append(dict(case=name, passed=True, rejected=str(error)))
            finally:
                helper.write_bytes(original)

        exercise("valid")
        exercise("changed-prompt", prompt_change=True)
        exercise("changed-checkpoint", identity_change=True)
        exercise("changed-calibrator", calibrator_change=True)
        exercise("changed-source", helper_change=True)
    receipt = dict(passed=True, scope=__doc__, cases=outcomes)
    args.output.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
