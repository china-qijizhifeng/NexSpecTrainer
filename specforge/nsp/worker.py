"""Private subprocess entrypoints; public submissions go through nsp."""

import argparse
import signal
import subprocess
import sys
from pathlib import Path

from specforge.nsp.store import atomic_json, checkpoint_entries, now, read_json


def execute(attempt, kind):
    attempt = Path(attempt)
    config = read_json(attempt / "resolved.json")
    if kind == "train":
        # The existing launcher still owns all trainer/capture processes and
        # signal handling. Do not create a second distributed implementation.
        child = subprocess.Popen(
            [
                sys.executable,
                "-m",
                "specforge.cli",
                "train",
                "--config",
                str(attempt / "resolved.json"),
            ]
        )
        # The supervisor signals the whole process group. Stay alive until
        # the canonical launcher's cleanup has drained all worker processes.
        signal.signal(signal.SIGTERM, lambda *_: None)
        signal.signal(signal.SIGINT, lambda *_: None)
        code = child.wait()
        if code:
            return code
        training = read_json(attempt / "training-completion.json")
        checkpoints = checkpoint_entries(config["output_dir"])
        if (
            training["optimizer_step"] <= 0
            or not checkpoints
            or checkpoints[-1]["step"] != training["optimizer_step"]
        ):
            raise ValueError(
                "training did not publish a positive final step with its completed checkpoint"
            )
        result = {
            "kind": "train",
            "finished_at": now(),
            **training,
            "checkpoints": checkpoints,
        }
    else:
        from specforge.nsp.eval_config import EvalSpec

        spec = EvalSpec.model_validate(config)
        result = {"schema_version": 1, "complete": False, "stages": {}}
        try:
            if any(m != "al" for m in spec.metrics):
                from specforge.nsp.offline_eval import run_offline

                result["stages"]["offline"] = run_offline(spec, attempt)
            if "al" in spec.metrics:
                from specforge.nsp.serving_eval import run_serving

                result["stages"]["serving"] = run_serving(spec, attempt)
            result["complete"] = True
        except BaseException as exc:
            result["error"] = f"{type(exc).__name__}: {exc}"
            for stage in ("offline", "serving"):
                partial = attempt / f"{stage}-results.json"
                if partial.exists():
                    result["stages"][stage] = read_json(partial)
            raise
        finally:
            atomic_json(attempt / "results.json", result)
    atomic_json(attempt / "completion.json", result)
    return 0


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("attempt")
    parser.add_argument("kind", choices=["train", "eval"])
    args = parser.parse_args()
    return execute(args.attempt, args.kind)


if __name__ == "__main__":
    raise SystemExit(main())
