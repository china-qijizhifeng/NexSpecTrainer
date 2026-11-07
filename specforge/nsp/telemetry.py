"""Rank-zero structured metrics, independent of external trackers."""

import json
import math
import os
from pathlib import Path

from specforge.nsp.store import now


def emit_metrics(metrics, step):
    destination = os.environ.get("NSP_ATTEMPT_DIR")
    if not destination or int(os.environ.get("RANK", "0")) != 0:
        return
    # Producer and trainer may both be rank zero; only trainer emits step metrics.
    values = {}
    for key, value in metrics.items():
        try:
            values[key] = float(value)
        except (TypeError, ValueError):
            continue
        if not math.isfinite(values[key]):
            values[key] = None
    with (Path(destination) / "metrics.jsonl").open("a") as stream:
        stream.write(
            json.dumps({"at": now(), "step": step, "metrics": values}, allow_nan=False)
            + "\n"
        )


def emit_train_completion(step):
    destination = os.environ.get("NSP_ATTEMPT_DIR")
    if destination and int(os.environ.get("RANK", "0")) == 0:
        from specforge.nsp.store import atomic_json

        atomic_json(
            Path(destination) / "training-completion.json",
            {"optimizer_step": int(step), "at": now()},
        )
