"""Fit the paper's small conditional n-gram estimator from exact T=0 traces."""

import hashlib
import json
from pathlib import Path

import torch
import torch.nn.functional as F

from .ngram import scan_reference


def replay_rows(traces, *, stride=1):
    """Only label positions through the first mismatch; later histories differ."""
    for sample, trace in enumerate(traces):
        prompt, output = trace["input_ids"], trace["output_ids"]
        if not prompt or not output or any(type(v) is not int or v < 0 for v in prompt + output):
            raise ValueError("calibration traces require nonempty integer input_ids/output_ids")
        for emitted in range(1, len(output), stride):
            visible = prompt + output[:emitted]
            result = scan_reference(torch.tensor([visible], dtype=torch.int32),
                                    torch.tensor([len(visible)]), torch.tensor([0]))
            length = min(int(result.lengths[0]), len(output) - emitted)
            n = int(result.match_len[0])
            if not length:
                continue
            full = float(n == 15)
            copy = full * __import__("math").log(max(15, int(result.copy_len[0])) / 15)
            for k in range(length):
                first = k == 0
                features = [0.0] * 22
                features[k] = 1.0
                features[15 + (0 if first else 1)] = n - 5
                features[17] = int(result.agreement[0])
                features[18 + (0 if first else 1)] = full
                features[20 + (0 if first else 1)] = copy
                label = float(int(result.tokens[0, k]) == output[emitted + k])
                yield sample, features, label
                if not label:
                    break


def fit_calibrator(traces, *, stride=1, ridge=1e-3, validation_fraction=0.2, seed=42,
                   max_iter=100, provenance=None):
    if len(traces) < 2:
        raise ValueError("calibration requires at least two independent traces for held-out validation")
    shuffled = torch.randperm(len(traces), generator=torch.Generator().manual_seed(seed)).tolist()
    validation = set(shuffled[:max(1, min(len(traces) - 1, round(len(traces) * validation_fraction)))])
    train_x, train_y, valid_x, valid_y = [], [], [], []
    for sample, features, label in replay_rows(traces, stride=stride):
        x, y = (valid_x, valid_y) if sample in validation else (train_x, train_y)
        x.append(features)
        y.append(label)
    if not train_y or not valid_y:
        raise ValueError("both trace splits need n-gram matches and observed conditional labels")
    x, y = torch.tensor(train_x, dtype=torch.float64), torch.tensor(train_y, dtype=torch.float64)
    vx, vy = torch.tensor(valid_x, dtype=torch.float64), torch.tensor(valid_y, dtype=torch.float64)
    weights = torch.zeros(22, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS([weights], max_iter=max_iter, line_search_fn="strong_wolfe")
    def closure():
        optimizer.zero_grad()
        loss = F.binary_cross_entropy_with_logits(x @ weights, y) + ridge * weights.square().sum() / 2
        loss.backward()
        return loss
    optimizer.step(closure)
    w = weights.detach()
    def metrics(inputs, labels):
        predictions = (inputs @ w).sigmoid()
        return {"labels": labels.numel(), "accepted": int(labels.sum()),
                "log_loss": float(F.binary_cross_entropy(predictions, labels)),
                "brier": float((predictions - labels).square().mean())}
    return {"schema_version": 1, "kind": "hybrid-ngram-logistic",
            "rules": {"min_match": 5, "max_match": 15, "max_draft": 15, "copy_cap": 256},
            "coefficients": {"intercepts": w[:15].tolist(), "match": w[15:17].tolist(),
                             "agreement": float(w[17]), "full": w[18:20].tolist(), "copy": w[20:22].tolist()},
            "fit": {"ridge": ridge, "stride": stride, "seed": seed, "max_iter": max_iter,
                    "split_unit": "trace", "training": metrics(x, y), "validation": metrics(vx, vy),
                    "validation_trace_indices": sorted(validation)},
            "provenance": provenance or {}}


def run_calibration(spec, attempt):
    path = Path(spec.traces)
    data = path.read_bytes()
    traces = [json.loads(line) for line in data.splitlines() if line.strip()]
    for row in traces:
        if row.get("phase", "measure") != "measure":
            raise ValueError("warmup traces cannot be calibration data")
    artifact = fit_calibrator(traces, stride=spec.stride, ridge=spec.ridge,
                              validation_fraction=spec.validation_fraction, seed=spec.seed,
                              max_iter=spec.max_iter, provenance={**spec.provenance,
                              "traces_sha256": hashlib.sha256(data).hexdigest(), "temperature": 0})
    if path.read_bytes() != data:
        raise ValueError("calibration traces changed while fitting")
    from specforge.nsp.store import atomic_json

    artifact_path = Path(attempt) / "ngram-calibrator.json"
    atomic_json(artifact_path, artifact)
    result = {"schema_version": 1, "complete": True, "stages": {},
              "calibrator": str(artifact_path), "calibrator_sha256": hashlib.sha256(artifact_path.read_bytes()).hexdigest(),
              "fit": artifact["fit"]}
    atomic_json(Path(attempt) / "results.json", result)
    return result
