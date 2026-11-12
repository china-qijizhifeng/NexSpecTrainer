"""Import-light Hybrid Spark configuration shared by SGLang argument resolution."""

from dataclasses import asdict, dataclass
import hashlib
import json
import math
import os
from pathlib import Path


@dataclass(frozen=True)
class HybridConfig:
    schema_version: int = 1
    model_horizon: int = 5
    min_match: int = 5
    max_match: int = 15
    max_draft: int = 15
    mode: str = "hybrid"
    contains: bool = True
    slot_price: float = 0.0
    extra_budget_per_request: int | None = None
    calibrator: str | None = None
    # An explicit smoke-test setting, never represented as fitted calibration.
    uncalibrated_acceptance: float | None = None

    def __post_init__(self):
        if (self.schema_version, self.model_horizon, self.min_match, self.max_match, self.max_draft) != (1, 5, 5, 15, 15):
            raise ValueError("native V4.1 Hybrid Spark v1 requires MTP=5, match=5..15, draft<=15")
        if self.mode not in ("hybrid", "mtp", "ngram"):
            raise ValueError("invalid Hybrid Spark mode")
        if not math.isfinite(self.slot_price) or not 0 <= self.slot_price <= 1:
            raise ValueError("slot_price must be finite in [0,1]")
        if self.extra_budget_per_request is not None and (
            type(self.extra_budget_per_request) is not int or not 0 <= self.extra_budget_per_request <= self.max_draft
        ):
            raise ValueError("extra_budget_per_request must be an integer in [0,15]")
        if bool(self.calibrator) == (self.uncalibrated_acceptance is not None):
            raise ValueError("provide a fitted calibrator OR explicit uncalibrated_acceptance for smoke tests")
        if self.uncalibrated_acceptance is not None and (
            not math.isfinite(self.uncalibrated_acceptance) or not 0 < self.uncalibrated_acceptance < 1
        ):
            raise ValueError("uncalibrated_acceptance must be in (0,1)")

    @property
    def verify_width(self):
        return self.max_draft + 1

    def coefficients(self):
        if self.calibrator:
            raw = json.loads(Path(self.calibrator).read_text())
            if raw.get("schema_version") != 1 or raw.get("kind") != "hybrid-ngram-logistic":
                raise ValueError("unsupported Hybrid Spark calibrator")
            if raw.get("rules") != {"min_match": 5, "max_match": 15, "max_draft": 15, "copy_cap": 256}:
                raise ValueError("calibrator n-gram rules differ from runtime")
            coefficients = raw["coefficients"]
        else:
            p = self.uncalibrated_acceptance
            coefficients = {"intercepts": [math.log(p / (1 - p))] * self.max_draft,
                            "match": [0.0, 0.0], "agreement": 0.0,
                            "full": [0.0, 0.0], "copy": [0.0, 0.0]}
        sizes = {"intercepts": 15, "match": 2, "full": 2, "copy": 2}
        for key, size in sizes.items():
            if len(coefficients[key]) != size or not all(math.isfinite(v) for v in coefficients[key]):
                raise ValueError(f"invalid calibrator coefficients: {key}")
        if not math.isfinite(coefficients["agreement"]):
            raise ValueError("invalid calibrator agreement coefficient")
        return coefficients

    def fingerprint(self):
        value = asdict(self)
        if self.calibrator:
            value["calibrator_sha256"] = hashlib.sha256(Path(self.calibrator).read_bytes()).hexdigest()
        return hashlib.sha256(json.dumps(value, sort_keys=True).encode()).hexdigest()


def load_config(path=None):
    path = path or os.environ.get("SGLANG_DSPARK_HYBRID_CONFIG")
    if not path:
        return None
    path = Path(path).resolve()
    raw = json.loads(path.read_text())
    if raw.get("calibrator"):
        raw["calibrator"] = str((path.parent / raw["calibrator"]).resolve())
    config = HybridConfig(**raw)
    config.coefficients()
    return config
