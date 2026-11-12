"""Reproducible CPU calibration of deterministic V4.1 serving trajectories."""

from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
import yaml


class CalibrateSpec(BaseModel):
    model_config = ConfigDict(extra="forbid")
    schema_version: Literal[1] = 1
    traces: str
    temperature: Literal[0] = 0
    provenance: dict
    stride: int = Field(default=1, gt=0)
    ridge: float = Field(default=1e-3, gt=0)
    validation_fraction: float = Field(default=0.2, gt=0, lt=1)
    seed: int = 42
    max_iter: int = Field(default=100, gt=0)

    @model_validator(mode="after")
    def identities(self):
        for key in ("target", "tokenizer", "runtime", "dataset"):
            if not self.provenance.get(key):
                raise ValueError(f"calibration requires pinned {key} provenance")
        return self


def load_calibration(path):
    path = Path(path).resolve()
    raw = yaml.safe_load(path.read_text())
    raw["traces"] = str((path.parent / raw["traces"]).resolve())
    if not Path(raw["traces"]).is_file():
        raise ValueError("calibration trace file does not exist")
    return CalibrateSpec.model_validate(raw)
