"""Versioned Muon experiment contract, independent of accelerator imports."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class MuonConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)

    version: Literal["muon_adamw.v1"] = "muon_adamw.v1"
    profile: Literal["generic", "v41_flash_muon"] = "generic"
    momentum: float = Field(default=0.95, ge=0, lt=1)
    nesterov: bool = True
    momentum_convention: Literal["ema", "sum"] = "ema"
    ns_steps: int = Field(default=5, gt=0, strict=True)
    ns_coefficients: (
        tuple[float, float, float] | tuple[tuple[float, float, float], ...]
    ) = (3.4445, -4.7750, 2.0315)
    ns_dtype: Literal["bfloat16", "float32"] = "bfloat16"
    eps: float = Field(default=1e-7, gt=0)
    scaling: Literal["original", "match_rms_adamw", "rms", "v41_rms"] = (
        "match_rms_adamw"
    )
    rms: float = Field(default=0.18, gt=0)
    muon_lr_multiplier: float = Field(default=1.0, gt=0)
    adamw_lr_multiplier: float = Field(default=1.0, gt=0)
    muon_weight_decay: float = Field(default=0.1, ge=0)
    adamw_weight_decay: float = Field(default=0.0, ge=0)
    adamw_betas: tuple[float, float] = (0.9, 0.999)
    adamw_eps: float = Field(default=1e-8, gt=0)
    # Bound a single matrix before allocation. Experts and Q heads are streamed.
    max_matrix_elements: int = Field(default=128 * 1024 * 1024, gt=0, strict=True)

    @model_validator(mode="before")
    @classmethod
    def resolve_profile(cls, value):
        if isinstance(value, dict) and value.get("profile") == "v41_flash_muon":
            value = dict(value)
            for key, expected in V41_FLASH_DEFAULTS.items():
                value.setdefault(key, expected)
        return value

    @model_validator(mode="after")
    def validate_options(self):
        if any(not 0 <= beta < 1 for beta in self.adamw_betas):
            raise ValueError("adamw_betas must be in [0, 1)")
        if not self.ns_coefficients:
            raise ValueError("NS coefficients must not be empty")
        if (
            isinstance(self.ns_coefficients[0], tuple)
            and len(self.ns_coefficients) != self.ns_steps
        ):
            raise ValueError("NS coefficient schedule must have ns_steps triples")
        if self.profile == "v41_flash_muon":
            for key, expected in V41_FLASH_DEFAULTS.items():
                if getattr(self, key) != expected:
                    raise ValueError(
                        f"v41_flash_muon requires {key}={expected!r}; use generic for ablations"
                    )
        return self


# V4.1 §2.5/§4.2.2 inherits V4 §2.4 / Algorithm 1. Precision and epsilon
# of NS are implementation choices; unlike these published algorithm values.
V41_FLASH_DEFAULTS = {
    "momentum": 0.95,
    "nesterov": True,
    "momentum_convention": "sum",
    "ns_steps": 10,
    "ns_coefficients": ((3.4445, -4.7750, 2.0315),) * 8 + ((2.0, -1.5, 0.5),) * 2,
    "scaling": "v41_rms",
    "rms": 0.18,
    "muon_weight_decay": 0.1,
    "adamw_weight_decay": 0.1,
    "adamw_betas": (0.9, 0.95),
    "adamw_eps": 1e-20,
}
