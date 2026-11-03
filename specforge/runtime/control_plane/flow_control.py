# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""Hysteretic producer flow control for the canonical streaming runtime."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass(frozen=True)
class FlowControlLimits:
    """Reference/byte watermarks and the per-worker capture lease cap."""

    high_watermark_refs: int = 256
    low_watermark_refs: Optional[int] = None
    high_watermark_bytes: Optional[int] = None
    low_watermark_bytes: Optional[int] = None
    max_prompt_lease_per_worker: int = 8
    capture_max_sample_bytes: Optional[int] = None

    def __post_init__(self) -> None:
        low_refs = (
            self.high_watermark_refs
            if self.low_watermark_refs is None
            else self.low_watermark_refs
        )
        low_bytes = (
            self.high_watermark_bytes
            if self.low_watermark_bytes is None
            else self.low_watermark_bytes
        )
        if self.high_watermark_refs < 1:
            raise ValueError("high_watermark_refs must be >= 1")
        if low_refs < 0 or low_refs > self.high_watermark_refs:
            raise ValueError(
                "low_watermark_refs must be between 0 and high_watermark_refs"
            )
        if self.high_watermark_bytes is None and self.low_watermark_bytes is not None:
            raise ValueError("low_watermark_bytes requires high_watermark_bytes")
        if self.high_watermark_bytes is not None:
            if self.high_watermark_bytes < 1:
                raise ValueError("high_watermark_bytes must be >= 1")
            if low_bytes is None or not 0 <= low_bytes <= self.high_watermark_bytes:
                raise ValueError(
                    "low_watermark_bytes must be between 0 and " "high_watermark_bytes"
                )
        if self.max_prompt_lease_per_worker < 1:
            raise ValueError("max_prompt_lease_per_worker must be >= 1")
        if self.capture_max_sample_bytes is not None:
            if self.capture_max_sample_bytes < 1:
                raise ValueError("capture_max_sample_bytes must be >= 1")
            if self.high_watermark_bytes is None:
                raise ValueError(
                    "capture_max_sample_bytes requires a byte high watermark"
                )

    def validate_capture_window(self, quantum: int) -> None:
        """A bounded buffer must hold a complete dispatch before any ack."""
        if self.capture_max_sample_bytes is None:
            return
        required = quantum * self.capture_max_sample_bytes
        for level, capacity in (
            ("high", self.high_watermark_bytes),
            ("low", self.resolved_low_watermark_bytes),
        ):
            if capacity < required:
                raise ValueError(
                    f"capture buffer {level} byte watermark {capacity} is smaller "
                    f"than dispatch quantum {quantum} * capture_max_sample_bytes "
                    f"{self.capture_max_sample_bytes} = {required}"
                )

    @property
    def resolved_low_watermark_refs(self) -> int:
        return (
            self.high_watermark_refs
            if self.low_watermark_refs is None
            else self.low_watermark_refs
        )

    @property
    def resolved_low_watermark_bytes(self) -> Optional[int]:
        if self.high_watermark_bytes is None:
            return None
        return (
            self.high_watermark_bytes
            if self.low_watermark_bytes is None
            else self.low_watermark_bytes
        )


class ProducerFlowControl:
    """Thread-safe pause/resume policy shared by all rollout workers."""

    def __init__(self, limits: FlowControlLimits) -> None:
        self.limits = limits
        self._paused = False
        self._reserved_refs = 0
        self._lock = threading.Lock()
        self._stats = {
            "pause_transitions": 0,
            "resume_transitions": 0,
            "wait_checks": 0,
        }

    def prompt_lease(self, requested: int) -> int:
        """Cap a worker's capture request at the configured in-flight lease."""
        return max(0, min(int(requested), self.limits.max_prompt_lease_per_worker))

    def should_pause(self, *, in_flight_refs: int, resident_bytes: int = 0) -> bool:
        """Apply ref/byte hysteresis and return the current latched state."""
        with self._lock:
            return self._should_pause(in_flight_refs, resident_bytes)

    def _should_pause(self, in_flight_refs: int, resident_bytes: int) -> bool:
        refs = max(0, int(in_flight_refs))
        resident = max(0, int(resident_bytes))
        if self.limits.capture_max_sample_bytes is not None:
            refs += self._reserved_refs
            resident += self._reserved_refs * self.limits.capture_max_sample_bytes
        high_bytes = self.limits.high_watermark_bytes
        low_bytes = self.limits.resolved_low_watermark_bytes
        over_high = refs >= self.limits.high_watermark_refs or (
            high_bytes is not None and resident >= high_bytes
        )
        under_low = refs <= self.limits.resolved_low_watermark_refs and (
            low_bytes is None or resident <= low_bytes
        )
        if not self._paused and over_high:
            self._paused = True
            self._stats["pause_transitions"] += 1
        elif self._paused and under_low:
            self._paused = False
            self._stats["resume_transitions"] += 1
        if self._paused:
            self._stats["wait_checks"] += 1
        return self._paused

    def reserve_prompt_lease(
        self, requested: int, *, in_flight_refs: int, resident_bytes: int
    ) -> int:
        """Reserve worst-case capacity before capture; return zero when full.

        The producer serializes this call with publication and byte-accounting
        updates. Reservations remain live until publication or failure cleanup.
        Without a configured per-sample bound, retain legacy soft watermarks.
        """
        size = self.limits.capture_max_sample_bytes
        if size is None:
            return self.prompt_lease(requested)
        with self._lock:
            if self._should_pause(in_flight_refs, resident_bytes):
                return 0
            available_refs = (
                self.limits.high_watermark_refs - in_flight_refs - self._reserved_refs
            )
            available_bytes = (
                self.limits.high_watermark_bytes
                - resident_bytes
                - self._reserved_refs * size
            )
            count = max(
                0,
                min(
                    self.prompt_lease(requested),
                    available_refs,
                    available_bytes // size,
                ),
            )
            if not count:
                self._paused = True
                self._stats["pause_transitions"] += 1
            self._reserved_refs += count
            return count

    def release_prompt_lease(self, count: int) -> None:
        if self.limits.capture_max_sample_bytes is None:
            return
        with self._lock:
            if not 0 <= count <= self._reserved_refs:
                raise ValueError("capture reservation released more than once")
            self._reserved_refs -= count

    def snapshot(
        self, *, in_flight_refs: int, resident_bytes: int = 0
    ) -> Dict[str, Any]:
        with self._lock:
            return {
                "paused": self._paused,
                **self._stats,
                "in_flight_refs": int(in_flight_refs),
                "resident_bytes": int(resident_bytes),
                "capture_reserved_refs": self._reserved_refs,
                "capture_reserved_bytes": self._reserved_refs
                * (self.limits.capture_max_sample_bytes or 0),
                "capture_max_sample_bytes": self.limits.capture_max_sample_bytes,
                "high_watermark_refs": self.limits.high_watermark_refs,
                "low_watermark_refs": self.limits.resolved_low_watermark_refs,
                "high_watermark_bytes": self.limits.high_watermark_bytes,
                "low_watermark_bytes": self.limits.resolved_low_watermark_bytes,
                "max_prompt_lease_per_worker": (
                    self.limits.max_prompt_lease_per_worker
                ),
            }


__all__ = ["FlowControlLimits", "ProducerFlowControl"]
