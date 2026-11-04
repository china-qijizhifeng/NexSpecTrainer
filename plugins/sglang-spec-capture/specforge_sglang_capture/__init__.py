# coding=utf-8
# Copyright 2024 The SpecForge team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
"""SGLang plugin: SpecForge server-side spec capture into Mooncake.

Installed into the SGLang environment, SGLang's ``load_plugins()`` calls
:func:`register` in every process (entry point group ``sglang.srt.plugins``).
Capture is enabled only when ``SPECFORGE_SPEC_CAPTURE=1``; the server must also
run with ``--enable-return-aux-hidden-states`` (plus ``--speculative-algorithm``
naming the capture convention), ``--return-hidden-states-mode full``
and ``--chunked-prefill-size -1`` (checked when the scheduler starts).
"""

from __future__ import annotations

import os

ENABLE_ENV = "SPECFORGE_SPEC_CAPTURE"

_SCHEDULER = "sglang.srt.managers.scheduler.Scheduler"


def enabled() -> bool:
    return os.environ.get(ENABLE_ENV) == "1"


def register() -> None:
    """Register the scheduler hooks; a no-op unless capture is enabled."""
    if not enabled():
        return
    from sglang.srt.plugins.hook_registry import HookRegistry, HookType

    HookRegistry.register(
        f"{_SCHEDULER}.get_output_streamer_class",
        _capture_streamer_class,
        HookType.AFTER,
    )
    HookRegistry.register(f"{_SCHEDULER}.__init__", _install, HookType.AFTER)
    HookRegistry.register(
        f"{_SCHEDULER}.get_internal_state", _capture_internal_state, HookType.AFTER
    )


def _capture_streamer_class(streamer_class, scheduler, *args, **kwargs):
    from specforge_sglang_capture.capture import capture_streamer_class

    return capture_streamer_class(streamer_class)


def _install(result, scheduler, *args, **kwargs):
    from specforge_sglang_capture.capture import install

    # Raising here fails scheduler startup, unlike errors in register(), which
    # load_plugins() only logs.
    install(scheduler)
    if os.environ.get("NORTHGATE_RESIDENT_DEBUG") == "1":
        from ops.northgate_capture.resident_control import ResidentControl

        scheduler._northgate_debug = ResidentControl(scheduler)
    if os.environ.get("NORTHGATE_CP_DIAGNOSTICS"):
        from ops.northgate_capture import cp_diagnostics

        cp_diagnostics.install()
    if os.environ.get("NORTHGATE_LONG_CAPTURE") == "1":
        from ops.northgate_capture import bounded_moe, mhc_index64
        from benchmarks.v41_prefill import fused_indexer

        bounded_moe.configure(8192)
        mhc_index64.install()
        fused_indexer.configure(True)
    if os.environ.get("NORTHGATE_CP_NUMERICAL_PARITY") == "1":
        from ops.northgate_capture import cp_numerics

        scheduler._northgate_cp_numerics_restore = cp_numerics.install(
            scheduler.tp_worker.model_runner.model
        )
    limit = int(os.environ.get("SPECFORGE_MOE_CHUNK_TOKENS", "0"))
    if limit:
        from specforge_sglang_capture import bounded_moe

        bounded_moe.configure(limit)
        print(f"SpecForge bounded MoE enabled: tokens={limit}", flush=True)
    if os.environ.get("SPECFORGE_MHC_INDEX64") == "1":
        from specforge_sglang_capture.mhc_index64 import install as widen

        widen()
        print("SpecForge MHC int64 row offsets enabled", flush=True)
    if os.environ.get("SPECFORGE_MEMORY_AUDIT") == "1":
        from specforge_sglang_capture.memory_audit import install as audit

        audit()
    return result


def _capture_internal_state(result, scheduler, *args, **kwargs):
    """Advertise the plugin, and the writer's cleanup endpoint, through
    SGLang's /server_info (``server_capture: auto`` detects the former)."""
    debug = getattr(scheduler, "_northgate_debug", None)
    if debug is not None:
        debug.poll()
        result.internal_state['northgate_resident_debug'] = True
    runtime = getattr(scheduler, "_specforge_capture_runtime", None)
    if runtime is not None:
        result.internal_state["spec_capture_backend"] = "plugin"
        from specforge_sglang_capture.sink import wire_encoding

        if wire_encoding() != "raw":
            result.internal_state["spec_capture_wire_encoding"] = wire_encoding()
        if runtime.is_writer:
            control = runtime.sink.control_endpoint()
            if control is not None:
                result.internal_state["spec_capture_control"] = control
    return result
