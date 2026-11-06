"""Run-local CP128K setup and allocator checks on every serving rank."""

import hashlib
import json

import torch


def run(scheduler, output, state):
    from sglang.srt.runtime_context import get_parallel

    parallel = get_parallel()
    model = scheduler.tp_worker.model_runner.model
    identity = [(n, p.data_ptr(), p._version) for n, p in model.named_parameters()]
    if "initial_parameters" not in state:
        state["initial_parameters"] = identity
        # Full TP residual offsets exceed int32 at 128K; keep arithmetic unchanged.
        from ops.northgate_capture import mhc_index64

        mhc_index64.install()
        if parallel.attn_cp_size == 8:
            from ops.northgate_capture.cp_numerics import install

            state["restore"] = install(model)
    assert identity == state["initial_parameters"]
    allocator = scheduler.token_to_kv_pool_allocator
    budget = allocator.create_prefill_budget(scheduler.tree_cache)
    fits = {}
    for n in [130999, 131072]:
        fits[n] = budget.check_prefill(
            extend_input_len=n,
            total_tokens=n + scheduler.page_size,
            max_new_tokens=0,
            input_tokens=n,
            swa_host_hit_length=0,
            chunk_limit=None,
        )[0]
    result = {
        "cp_size": parallel.attn_cp_size,
        "idle": scheduler.is_fully_idle(),
        "input_limit": scheduler.max_req_input_len,
        "page_size": scheduler.page_size,
        "remaining_total": budget.remaining_total,
        "remaining_swa": budget.remaining_swa,
        "fits": fits,
        "parity_installed": "restore" in state,
        "wide_offsets_int64": True,
        "parameters_unchanged_since_initial": identity == state["initial_parameters"],
        "parameter_identity_sha256": hashlib.sha256(
            json.dumps(identity).encode()
        ).hexdigest(),
        "cuda_allocated": torch.cuda.memory_allocated(),
        "cuda_reserved": torch.cuda.memory_reserved(),
        "cuda_peak_allocated": torch.cuda.max_memory_allocated(),
        "cuda_free_total": torch.cuda.mem_get_info(),
    }
    assert scheduler.max_req_input_len >= 131072 and all(fits.values()), result
    if parallel.tp_rank == 0:
        (output / "ADMISSION.json").write_text(json.dumps(result, indent=2))
    return result
