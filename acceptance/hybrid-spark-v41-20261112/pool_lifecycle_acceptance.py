"""Exercise the patched worker pool lifecycle and confidence builder on CPU.

The GPU history-write import is stubbed and raises if used. No kernels or model
weights run here. The actual confidence builder is tested with a stub head;
native GPU execution and trained weight loading remain separate gates.
"""

import argparse
import ast
import hashlib
import json
import os
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import patch

import torch


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sglang-root", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    worker_path = args.sglang_root / "python/sglang/srt/speculative/dspark_components/dspark_worker_v2.py"
    source = ast.parse(worker_path.read_text())
    worker = next(node for node in source.body if isinstance(node, ast.ClassDef) and node.name == "DSparkWorkerV2")
    constructor = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    call = next(node for node in ast.walk(constructor) if isinstance(node, ast.Call)
                and isinstance(node.func, ast.Name) and node.func.id == "HybridRuntime")
    hook = next(node for node in worker.body if isinstance(node, ast.FunctionDef) and node.name == "alloc_memory_pool")

    def unavailable(*_, **__):
        raise AssertionError("GPU writes must not run in the CPU lifecycle gate")

    kernels = ModuleType("specforge.hybrid.kernels")
    kernels.write_history = unavailable
    with patch.dict(sys.modules, {kernels.__name__: kernels}):
        from specforge.hybrid.sglang_runtime import HybridRuntime

    events = []
    config = SimpleNamespace(calibrator=None, coefficients=lambda: {})
    state = SimpleNamespace(device="cpu", model_runner=SimpleNamespace(req_to_token_pool=None),
                            _hosts_draft=True)
    runtime = eval(compile(ast.Expression(call), str(worker_path), "eval"),
                   {"HybridRuntime": HybridRuntime, "load_config": lambda: config,
                    "self": state, "get_model": lambda: SimpleNamespace(model_path="unused")})
    assert runtime.history is runtime.lengths is None
    pool = SimpleNamespace(req_to_token=torch.empty((7, 128), dtype=torch.int32))
    memory_config, allocator = object(), object()

    def draft_alloc(**kwargs):
        assert kwargs == dict(memory_pool_config=memory_config, req_to_token_pool=pool,
                              token_to_kv_pool_allocator=allocator)
        events.append("draft")

    def hybrid_alloc(actual):
        assert actual is pool and events == ["draft"]
        HybridRuntime.alloc_memory_pool(runtime, actual)
        events.append("history")

    runtime.alloc_memory_pool = hybrid_alloc
    state._hybrid = runtime
    state._draft_worker = SimpleNamespace(alloc_memory_pool=draft_alloc)
    namespace = {}
    exec(compile(ast.Module(body=[hook], type_ignores=[]), str(worker_path), "exec"), namespace)
    allocate = namespace["alloc_memory_pool"]
    allocate(state, memory_config, pool, allocator)
    assert events == ["draft", "history"]
    assert runtime.history.shape == pool.req_to_token.shape
    assert runtime.history.dtype == torch.int32
    assert runtime.lengths.shape == (7,) and runtime.lengths.dtype == torch.int32
    assert runtime.lengths.count_nonzero().item() == 0
    assert runtime.history.data_ptr() != pool.req_to_token.data_ptr()
    events.clear()
    state._hybrid = None
    allocate(state, memory_config, pool, allocator)
    assert events == ["draft"]
    events.clear()
    state._hosts_draft = False
    allocate(state, memory_config, pool, allocator)
    assert not events
    model_path = args.sglang_root / "python/sglang/srt/models/deepseek_v4_dspark.py"
    model_tree = ast.parse(model_path.read_text())
    builder = next(node for node in model_tree.body if isinstance(node, ast.FunctionDef)
                   and node.name == "build_dspark_v4_confidence_head")
    builder.returns = None
    for arg in builder.args.kwonlyargs:
        arg.annotation = None
    static, adaptive = object(), object()
    mode = static
    namespace = dict(read_ragged_verify_mode=lambda: mode,
                     RaggedVerifyMode=SimpleNamespace(STATIC=static),
                     DSparkConfidenceHead=lambda **kwargs: kwargs)
    exec(compile(ast.Module(body=[builder], type_ignores=[]), str(model_path), "exec"), namespace)
    build = namespace["build_dspark_v4_confidence_head"]
    config = SimpleNamespace(enable_confidence_head=True, hidden_size=5120)
    with patch.dict(os.environ):
        os.environ.pop("SGLANG_DSPARK_HYBRID_CONFIG", None)
        assert build(config=config, markov_rank=256) is None
        os.environ["SGLANG_DSPARK_HYBRID_CONFIG"] = "/unused/explicit-hybrid-config.json"
        expected = dict(hidden_size=5120, markov_rank=256, with_markov=True, bias=False)
        assert build(config=config, markov_rank=256) == expected
        mode = adaptive
        assert build(config=config, markov_rank=256) == expected
        os.environ.pop("SGLANG_DSPARK_HYBRID_CONFIG")
        assert build(config=config, markov_rank=256) == expected
    result = dict(passed=True, scope=__doc__, constructor_allocation_deferred=True,
                  hook_pool_identity=True, draft_pool_allocation_precedes_history=True,
                  history_shape=list(runtime.history.shape), history_dtype=str(runtime.history.dtype),
                  lengths_initially_zero=True, hybrid_disabled_hook=True, non_host_rank_skipped=True,
                  worker_sha256=hashlib.sha256(worker_path.read_bytes()).hexdigest(),
                  static_hybrid_keeps_trained_confidence_head=True,
                  static_native_skips_confidence_head=True, adaptive_head_unchanged=True,
                  confidence_builder_sha256=hashlib.sha256(model_path.read_bytes()).hexdigest(),
                  model_or_gpu_tested=False)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result))


if __name__ == "__main__":
    main()
