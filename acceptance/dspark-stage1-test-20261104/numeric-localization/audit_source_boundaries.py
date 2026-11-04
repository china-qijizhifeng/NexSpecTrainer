"""Read the actual pinned source and record call-site contracts without importing it."""

import argparse
import ast
import hashlib
import json
from pathlib import Path

CONTRACTS = (
    ("models/deepseek_v4.py", "DeepseekV4ForCausalLM", "forward", "forward_batch"),
    ("models/deepseek_v4.py", "DeepseekV4Model", "forward", "forward_batch"),
    ("models/deepseek_v4.py", "DeepseekV4CudaDecoderLayer", "forward", "hidden_states"),
    ("models/deepseek_v4.py", "MQALayer", "forward", "x"),
    ("models/deepseek_v2.py", "DeepseekV2MoE", "forward", "leave_output_sum"),
    (
        "layers/attention/dsv4/indexer.py",
        "C4IndexerBackendMixin",
        "forward_c4_indexer",
        "c4_indexer",
    ),
    ("layers/layer_boundary/stage.py", "StageBoundary", "prepare", "forward_batch"),
    ("layers/logits_processor.py", "LogitsProcessor", "forward", "hidden_states"),
    ("layers/sampler.py", "Sampler", "forward", "positions"),
)
EXTRA = (
    "models/deepseek_v4_boundary.py",
    "models/deepseek_v4_mhc.py",
    "layers/layer_boundary/residual/batch.py",
    "state_capturer/indexer_topk.py",
    "plugins/__init__.py",
    "plugins/hook_registry.py",
    "debug_utils/tensor_dump_forward_hook.py",
)


def member(tree, owner, name):
    cls = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == owner
    )
    return next(
        node
        for node in cls.body
        if isinstance(node, ast.FunctionDef) and node.name == name
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--source", type=Path, required=True, help="sglang package root"
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    base = args.source / "srt"
    files = {name for name, *_ in CONTRACTS} | set(EXTRA)
    content = {name: (base / name).read_text() for name in files}
    trees = {name: ast.parse(value) for name, value in content.items()}
    contracts = []
    for name, owner, method, parameter in CONTRACTS:
        node = member(trees[name], owner, method)
        parameters = [
            arg.arg
            for arg in [*node.args.posonlyargs, *node.args.args, *node.args.kwonlyargs]
        ]
        assert parameter in parameters, (name, owner, parameter)
        contracts.append(
            {
                "file": name,
                "class": owner,
                "method": method,
                "line": node.lineno,
                "parameters": parameters,
            }
        )
    model = member(trees["models/deepseek_v4.py"], "DeepseekV4ForCausalLM", "forward")
    assert any(
        isinstance(n, ast.Call) and ast.unparse(n.func) == "self.model.forward"
        for n in ast.walk(model)
    )
    cuda = member(
        trees["models/deepseek_v4.py"], "DeepseekV4CudaDecoderLayer", "forward"
    )
    calls = {ast.unparse(n.func) for n in ast.walk(cuda) if isinstance(n, ast.Call)}
    assert {
        "self.attn_boundary.prepare",
        "self.attn_boundary.finish",
        "self.ffn_boundary.prepare",
        "self.mlp",
        "output.finish",
    } <= calls
    indexer = member(
        trees["state_capturer/indexer_topk.py"], "IndexerTopkCapturer", "__init__"
    )
    assert any(
        isinstance(n, ast.Assert) and ast.unparse(n.test) == "attn_tp_size == 1"
        for n in ast.walk(indexer)
    )
    sampler = member(trees["layers/sampler.py"], "Sampler", "forward")
    assert any(
        isinstance(n, ast.Call) and ast.unparse(n.func) == "torch.argmax"
        for n in ast.walk(sampler)
    )
    for name in ("plugins/__init__.py", "plugins/hook_registry.py"):
        assert any(
            isinstance(n, ast.ExceptHandler) and ast.unparse(n.type) == "Exception"
            for n in ast.walk(trees[name])
            if isinstance(n, ast.ExceptHandler) and n.type
        )
    result = {
        "passed": True,
        "upstream_commit": "38d9e4a3fe4a493b0ae468477a274e83defda81a",
        "source_sha256": {
            name: hashlib.sha256((base / name).read_bytes()).hexdigest()
            for name in sorted(files)
        },
        "contracts": contracts,
        "direct_model_forward_confirmed": True,
        "cuda_deferred_boundary_call_sites_confirmed": True,
        "builtin_indexer_capture_rejects_tp8": True,
        "greedy_torch_argmax_confirmed": True,
        "plugin_and_patch_exceptions_are_caught": True,
        "scope": "Static pinned-source contracts only; adapter installation and actual worker/GPU coverage remain unverified.",
        "gpu_attempt_submitted": False,
        "held_out_requests": 0,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {
                "passed": True,
                "source_files": len(files),
                "methods": len(contracts),
                "gpu_attempt_submitted": False,
            }
        )
    )


if __name__ == "__main__":
    main()
