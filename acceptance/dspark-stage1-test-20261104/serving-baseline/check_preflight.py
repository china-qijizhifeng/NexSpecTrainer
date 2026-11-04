"""Check immutable-execution guards, credential redaction and deployment boundaries on CPU."""

import ast
import hashlib
import json
import tempfile
from pathlib import Path

import yaml
from execution_contract import verify_execution

ROOT = Path(__file__).resolve().parent


def rejects(function):
    try:
        function()
    except AssertionError:
        return
    raise AssertionError("invalid execution manifest was accepted")


def main():
    with tempfile.TemporaryDirectory() as temp:
        path = Path(temp)
        (path / "source.py").write_text("original\n")
        manifest = {
            "source.py": hashlib.sha256((path / "source.py").read_bytes()).hexdigest()
        }
        (path / "execution-sha256.json").write_text(json.dumps(manifest))
        verify_execution(path)
        (path / "source.py").write_text("modified\n")
        rejects(lambda: verify_execution(path))
        (path / "execution-sha256.json").write_text(json.dumps({"../escape": "bad"}))
        rejects(lambda: verify_execution(path))
        (path / "link.py").symlink_to(path / "source.py")
        manifest = {
            "link.py": hashlib.sha256((path / "source.py").read_bytes()).hexdigest()
        }
        (path / "execution-sha256.json").write_text(json.dumps(manifest))
        rejects(lambda: verify_execution(path))
    tree = ast.parse((ROOT / "worker.py").read_text())
    function = next(
        n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "sanitize"
    )
    namespace = {"KEY": "private-probe-value-do-not-publish-1234567890"}
    exec(
        compile(ast.Module(body=[function], type_ignores=[]), "worker.py", "exec"),
        namespace,
    )
    secret = namespace["KEY"]
    sample = {
        "api_key": "different-secret",
        "nested": ["prefix " + secret, {"detail": secret}],
        "good": "kept",
    }
    result = namespace["sanitize"](sample)
    assert secret not in json.dumps(result) and "different-secret" not in json.dumps(
        result
    )
    assert result["good"] == "kept"
    for worker, node in (("replica-a", "node205"), ("replica-b", "node207")):
        job = yaml.safe_load((ROOT / (worker + "-job.yaml")).read_text())
        assert job["metadata"]["annotations"]["project"] == "WeaverRSI"
        assert job["spec"]["queue"] == "proj-32"
        spec = job["spec"]["tasks"][0]["template"]["spec"]
        assert spec["nodeSelector"] == {
            "kubernetes.io/hostname": node,
            "node_shcheduleable": "true",
        }
        assert spec["restartPolicy"] == "Never" and not spec.get("hostNetwork")
        container = spec["containers"][0]
        assert container["imagePullPolicy"] == "Never"
        assert container["resources"]["requests"]["nvidia.com/gpu"] == "8"
        assert container["securityContext"]["runAsNonRoot"]
        assert not container["securityContext"].get("privileged")
        assert {v["name"] for v in container["volumeMounts"]} == {
            "task",
            "model",
            "shm",
        }
        assert next(v for v in container["volumeMounts"] if v["name"] == "model")[
            "readOnly"
        ]
        assert {v["name"] for v in spec["initContainers"][0]["volumeMounts"]} == {
            "task",
            "archive",
            "runtime",
            "host-ctr",
            "host-crictl",
            "host-snapshotter",
        }
    profile = json.loads((ROOT / "candidate-profile.json").read_text())
    arguments = profile["arguments"]
    assert arguments[arguments.index("--speculative-algorithm") + 1] == "DSPARK"
    assert arguments[arguments.index("--speculative-dspark-block-size") + 1] == "5"
    assert not {
        "--disable-radix-cache",
        "--disable-cuda-graph",
        "--skip-tokenizer-init",
    }.intersection(arguments)
    print(
        json.dumps(
            {
                "passed": True,
                "scope": "CPU execution guards/redaction/YAML/profile only; no GPU/model/API acceptance",
                "checks": [
                    "valid execution",
                    "mutation rejected",
                    "escape rejected",
                    "symlink rejected",
                    "nested secret redaction",
                    "two worker permission/topology profiles",
                    "real DSPARK/cache/tokenizer flags",
                ],
            }
        )
    )


if __name__ == "__main__":
    main()
