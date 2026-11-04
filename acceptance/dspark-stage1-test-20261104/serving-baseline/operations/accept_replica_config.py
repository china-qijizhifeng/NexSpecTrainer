"""Compare a replica's saved, redacted configuration with the accepted first worker."""

import hashlib
import json
import sys
from pathlib import Path


def check(root):
    root = Path(root)
    reference = json.loads(
        (root / "replica-a/resolved-config-acceptance.json").read_text()
    )
    assert reference["passed"]
    receipt = root / "replica-b/server-info.json"
    actual = json.loads(receipt.read_text())
    configuration = {key: actual[key] for key in reference["configuration"]}
    assert configuration == reference["configuration"], "resolved configuration differs"
    first = json.loads((root / "replica-a/environment.json").read_text())
    second = json.loads((root / "replica-b/environment.json").read_text())
    assert first["node"] == "node205" and second["node"] == "node207"
    assert first["gpus"] == second["gpus"] and len(second["gpus"]) == 8
    for field in ("versions", "protobuf_backend", "source_sha256", "model_identity"):
        assert first[field] == second[field], field
    return {
        "passed": True,
        "scope": "Second worker resolved configuration and environment match accepted first worker; no numerical equivalence or performance optimum claim",
        "source_receipt_sha256": hashlib.sha256(receipt.read_bytes()).hexdigest(),
        "reference_receipt_sha256": hashlib.sha256(
            (root / "replica-a/resolved-config-acceptance.json").read_bytes()
        ).hexdigest(),
        "configuration": configuration,
        "environment_matches_first_worker": True,
    }


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    result = check(root)
    (root / "replica-b/resolved-config-acceptance.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps({"worker": "replica-b", "passed": True}))
