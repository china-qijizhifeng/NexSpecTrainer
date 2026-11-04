"""Regression for owner, lifecycle and live dependency cleanup exclusions."""

import json

from cleanup_weaver_checkpoints import OWNER, eligible

row = {
    "principal_id": OWNER,
    "kind": "trainer",
    "purpose": "user",
    "status": "ready",
    "deletion_state": "live",
    "desired_delete": False,
    "refs": [],
    "active_sources": 0,
    "derived_sources": 0,
}
assert eligible(row)
exclusions = {
    "principal_id": "other",
    "kind": "hf_model",
    "purpose": "slot_home",
    "status": "pending",
    "deletion_state": "deleting",
    "desired_delete": True,
    "refs": [{"kind": "binding"}],
    "active_sources": 1,
    "derived_sources": 1,
}
for key, value in exclusions.items():
    assert not eligible({**row, key: value}), key
print(
    json.dumps(
        {"passed": True, "eligibility_positive": 1, "rejections": len(exclusions)}
    )
)
