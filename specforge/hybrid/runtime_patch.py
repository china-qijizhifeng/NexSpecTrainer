"""Apply the version-pinned native V4.1 runtime bundle, with all-file preflight."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def apply_bundle(root, bundle, *, execute=False):
    root, bundle = Path(root).resolve(), Path(bundle).resolve()
    manifest = json.loads((bundle / "manifest.json").read_text())
    patch = bundle / "hybrid-runtime.patch"
    sha = lambda path: hashlib.sha256(path.read_bytes()).hexdigest()
    if sha(patch) != manifest["patch_sha256"]:
        raise ValueError("Hybrid runtime patch digest mismatch")
    states = []
    for name, identity in manifest["files"].items():
        path = root / name
        actual = sha(path)
        if actual == identity["before_sha256"]:
            states.append("before")
        elif actual == identity["after_sha256"]:
            states.append("after")
        else:
            raise ValueError(f"unsupported SGLang source: {name}, sha256={actual}")
    if set(states) == {"after"}:
        return {"state": "already-applied", "patch_sha256": manifest["patch_sha256"]}
    if set(states) != {"before"}:
        raise ValueError("partially applied Hybrid patch; restore the known pinned source first")
    subprocess.run(["git", "apply", "--check", str(patch)], cwd=root, check=True)
    if execute:
        subprocess.run(["git", "apply", str(patch)], cwd=root, check=True)
        for name, identity in manifest["files"].items():
            if sha(root / name) != identity["after_sha256"]:
                raise ValueError(f"runtime patch postcondition failed: {name}")
    return {"state": "applied" if execute else "validated", "base_revision": manifest["base_revision"],
            "patch_sha256": manifest["patch_sha256"], "files": len(states)}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, type=Path)
    parser.add_argument("--bundle", required=True, type=Path)
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    print(json.dumps(apply_bundle(args.root, args.bundle, execute=args.execute)))


if __name__ == "__main__":
    main()
