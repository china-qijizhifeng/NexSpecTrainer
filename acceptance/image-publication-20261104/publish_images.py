"""Push two existing cpu100 images with ephemeral stdin-supplied Harbor auth."""

import base64
import hashlib
import json
import os
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

REGISTRY = "harbor.local.clusters"
ORIGIN = "https://" + REGISTRY
IMAGES = [
    {
        "repository": "bp/nexspec-trainer",
        "tag": "v41-weaver-torch210-20261103-r1",
        "source": REGISTRY + "/bp/nexspec-trainer:v41-weaver-torch210-20261103-r1",
        "config": "sha256:da11e080cc11b041ecb97342c80da74ff66d48b612faa7d4e5a8ca7c0b093336",
        "scope": "historical runtime; current NFC/source compatibility requires separate acceptance",
    },
    {
        "repository": "bp/nexspec-sglang",
        "tag": "v41-c6da252c6837-20261104",
        "source": "lmsysorg/sglang@sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa",
        "config": "sha256:31d58781877bdcd322d5fcc7a858d427db54f5cbf5a9e6342c5fafc01bc67c6e",
        "scope": "exact existing serving image; no worker restart or frozen manifest change",
    },
]


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, request, fp, code, message, headers, new_url):
        raise RuntimeError("unexpected registry redirect; credentials not forwarded")


OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}), NoRedirect())
ACCEPT = ", ".join(
    [
        "application/vnd.docker.distribution.manifest.v2+json",
        "application/vnd.oci.image.manifest.v1+json",
    ]
)


def request(path, token=None, basic=None):
    headers = {"Accept": ACCEPT}
    if token:
        headers["Authorization"] = "Bearer " + token
    if basic:
        headers["Authorization"] = "Basic " + basic
    with OPENER.open(
        urllib.request.Request(ORIGIN + path, headers=headers), timeout=45
    ) as response:
        return response.read(), dict(response.headers)


def manifest(repository, reference, token, missing_ok=False):
    try:
        raw, headers = request(f"/v2/{repository}/manifests/{reference}", token=token)
    except urllib.error.HTTPError as error:
        if missing_ok and error.code == 404:
            return None
        raise RuntimeError(f"registry manifest HTTP {error.code}") from None
    parsed = json.loads(raw)
    assert parsed["schemaVersion"] == 2 and parsed["config"]["digest"].startswith(
        "sha256:"
    )
    digest = "sha256:" + hashlib.sha256(raw).hexdigest()
    assert headers.get("Docker-Content-Digest", digest) == digest
    return {"raw": raw, "parsed": parsed, "digest": digest}


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def main():
    os.umask(0o077)
    credentials = json.load(sys.stdin)
    assert set(credentials) == {"username", "password"}
    assert all(isinstance(value, str) and value for value in credentials.values())
    basic = base64.b64encode(
        (credentials["username"] + ":" + credentials["password"]).encode()
    ).decode()
    # Registry challenge advertises this token endpoint. A robot/repository account
    # need not have permission to call the unrelated Harbor user-management API.
    tokens = {}
    for image in IMAGES:
        query = urllib.parse.urlencode(
            {
                "service": "harbor-registry",
                "scope": f"repository:{image['repository']}:pull,push",
            }
        )
        data, _ = request("/service/token?" + query, basic=basic)
        token_data = json.loads(data)
        token = token_data.get("token") or token_data["access_token"]
        payload = token.split(".")[1]
        claims = json.loads(
            base64.urlsafe_b64decode(payload + "=" * (-len(payload) % 4))
        )
        grants = [
            g for g in claims.get("access", []) if g.get("name") == image["repository"]
        ]
        assert len(grants) == 1 and {"pull", "push"} <= set(
            grants[0]["actions"]
        ), "account lacks required repository pull/push permission"
        tokens[image["repository"]] = token
    root = Path("/root/nexspec-image-publication-20261104")
    root.mkdir(exist_ok=True)
    write(
        root / "auth-check.json",
        {
            "passed": True,
            "registry": REGISTRY,
            "tls_verified": True,
            "time": time.time(),
        },
    )
    results = []
    with tempfile.TemporaryDirectory(prefix="nexspec-harbor-auth-") as temporary:
        config = Path(temporary) / "config.json"
        config.write_text(json.dumps({"auths": {REGISTRY: {"auth": basic}}}))
        config.chmod(0o600)
        for image in IMAGES:
            destination = f"{REGISTRY}/{image['repository']}:{image['tag']}"
            local = json.loads(
                subprocess.check_output(
                    ["docker", "image", "inspect", image["source"]], text=True
                )
            )[0]
            assert local["Id"] == image["config"], "source image changed"
            token = tokens[image["repository"]]
            before = manifest(image["repository"], image["tag"], token, missing_ok=True)
            if before is not None:
                assert (
                    before["parsed"]["config"]["digest"] == image["config"]
                ), "destination differs; do not overwrite"
            name = image["repository"].rsplit("/", 1)[-1]
            if before is None:
                with (root / f"{name}-push-intent.json").open("x") as stream:
                    json.dump(
                        {
                            "time": time.time(),
                            "destination": destination,
                            "source_config": image["config"],
                            "single_use": True,
                        },
                        stream,
                    )
                    stream.flush()
                    os.fsync(stream.fileno())
                if destination != image["source"]:
                    subprocess.run(
                        ["docker", "image", "tag", image["source"], destination],
                        check=True,
                    )
                with (root / f"{name}-push.log").open("w") as log:
                    result = subprocess.run(
                        ["docker", "--config", temporary, "push", destination],
                        stdout=log,
                        stderr=subprocess.STDOUT,
                        timeout=7200,
                    )
                assert (
                    result.returncode == 0
                ), f"push exited {result.returncode}; inspect redacted status before any retry"
            after = manifest(image["repository"], image["tag"], token)
            assert after["parsed"]["config"]["digest"] == image["config"]
            pinned = manifest(image["repository"], after["digest"], token)
            assert pinned["raw"] == after["raw"]
            config_raw, _ = request(
                f"/v2/{image['repository']}/blobs/{image['config']}", token=token
            )
            assert "sha256:" + hashlib.sha256(config_raw).hexdigest() == image["config"]
            remote_config = json.loads(config_raw)
            assert remote_config["rootfs"]["diff_ids"] == local["RootFS"]["Layers"]
            receipt = {
                "passed": True,
                "registry": REGISTRY,
                "destination": destination,
                "pinned_image": f"{REGISTRY}/{image['repository']}@{after['digest']}",
                "manifest_digest": after["digest"],
                "config_digest": image["config"],
                "rootfs_layers": len(local["RootFS"]["Layers"]),
                "manifest_tag_and_digest_readback": True,
                "config_sha_and_rootfs_verified": True,
                "scope": image["scope"],
                "time": time.time(),
            }
            write(root / f"{name}-receipt.json", receipt)
            results.append(receipt)
            print(json.dumps(receipt), flush=True)
    write(
        root / "result.json",
        {"passed": True, "images": results, "credentials_persisted": False},
    )


if __name__ == "__main__":
    main()
