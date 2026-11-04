"""Record metadata credential handling and request-counter availability without copying secrets."""

import json
import sys
import time
import urllib.request
from pathlib import Path


def inspect(root):
    root = Path(root)
    key = (root / "private/worker-key").read_text().strip()
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    workers = {}
    for worker in ("replica-a", "replica-b"):
        service = json.loads((root / worker / "service-identity.json").read_text())
        port = service["ports"][0]["node_port"]
        origin = "http://10.51.6.31:" + str(port)
        responses = {}
        for path in ("/get_server_info", "/metrics"):
            request = urllib.request.Request(
                origin + path, headers={"Authorization": "Bearer " + key}
            )
            with opener.open(request, timeout=30) as response:
                responses[path] = response.read()
        counters = [
            line
            for line in responses["/metrics"].decode().splitlines()
            if line.startswith("sglang:num_requests_total{")
        ]
        assert counters
        assert key not in "\n".join(counters)
        workers[worker] = {
            "endpoint": origin,
            "authenticated_server_info_echoes_known_caller_key": key.encode()
            in responses["/get_server_info"],
            "request_counters": counters,
        }
    return {
        "time": time.time(),
        "scope": "Read-only authenticated metadata inspection; no inference or router runtime validation",
        "workers": workers,
        "raw_metadata_saved": False,
        "boundary": "Saved server-info receipts require redaction. Router metadata forwards caller Authorization; missing/invalid-key rejection and control-plane isolation still need runtime acceptance. Do not inject worker credentials into public metadata requests.",
    }


if __name__ == "__main__":
    root = Path(sys.argv[1]).resolve()
    result = inspect(root)
    (root / "metadata-boundary-audit.json").write_text(
        json.dumps(result, indent=2) + "\n"
    )
    print(json.dumps(result))
