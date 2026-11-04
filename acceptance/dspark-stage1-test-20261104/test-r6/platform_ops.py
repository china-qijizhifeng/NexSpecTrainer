"""Single-use owned Northjob Services; credentials remain on the SSH host."""

import json
import os
import sys
import time
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
ORIGIN = "http://10.1.2.7:31000"
APP = "dspark-v41-baseline-20261104-r1"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def write(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def connection():
    def request(path, method="GET", body=None, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        req = urllib.request.Request(
            ORIGIN + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
            method=method,
        )
        with OPENER.open(req, timeout=40) as response:
            result = json.load(response)
        assert not isinstance(result, dict) or result.get("success") is not False
        return result

    credentials = json.loads(
        (Path.home() / ".config/northjob/userinfo.conf").read_text()
    )
    token = request(
        "/api/auth/get_token",
        "POST",
        {
            "username": credentials["username"],
            "password": credentials["password"],
            "expireTime": 3600,
        },
    )["data"]
    assert isinstance(token, str) and token

    def authorized(path, method="GET", body=None):
        return request(path, method, body, token)

    return authorized


def service_alias(worker):
    assert worker in ("replica-a", "replica-b", "router")
    return f"dspark-v41-{worker}-20261104-r1"


def own_services(api):
    query = urllib.parse.urlencode(
        {"creator": "linlizhi", "page": 1, "page_size": 100, "isAdmin": "false"}
    )
    result = api("/api/v1/service/list?" + query)
    assert result["code"] == 0
    data = result["data"]
    rows = data["services"]
    assert len(rows) == data["total"] <= 100, "incomplete service list"
    assert all(r["creator"] == "linlizhi" and r["namespace"] == "qiji" for r in rows)
    return result


def main():
    action = sys.argv[1]
    api = connection()
    if action == "service-list":
        result = own_services(api)
        write(ROOT / "service-list-latest.json", result)
        print(json.dumps(result))
        return
    worker = sys.argv[2]
    alias = service_alias(worker)
    out = ROOT / worker
    out.mkdir(exist_ok=True)
    if action == "service-create":
        if worker != "router":
            assert (
                json.loads((out / "phase.json").read_text())["stage"] == "worker_ready"
            )
            identity = json.loads((out / "job.json").read_text())
            jobs = api("/api/get_jobs")["data"]
            matches = [
                r for r in jobs if r.get("user_def_name") == identity["user_def_name"]
            ]
            assert len(matches) == 1
            job = matches[0]
            assert all(
                job[k] == identity[k] for k in ("name", "namespace", "job_creator")
            )
            assert job["status"] == "Running"
        before = own_services(api)
        write(out / "service-list-before.json", before)
        # The frontend uses service_alias for creation and the API allocates its
        # Kubernetes service_name; never infer it from the submitted alias.
        assert not any(
            r["service_alias"] == alias for r in before["data"]["services"]
        ), "owned alias already exists"
        payload = {
            "service_alias": alias,
            "service_type": "NodePort",
            "selector": {"app": APP, "replica": worker},
            "ports": [
                {
                    "name": "http",
                    "protocol": "TCP",
                    "port": 8000,
                    "target_port": 8000,
                    "node_port": None,
                }
            ],
            "remark": "Owned official Flash-1017 DSPARK baseline; seven-day worker lease",
        }
        intent = {"time": time.time(), "payload": payload, "single_use": True}
        with (out / "service-create-intent.json").open("x") as stream:
            json.dump(intent, stream, indent=2)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            result = api("/api/v1/service/create", "POST", payload)
            write(out / "service-create-response.json", result)
            assert result["code"] == 0, "service rejected; do not retry"
        except Exception as exc:
            write(
                out / "service-create-error.json",
                {"type": type(exc).__name__, "time": time.time(), "no_retry": True},
            )
            raise
        after = own_services(api)
        write(out / "service-list-after.json", after)
        matches = [r for r in after["data"]["services"] if r["service_alias"] == alias]
        assert len(matches) == 1, "uncertain service identity; do not retry"
        identity = matches[0]
        assert identity["selector"] == payload["selector"]
        assert identity["service_type"] == "NodePort"
        assert len(identity["ports"]) == 1
        assert 30000 <= identity["ports"][0]["node_port"] <= 32767
        previous = {r["service_name"] for r in before["data"]["services"]}
        current = {r["service_name"] for r in after["data"]["services"]}
        assert previous.issubset(current) and identity["service_name"] not in previous
        write(out / "service-identity.json", identity)
        print(json.dumps(identity))
        return
    assert action == "service-detail"
    identity = json.loads((out / "service-identity.json").read_text())
    assert identity["service_alias"] == alias and identity["namespace"] == "qiji"
    query = urllib.parse.urlencode(
        {"service_name": identity["service_name"], "namespace": "qiji"}
    )
    result = api("/api/v1/service/detail?" + query)
    write(out / "service-detail.json", result)
    print(json.dumps(result))


if __name__ == "__main__":
    main()
