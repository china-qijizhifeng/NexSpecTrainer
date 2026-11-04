import json
import pathlib
import urllib.error
import urllib.request


def main(options):
    origin = "http://10.1.2.7:31000"
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))

    def request(path, method="GET", body=None, token=None):
        headers = {"Content-Type": "application/json"}
        if token:
            headers["Authorization"] = "Bearer " + token
        data = None if body is None else json.dumps(body).encode()
        req = urllib.request.Request(
            origin + path, data=data, headers=headers, method=method
        )
        try:
            with opener.open(req, timeout=30) as response:
                result = json.load(response)
        except urllib.error.HTTPError as exc:
            raise RuntimeError(
                "northjob HTTP " + str(exc.code) + " at " + path
            ) from None
        if isinstance(result, dict) and result.get("success") is False:
            raise RuntimeError("northjob rejected request at " + path)
        return result

    creds = json.loads(
        (pathlib.Path.home() / ".config/northjob/userinfo.conf").read_text()
    )
    token = request(
        "/api/auth/get_token",
        "POST",
        {
            "username": creds["username"],
            "password": creds["password"],
            "expireTime": 3600,
        },
    )["data"]
    if not isinstance(token, str) or not token:
        raise RuntimeError("northjob returned no token")

    def clean(value):
        if isinstance(value, dict):
            return {
                k: (
                    "[REDACTED]"
                    if any(
                        s in k.lower()
                        for s in ("token", "password", "secret", "authorization")
                    )
                    else clean(v)
                )
                for k, v in value.items()
            }
        if isinstance(value, list):
            return [clean(v) for v in value]
        if isinstance(value, str):
            for secret in (token, creds["password"]):
                if secret:
                    value = value.replace(secret, "[REDACTED]")
        return value

    action = options["action"]
    if action == "partitions":
        print(
            json.dumps(
                clean(request("/api/partitions", token=token)), ensure_ascii=False
            )
        )
        return
    if action == "submit":
        result = request(
            "/api/task/create_job_by_yaml", "POST", {"code": options["yaml"]}, token
        )
        print(json.dumps(clean(result), ensure_ascii=False))
        return
    result = request("/api/get_jobs", token=token)
    rows = result.get("data")
    if not isinstance(rows, list):
        raise RuntimeError("unexpected northjob jobs response shape")
    rows = [
        x
        for x in rows
        if x.get("namespace") == options["namespace"]
        and x.get("job_creator") == options["creator"]
    ]
    if options.get("alias"):
        rows = [x for x in rows if x.get("user_def_name") == options["alias"]]
    if options.get("prefix"):
        rows = [
            x for x in rows if x.get("user_def_name", "").startswith(options["prefix"])
        ]
    if action == "delete":
        if not rows:
            print(json.dumps({"alias": options["alias"], "status": "absent"}))
            return
        if len(rows) != 1:
            raise RuntimeError(
                "refusing deletion: expected one exact owned alias, found "
                + str(len(rows))
            )
        row = rows[0]
        deleted = request(
            "/api/delete_jobs",
            "DELETE",
            {"name": row["name"], "namespace": options["namespace"]},
            token,
        )
        print(
            json.dumps(
                clean(
                    {
                        "name": row["name"],
                        "alias": row["user_def_name"],
                        "response": deleted,
                    }
                ),
                ensure_ascii=False,
            )
        )
        return
    keys = (
        "name",
        "user_def_name",
        "status",
        "ready_status",
        "namespace",
        "job_creator",
        "user_gpu",
    )
    print(
        json.dumps(
            clean([{k: x.get(k) for k in keys} for x in rows]),
            ensure_ascii=False,
            indent=2,
        )
    )
