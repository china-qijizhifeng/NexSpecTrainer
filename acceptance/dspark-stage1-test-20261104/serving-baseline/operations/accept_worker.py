"""Bounded synthetic API acceptance, independent of held-out test inputs."""

import concurrent.futures
import hashlib
import json
import math
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MODEL = "deepseek-v4.1-flash-dspark-baseline"
OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))
TOOL = {
    "type": "function",
    "function": {
        "name": "lookup_temperature",
        "description": "Get the current temperature for the requested city.",
        "parameters": {
            "type": "object",
            "properties": {"city": {"type": "string"}},
            "required": ["city"],
            "additionalProperties": False,
        },
    },
}


def write(path, value):
    path.write_text(json.dumps(value, indent=2) + "\n")


def native_audit(response, budget):
    ids, meta = response["output_ids"], response["meta_info"]
    assert 1 <= len(ids) <= budget and all(type(i) is int and i >= 0 for i in ids)
    assert meta["completion_tokens"] == len(ids)
    finish = meta["finish_reason"]
    if 1 in ids:
        assert ids.index(1) == len(ids) - 1, "native tokens after EOS"
        assert finish["type"] == "stop" and finish["matched"] == 1
    else:
        assert (
            len(ids) == budget
            and finish["type"] == "length"
            and finish["length"] == budget
        )
    verify = meta.get("spec_verify_ct", 0)
    proposed = meta.get("spec_num_proposed_drafts", 0)
    correct = meta.get("spec_num_correct_drafts", 0)
    assert type(verify) is int and verify >= 0
    assert proposed == verify * 5 and 0 <= correct <= proposed
    if verify:
        assert math.isclose(meta["spec_accept_length"], len(ids) / verify, rel_tol=1e-6)
        assert math.isclose(meta["spec_accept_rate"], correct / proposed, rel_tol=1e-6)
    else:
        assert len(ids) == 1 and not correct and not proposed
    return {
        "tokens": len(ids),
        "finish": finish,
        "verify": verify,
        "raw_proposed": proposed,
        "raw_correct": correct,
        "cached_tokens": meta.get("cached_tokens", 0),
    }


class Client:
    def __init__(self, url, key, out):
        self.url, self.key, self.out = url.rstrip("/"), key, out
        self.receipts = {}

    def call(self, name, path, body=None, stream=False):
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.key,
        }
        request = urllib.request.Request(
            self.url + path,
            data=None if body is None else json.dumps(body).encode(),
            headers=headers,
        )
        started = time.monotonic()
        with OPENER.open(request, timeout=900) as response:
            if not stream:
                result = json.load(response)
            else:
                events, done, data = [], False, []
                for raw in response:
                    line = raw.decode().rstrip("\r\n")
                    if line.startswith("data:"):
                        data.append(line[5:].lstrip(" "))
                    elif not line and data:
                        payload = "\n".join(data)
                        data = []
                        if payload == "[DONE]":
                            done = True
                            break
                        event = json.loads(payload)
                        assert "error" not in event
                        events.append(
                            {"seconds": time.monotonic() - started, "response": event}
                        )
                assert done and not data and events, "incomplete SSE stream"
                result = {"events": events, "done": done}
        elapsed = time.monotonic() - started
        assert math.isfinite(elapsed) and elapsed > 0
        assert "error" not in result
        serialized = json.dumps(result)
        assert self.key not in serialized, "credential in inference response"
        write(self.out / (name + ".json"), result)
        self.receipts[name] = {
            "seconds": elapsed,
            "response_sha256": hashlib.sha256(serialized.encode()).hexdigest(),
        }
        return result

    def chat(self, name, messages, *, stream=False, thinking=False, **extra):
        body = {
            "model": MODEL,
            "messages": messages,
            "max_tokens": 512,
            "temperature": 0,
            "chat_template_kwargs": {"thinking": thinking},
            **extra,
            "stream": stream,
        }
        result = self.call(name, "/v1/chat/completions", body, stream)
        if stream:
            message = {"role": "assistant", "content": "", "reasoning_content": ""}
            calls, finishes = {}, []
            for item in result["events"]:
                for choice in item["response"].get("choices", []):
                    assert choice["index"] == 0
                    delta = choice.get("delta", {})
                    for field in ("content", "reasoning_content"):
                        message[field] += delta.get(field) or ""
                    for tool in delta.get("tool_calls") or []:
                        call = calls.setdefault(
                            tool["index"],
                            {
                                "type": "function",
                                "function": {"name": "", "arguments": ""},
                            },
                        )
                        if tool.get("id"):
                            assert "id" not in call or call["id"] == tool["id"]
                            call["id"] = tool["id"]
                        for field in ("name", "arguments"):
                            call["function"][field] += (tool.get("function") or {}).get(
                                field
                            ) or ""
                    if choice.get("finish_reason"):
                        finishes.append(choice["finish_reason"])
            assert len(finishes) == 1
            if calls:
                message["tool_calls"] = [calls[i] for i in sorted(calls)]
            finish = finishes[0]
        else:
            assert result["model"] == MODEL and len(result["choices"]) == 1
            choice = result["choices"][0]
            message, finish = choice["message"], choice["finish_reason"]
            assert result["usage"]["completion_tokens"] > 0
        assert finish in ("stop", "length", "tool_calls")
        self.receipts[name]["finish_reason"] = finish
        return message, result


def main():
    os.umask(0o077)
    worker, endpoint = sys.argv[1:]
    assert worker in ("replica-a", "replica-b")
    out = ROOT / worker / "api-acceptance"
    out.mkdir(mode=0o700, exist_ok=True)
    with (out / "intent.json").open("x") as stream:
        json.dump(
            {"time": time.time(), "single_use": True, "endpoint": endpoint}, stream
        )
        stream.flush()
        os.fsync(stream.fileno())
    key = (ROOT / "private/worker-key").read_text().strip()
    client = Client(endpoint, key, out)
    try:
        for path in ("/v1/models", "/get_server_info", "/v1/chat/completions"):
            for auth in (None, "Bearer invalid-acceptance-key"):
                headers = {} if auth is None else {"Authorization": auth}
                request = urllib.request.Request(endpoint + path, headers=headers)
                if path.endswith("completions"):
                    request = urllib.request.Request(
                        endpoint + path,
                        data=b"{}",
                        headers={**headers, "Content-Type": "application/json"},
                    )
                try:
                    OPENER.open(request, timeout=30).close()
                    raise AssertionError("unauthorized request accepted: " + path)
                except urllib.error.HTTPError as exc:
                    assert exc.code == 401, (path, exc.code)
        models = client.call("models", "/v1/models")
        assert MODEL in {r["id"] for r in models["data"]}
        question = [{"role": "user", "content": "Reply with a short greeting."}]
        message, initial = client.chat("plain", question, return_token_ids=True)
        assert message.get("content")
        prompt_ids = initial["choices"][0]["prompt_token_ids"]
        assert prompt_ids and all(type(i) is int for i in prompt_ids)
        for streaming in (False, True):
            name = "reasoning-stream" if streaming else "reasoning"
            message, _ = client.chat(
                name,
                [
                    {
                        "role": "user",
                        "content": "What is 7 + 8? Think briefly, then answer.",
                    }
                ],
                stream=streaming,
                thinking=True,
                max_tokens=2048,
            )
            assert message.get("reasoning_content") and message.get("content")
            tool_messages = [
                {
                    "role": "user",
                    "content": "Use lookup_temperature to get the current temperature in Paris. Do not guess.",
                }
            ]
            name = "tool-stream" if streaming else "tool"
            message, _ = client.chat(
                name,
                tool_messages,
                stream=streaming,
                tools=[TOOL],
                tool_choice="required",
            )
            assert len(message.get("tool_calls", [])) == 1
            call = message["tool_calls"][0]
            assert call["id"] and call["function"]["name"] == TOOL["function"]["name"]
            assert json.loads(call["function"]["arguments"])["city"].lower() == "paris"
            followup = [
                *tool_messages,
                message,
                {
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": '{"city":"Paris","temperature_c":17}',
                },
            ]
            final, _ = client.chat(
                name + "-result",
                followup,
                stream=streaming,
                tools=[TOOL],
                tool_choice="none",
            )
            assert final.get("content") and not final.get("tool_calls")
            client.receipts[name]["tool_roundtrip_passed"] = True
        native = []
        for streaming in (False, True):
            name = "native-stream" if streaming else "native"
            response = client.call(
                name,
                "/generate",
                {
                    "input_ids": prompt_ids,
                    "sampling_params": {
                        "temperature": 0,
                        "ignore_eos": False,
                        "max_new_tokens": 512,
                    },
                    "stream": streaming,
                },
                streaming,
            )
            if streaming:
                ids = []
                for event in response["events"]:
                    value = event["response"]
                    delta, count = (
                        value.get("output_ids", []),
                        value["meta_info"]["completion_tokens"],
                    )
                    if len(delta) == count:
                        assert delta[: len(ids)] == ids
                        ids = list(delta)
                    else:
                        assert len(ids) + len(delta) == count
                        ids.extend(delta)
                response = {
                    "output_ids": ids,
                    "meta_info": response["events"][-1]["response"]["meta_info"],
                }
            audited = native_audit(response, 512)
            assert audited["finish"]["type"] == "stop", "greeting did not reach EOS"
            client.receipts[name].update(audited)
            native.append(audited)
        assert sum(r["verify"] for r in native) > 0

        def concurrent_chat(i):
            msg, _ = client.chat(
                f"concurrent-{i}",
                [{"role": "user", "content": f"Briefly say hello to reader {i}."}],
                max_tokens=128,
            )
            assert msg.get("content")

        with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
            list(pool.map(concurrent_chat, range(2)))
        long_text = "\n".join(
            f"Record {i:04d}: sample alpha beta gamma; value {i % 97}."
            for i in range(900)
        )
        msg, response = client.chat(
            "long-prefix",
            [
                {
                    "role": "user",
                    "content": long_text
                    + "\nAcknowledge these records in one sentence.",
                }
            ],
            max_tokens=128,
            return_token_ids=True,
        )
        assert msg.get("content") and 8192 < response["usage"]["prompt_tokens"] < 130944
        cached = client.call(
            "long-prefix-cache",
            "/generate",
            {
                "input_ids": response["choices"][0]["prompt_token_ids"],
                "sampling_params": {
                    "temperature": 0,
                    "ignore_eos": False,
                    "max_new_tokens": 128,
                },
            },
        )
        client.receipts["long-prefix-cache"].update(native_audit(cached, 128))
        assert cached["meta_info"]["cached_tokens"] > 0
        result = {
            "passed": True,
            "time": time.time(),
            "worker": worker,
            "scope": "synthetic auth/models/reasoning/tools+result, stream/nonstream, native EOS/spec counts, concurrency2 and long-prefix/cache; no throughput optimum or held-out-test claim",
            "lossless_greedy_verified": False,
            "requests": client.receipts,
        }
        write(ROOT / worker / "api-acceptance.json", result)
        print(
            json.dumps(
                {"worker": worker, "passed": True, "requests": len(client.receipts)}
            )
        )
    except Exception as exc:
        write(
            ROOT / worker / "api-acceptance-failure.json",
            {
                "type": type(exc).__name__,
                "time": time.time(),
                "completed_requests": client.receipts,
                "no_retry": True,
            },
        )
        raise


if __name__ == "__main__":
    main()
