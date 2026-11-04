"""Exercise the real HTTP/SSE client and exact pinned server EOS contracts on CPU."""

import argparse
import ast
import importlib.util
import json
import math
import runpy
import tempfile
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace

import evaluate


def native_contracts(source):
    base = source / "python/sglang" if (source / "python/sglang").is_dir() else source
    path = base / "srt/managers/schedule_batch.py"
    tree = ast.parse(path.read_text())
    names = {"BaseFinishReason", "FINISH_MATCHED_TOKEN", "FINISH_LENGTH"}
    definitions = [
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name in names
    ]
    req = next(
        node
        for node in tree.body
        if isinstance(node, ast.ClassDef) and node.name == "Req"
    )
    methods = {
        "_check_token_based_finish",
        "_cap_finished_len_at_max_new_tokens",
        "update_finish_state",
        "output_ids_through_stop",
    }
    definitions += [
        node
        for node in req.body
        if isinstance(node, ast.FunctionDef) and node.name in methods
    ]
    code = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *definitions,
        ],
        type_ignores=[],
    )
    namespace = {}
    exec(compile(ast.fix_missing_locations(code), str(path), "exec"), namespace)
    fixture = type("NativeReqContract", (), {name: namespace[name] for name in methods})
    fixture.finished = lambda self: self.finished_reason is not None
    fixture._check_vocab_boundary_finish = lambda self, tokens: False
    fixture._check_str_based_finish = lambda self, count: False
    for tokens, accepted, budget, expected, reason in (
        ([1, 9, 9], 3, 1024, [1], "stop"),
        ([7, 8, 1, 9, 10, 11], 6, 1024, [7, 8, 1], "stop"),
        ([7, 8, 9, 1, 10], 5, 3, [7, 8, 9], "length"),
        ([7, 8, 1, 10], 4, 3, [7, 8, 1], "stop"),
    ):
        row = fixture()
        row.output_ids = tokens
        row.finished_reason = row.finished_len = row.to_finish = row.tokenizer = (
            row.grammar
        ) = None
        row.eos_token_ids = {1}
        row.sampling_params = SimpleNamespace(
            ignore_eos=False, stop_token_ids=set(), max_new_tokens=budget
        )
        row.update_finish_state(accepted)
        assert row.output_ids_through_stop == expected
        finish = row.finished_reason.to_json()
        assert finish["type"] == reason
        evaluate.validate_completion(
            expected,
            {
                "completion_tokens": len(expected),
                "prompt_tokens": 4,
                "finish_reason": finish,
            },
            budget,
            4,
        )
    spec_path = base / "srt/speculative/spec_decoding_metrics.py"
    spec = importlib.util.spec_from_file_location("pinned_spec_metrics", spec_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    # Static verify mode; this pure metrics function otherwise imports the GPU package.
    module.ragged_verify_cap_accept = lambda: False
    for count, verifies, correct in ((1, 0, 0), (2, 1, 5), (61, 20, 60)):
        meta = {"finish_reason": {"type": "stop", "matched": 1}}
        module.populate_spec_info(
            meta,
            num_draft_tokens=6,
            completion_tokens=count,
            verify_ct=verifies,
            num_correct_drafts=correct,
            num_cap_tokens=None,
            num_block_accept_tokens=None,
            correct_drafts_histogram=None,
            cap_lens_histogram=None,
        )
        evaluate.validate_stats(
            {"output_ids": [7] * (count - 1) + [1], "meta_info": meta}, True
        )


def client_contracts(previous_evaluate):
    pending = []
    bodies = []

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            bodies.append(body)
            events = pending.pop(0)
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.end_headers()
            for event in events:
                self.wfile.write(b"data: " + json.dumps(event).encode() + b"\n\n")
            self.wfile.write(b"data: [DONE]\n\n")
            self.wfile.flush()

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    original_url = evaluate.URL
    evaluate.URL = f"http://127.0.0.1:{server.server_port}"
    row = {"id": "fixture", "input_ids": [4, 5, 6, 7]}

    def events(ids, first_count, finish):
        return [
            {
                "output_ids": ids[:first_count],
                "meta_info": {"prompt_tokens": 4, "completion_tokens": first_count},
            },
            {
                "output_ids": ids,
                "meta_info": {
                    "prompt_tokens": 4,
                    "completion_tokens": len(ids),
                    "finish_reason": finish,
                },
            },
        ]

    try:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            results = []
            for index, (ids, first, budget, finish) in enumerate(
                (
                    ([7, 8, 9, 1], 2, 1024, {"type": "stop", "matched": 1}),
                    ([1], 1, 1024, {"type": "stop", "matched": 1}),
                    ([7, 8, 9], 1, 3, {"type": "length", "length": 3}),
                )
            ):
                pending.append(events(ids, first, finish))
                result = evaluate.generate(row, budget, root / f"case-{index}.json")
                assert bodies[-1]["sampling_params"]["ignore_eos"] is False
                assert (
                    result["output_ids"] == ids
                    and result["first_chunk_tokens"] == first
                )
                assert math.isclose(
                    result["output_tokens_per_second"],
                    len(ids) / result["duration_seconds"],
                )
                if len(ids) > first:
                    assert math.isclose(
                        result["tpot_seconds"],
                        (result["duration_seconds"] - result["ttft_seconds"])
                        / (len(ids) - first),
                    )
                else:
                    assert result["tpot_seconds"] is None
                evaluate.validate_stats(result, False)
                results.append(result)
            summary = evaluate.aggregate(results, False)
            assert (
                summary["output_tokens"] == 8
                and summary["requests_with_defined_tpot"] == 2
            )
            if previous_evaluate:
                old = runpy.run_path(str(previous_evaluate))
                generate = old["generate"]
                generate.__globals__["URL"] = evaluate.URL
                pending.append(events([7, 1], 1, {"type": "stop", "matched": 1}))
                try:
                    generate(row, 1024, root / "old-eos-contract.json")
                except AssertionError:
                    pass
                else:
                    raise AssertionError(
                        "old fixed-length client unexpectedly accepts short EOS completion"
                    )
            for ids, finish, budget in (
                ([7, 1, 9], {"type": "stop", "matched": 1}, 1024),
                ([7], {"type": "length", "length": 1024}, 1024),
                ([7, 1], {"type": "length", "length": 2}, 2),
            ):
                try:
                    evaluate.validate_completion(
                        ids,
                        {
                            "completion_tokens": len(ids),
                            "prompt_tokens": 4,
                            "finish_reason": finish,
                        },
                        budget,
                        4,
                    )
                except AssertionError:
                    pass
                else:
                    raise AssertionError("invalid EOS/length finish accepted")
    finally:
        evaluate.URL = original_url
        server.shutdown()
        server.server_close()
        thread.join()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--serving-source", type=Path, required=True)
    parser.add_argument("--previous-evaluate", type=Path)
    args = parser.parse_args()
    native_contracts(args.serving_source)
    client_contracts(args.previous_evaluate)
    print(
        json.dumps(
            {
                "passed": True,
                "scope": "Actual pinned native EOS/cap methods and spec metric function, real HTTP/SSE client, short/one-token/budget completions, post-EOS and premature-finish rejection, terminal block counters, variable length timing; no GPU numerical parity claim",
                "old_client_short_eos_failure_reproduced": bool(args.previous_evaluate),
            }
        )
    )
