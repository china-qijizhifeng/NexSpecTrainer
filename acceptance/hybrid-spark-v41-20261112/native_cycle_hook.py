"""Observe committed native DSPARK cycles in a dedicated acceptance server.

The hook reads existing CPU request fields immediately before output streaming.
It does not alter responses, model state, or introduce CUDA synchronizations.
"""

import ast
import hashlib
import json
import os
from pathlib import Path


PROCESSOR_PATH = "srt/managers/scheduler_components/batch_result_processor.py"
PROCESSOR_SHA256 = "31870b8591b10ce6232546ff89171f49c20f066a40cbc848ee2901f5a53aaaa3"
DIRECTORY_ENV = "HYBRID_ACCEPTANCE_VERIFY_CYCLES_DIR"


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(f".tmp-{os.getpid()}")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    os.replace(temporary, path)


def cycle_path(directory, request_id):
    return Path(directory) / (hashlib.sha256(request_id.encode()).hexdigest() + ".json")


def capture_before(reqs):
    if not os.environ.get(DIRECTORY_ENV):
        return []
    return [(req, req.spec_verify_ct, len(req.output_ids_through_stop), req.spec_num_correct_drafts)
            for req in reqs]


def capture_after(before):
    if not before:
        return
    from sglang.srt.runtime_context import get_parallel

    if get_parallel().attn_tp_rank != 0:
        return
    for req, verify_before, output_before, accepted_before in before:
        if req.spec_verify_ct == verify_before:
            continue
        if req.spec_verify_ct != verify_before + 1:
            raise ValueError("acceptance observer expected one native verify in a decode batch")
        if verify_before == 0:
            req._hybrid_acceptance_cycles = []
            req._hybrid_acceptance_prefill = output_before
        if len(req._hybrid_acceptance_cycles) != verify_before:
            raise ValueError("acceptance observer missed native verify cycles")
        emitted = len(req.output_ids_through_stop) - output_before
        accepted = req.spec_num_correct_drafts - accepted_before
        if min(emitted, accepted) < 0 or emitted > accepted + 1:
            raise ValueError("invalid committed native verify counters")
        req._hybrid_acceptance_cycles.append(dict(
            cycle_id=verify_before, ctx_len=len(req.origin_input_ids) + output_before,
            committed_output_tokens=emitted, accepted_draft_tokens=accepted))
        if req.finished():
            atomic_json(cycle_path(os.environ[DIRECTORY_ENV], req.rid), dict(
                schema_version=1, observer="native CPU request deltas before output streaming",
                request_id=req.rid, prompt_tokens=len(req.origin_input_ids),
                prefill_tokens=req._hybrid_acceptance_prefill,
                completion_tokens=len(req.output_ids_through_stop),
                spec_verify_ct=req.spec_verify_ct,
                spec_num_correct_drafts=req.spec_num_correct_drafts,
                cycles=req._hybrid_acceptance_cycles))


def instrument_text(text):
    method = "    def process_batch_result_decode("
    start = "        auxiliary_output_starts = self.snapshot_auxiliary_output_starts(batch, result)"
    end = ("        self.output_streamer.stream_output(batch.reqs, batch.return_logprob)\n"
           "        self.token_to_kv_pool_allocator.free_group_end()")
    if text.count(method) != 1:
        raise ValueError("expected one pinned decode result processor")
    position = text.index(method)
    head, body = text[:position], text[position:]
    if body.count(start) != 1 or body.count(end) != 1:
        raise ValueError("native acceptance observation anchors changed")
    body = body.replace(start, "        from native_cycle_hook import capture_before, capture_after\n"
                              "        hybrid_acceptance_before = capture_before(batch.reqs)\n" + start, 1)
    body = body.replace(end, "        capture_after(hybrid_acceptance_before)\n" + end, 1)
    result = head + body
    ast.parse(result)
    return result


def install(sglang_source, receipt):
    path = Path(sglang_source) / PROCESSOR_PATH
    original = path.read_bytes()
    before = hashlib.sha256(original).hexdigest()
    if before != PROCESSOR_SHA256:
        raise ValueError(f"acceptance processor source drift: {before}")
    patched = instrument_text(original.decode()).encode()
    path.write_bytes(patched)
    atomic_json(receipt, dict(source_path=str(path), before_sha256=before,
        after_sha256=hashlib.sha256(patched).hexdigest(),
        observer_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        scope="dedicated acceptance server only; CPU request counters; no response schema change"))


def load_cycles(directory, row, max_draft):
    path = cycle_path(directory, row["request_id"])
    record = json.loads(path.read_text())
    prompt_tokens = len(row["input_ids"])
    meta = row["meta_info"]
    if (record["schema_version"] != 1 or record["request_id"] != row["request_id"]
            or record["prompt_tokens"] != prompt_tokens):
        raise ValueError("native observation request identity mismatch")
    prefill, cycles = record["prefill_tokens"], record["cycles"]
    if type(prefill) is not int or prefill != 1:
        raise ValueError("expected one initial native prefill token")
    if not cycles or len(cycles) != meta["spec_verify_ct"] or record["spec_verify_ct"] != len(cycles):
        raise ValueError("native observation verify count mismatch")
    output, correct = prefill, 0
    for index, cycle in enumerate(cycles):
        ctx, emitted, accepted = (cycle[key] for key in
            ("ctx_len", "committed_output_tokens", "accepted_draft_tokens"))
        if any(type(value) is not int or value < 0 for value in (ctx, emitted, accepted)):
            raise ValueError("native observation contains invalid counters")
        if cycle["cycle_id"] != index or ctx != prompt_tokens + output:
            raise ValueError("native observation cycles are not contiguous")
        if accepted > max_draft or emitted > accepted + 1:
            raise ValueError("native observation exceeds the configured draft horizon")
        output += emitted
        correct += accepted
    if output != row["completion_tokens"] or output != record["completion_tokens"]:
        raise ValueError("native observation completion total mismatch")
    if correct != record["spec_num_correct_drafts"] or correct != meta["spec_num_correct_drafts"]:
        raise ValueError("native observation accepted draft total mismatch")
    return dict(request_id=row["request_id"], sample_id=row["sample_id"],
                source=str(path), prefill_tokens=prefill, cycles=cycles)
