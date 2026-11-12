"""Independent semantic acceptance for request-local GPU n-gram drafting.

Run through an immutable YiCloud module Run. ``--reference-only`` checks the
fixture/oracle locally and does not claim CUDA acceptance.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import random
import struct
import time
from pathlib import Path


FIELDS = ("tokens", "lengths", "match_len", "agreement", "copy_len")


@dataclasses.dataclass
class Case:
    name: str
    histories: list[list[int]]
    lengths: list[int]
    slots: list[int]
    width: int


def oracle(history, length, min_match=5, max_match=15, max_draft=15):
    """Enumerate every visible endpoint without calling production helpers."""
    empty = dict(tokens=[0] * max_draft, lengths=0, match_len=0,
                 agreement=1, copy_len=0)
    if not 0 <= length <= len(history):
        return empty
    visible = history[:length]
    ranked = []
    for endpoint in range(length - 1):
        matched = 0
        while (matched < max_match and matched <= endpoint
               and visible[endpoint - matched] == visible[-1 - matched]):
            matched += 1
        if matched >= min_match:
            ranked.append((matched, endpoint))
    if not ranked:
        return empty
    ranked.sort(reverse=True)
    matched, endpoint = ranked[0]
    continuation = visible[endpoint + 1:endpoint + 1 + max_draft]
    agreement = (len(ranked) < 2 or
                 visible[ranked[1][1] + 1] == continuation[0])
    copy_len = matched
    if matched == max_match:
        while (copy_len < 256 and copy_len <= endpoint
               and visible[endpoint - copy_len] == visible[-1 - copy_len]):
            copy_len += 1
    return dict(tokens=continuation + [0] * (max_draft - len(continuation)),
                lengths=len(continuation), match_len=matched,
                agreement=int(agreement), copy_len=copy_len)


def expected(case):
    rows = []
    for slot in case.slots:
        if 0 <= slot < len(case.histories):
            rows.append(oracle(case.histories[slot], case.lengths[slot]))
        else:
            rows.append(oracle([], 0))
    return {field: [row[field] for row in rows] for field in FIELDS}


def make_case(name, histories, *, slots=None, lengths=None, width=None):
    lengths = list(map(len, histories)) if lengths is None else list(lengths)
    width = max(16, max(map(len, histories), default=0) + 19) if width is None else width
    # Padding contains a repeated pattern that would become a valid match if
    # the scan uses allocation width rather than the device-visible length.
    padded = [row + [701 + i % 15 for i in range(width - len(row))]
              for row in histories]
    return Case(name, padded, lengths,
                list(range(len(histories))) if slots is None else slots, width)


def fixtures(seed, random_cases):
    short = list(range(101, 106))
    full = list(range(201, 216))
    copy = list(range(801, 816))
    fixed = [
        ("empty", []),
        ("below-minimum", [11, 12, 13, 14, 11, 12, 13, 14]),
        ("no-match", list(range(64))),
        ("minimum-match-full-copy", short + copy + [900] + short),
        ("full-match-full-copy", full + copy + [900] + full),
        ("one-visible-continuation", [7] * 20),
        ("overlap", [7, 8, 9] * 19),
        ("runner-up-disagrees", short + [31, 900] + short + [32, 901] + short),
        ("runner-up-agrees", short + [31, 900] + short + [31, 901] + short),
        ("longest-before-recent", full + copy + [900] + full[-5:] + [31, 901] + full),
        ("full-copy-backward-cap", [9] * 600),
        ("copy-backward-before-cap", list(range(1000, 1040)) + [700]
         + list(range(1000, 1040))),
        ("token-zero", [0] * 7 + [1, 2] + [0] * 7),
        ("large-token-ids", [2**30 + x for x in range(15)] + [2**31 - 1]
         + [2**30 + x for x in range(15)]),
    ]
    yield make_case("fixed", [row for _, row in fixed],
                    slots=[*range(len(fixed)), 5, 3, -1, len(fixed)])
    invalid = make_case("invalid-lengths", [full + copy + full] * 4,
                        lengths=[-1, 0, 15, 100], width=64)
    yield invalid
    # A self-copy in storage is invisible beyond length=15.
    yield make_case("adversarial-hidden-future", [full + copy + full], lengths=[15])
    rng = random.Random(seed)
    for index in range(random_cases):
        count = rng.choice([1, 2, 3, 4, 7, 17])
        histories = []
        for _ in range(count):
            length = rng.choice([0, 4, 5, 14, 15, 16, 31, 63, 257, 513, 1025])
            vocab = rng.choice([2, 3, 8, 64, 2048])
            row = [rng.randrange(vocab) for _ in range(length)]
            if length > 40 and rng.random() < 0.75:
                match = rng.randint(5, 15)
                end = rng.randint(match - 1, length - match - 2)
                row[-match:] = row[end - match + 1:end + 1]
            histories.append(row)
        slots = list(range(count))
        rng.shuffle(slots)
        slots += [rng.randrange(count), -1, count]
        yield make_case(f"random-{index:04d}", histories, slots=slots)
    for length in (8192, 32768, 65536):
        histories = []
        for row_index in range(4):
            row = list(range(10000 + row_index * length,
                             10000 + (row_index + 1) * length))
            row[1024:1039] = full
            row[1039:1054] = copy
            row[-15:] = full
            histories.append(row)
        yield make_case(f"context-{length}", histories, slots=[3, 0, 1, 2], width=length)
    # Large request batches exercise grid tails and duplicate slot mappings.
    yield make_case("batch-129", [full + copy + [900] + full] * 129,
                    slots=list(reversed(range(129))) + [0, -1, 129])


def fingerprint(case):
    digest = hashlib.sha256()
    for row in case.histories:
        digest.update(struct.pack(f"<{len(row)}i", *row))
    digest.update(struct.pack(f"<{len(case.lengths)}q", *case.lengths))
    digest.update(struct.pack(f"<{len(case.slots)}q", *case.slots))
    return digest.hexdigest()


def validate_oracle():
    assert oracle([], 0)["agreement"] == 1
    assert oracle([1, 2, 3, 4], 4)["lengths"] == 0
    assert oracle([7] * 20, 20) == dict(tokens=[7] + [0] * 14,
        lengths=1, match_len=15, agreement=1, copy_len=19)
    assert oracle([9] * 600, 600)["copy_len"] == 256
    short = list(range(101, 106))
    row = short + [31, 900] + short + [32, 901] + short
    result = oracle(row, len(row))
    assert result["tokens"][:2] == [32, 901]
    assert result["match_len"] == result["copy_len"] == 5
    assert result["agreement"] == 0
    row = list(range(15)) + [99] + list(range(15))
    assert oracle(row, len(row))["lengths"] == 15


def to_device(case, torch, length_dtype):
    return (torch.tensor(case.histories, device="cuda", dtype=torch.int32),
            torch.tensor(case.lengths, device="cuda", dtype=length_dtype),
            torch.tensor(case.slots, device="cuda", dtype=torch.int64))


def compare(actual, reference, name, torch):
    for field in FIELDS:
        tensor = getattr(actual, field)
        dtype = torch.int64 if field == "tokens" else torch.int32
        assert tensor.device.type == "cuda" and tensor.dtype == dtype, (name, field)
        got = tensor.cpu().tolist()
        assert got == reference[field], (
            f"{name}: {field} differs; actual={got!r}; expected={reference[field]!r}")


def selection_oracle(mtp, mp, ngram, np, price, contains, mode):
    output = dict(tokens=[], survival=[], lengths=[], source=[])
    for row, mt in enumerate(mtp):
        horizon, width = len(ngram["tokens"][row]), len(mt)
        m, n = [], []
        pm = pn = 1.0
        for pos in range(horizon):
            if pos < width:
                pm *= mp[row][pos]
            pn *= np[row][pos]
            m.append(pm if pos < width else 0.0)
            n.append(pn if pos < ngram["lengths"][row] else 0.0)
        extension = (contains and ngram["lengths"][row] > width
                     and ngram["tokens"][row][:width] == mt)
        mv, nv = (sum(max(s - price, 0) for s in chain) for chain in (m, n))
        use_ngram = ngram["lengths"][row] > 0 and (nv >= mv or extension)
        if mode == "mtp":
            use_ngram = extension = False
        elif mode == "ngram":
            use_ngram = True  # n-gram-only ablation uses anchor-only on no match
            extension = False
        if use_ngram and extension:
            n[:width] = [max(a, b) for a, b in zip(m[:width], n[:width])]
            for pos in range(width, horizon):
                n[pos] = n[pos - 1] * np[row][pos]
        length = ngram["lengths"][row] if use_ngram else width
        tokens = ngram["tokens"][row] if use_ngram else mt + [0] * (horizon - width)
        chain = n if use_ngram else m
        output["tokens"].append(tokens[:length] + [0] * (horizon - length))
        output["survival"].append(chain[:length] + [0.0] * (horizon - length))
        output["lengths"].append(length)
        output["source"].append(2 if use_ngram and extension else int(use_ngram))
    return output


def allocation_oracle(selected, budget):
    if budget is None:
        return [length + 1 for length in selected["lengths"]]
    ranked = sorted((-survival, pos, row)
                    for row, chain in enumerate(selected["survival"])
                    for pos, survival in enumerate(chain)
                    if pos < selected["lengths"][row] and survival >= 1e-6)
    counts = [1] * len(selected["lengths"])
    for _, _, row in ranked[:budget]:
        counts[row] += 1
    return counts


def compare_selection(actual, reference, name, torch):
    for field in ("tokens", "survival", "lengths", "source"):
        tensor = getattr(actual, field)
        target = torch.tensor(reference[field], dtype=tensor.dtype)
        if field == "survival":
            torch.testing.assert_close(tensor.cpu(), target, atol=1e-7, rtol=1e-5,
                                       msg=lambda message: f"{name}: {message}")
            assert torch.all(tensor[:, 1:] <= tensor[:, :-1])
            assert torch.isfinite(tensor).all()
        else:
            assert tensor.cpu().tolist() == reference[field], (name, field)


COEFFICIENTS = dict(intercepts=[-0.4 + 0.03 * k for k in range(15)],
                    match=[0.07, 0.03], agreement=0.2,
                    full=[0.1, 0.08], copy=[0.12, 0.04])


def estimate_oracle(ngram):
    output = []
    for row, length in enumerate(ngram["lengths"]):
        n, m = ngram["match_len"][row], ngram["copy_len"][row]
        probabilities = []
        for pos in range(15):
            kind = int(pos > 0)
            value = (COEFFICIENTS["intercepts"][pos]
                     + COEFFICIENTS["match"][kind] * (n - 5)
                     + COEFFICIENTS["agreement"] * ngram["agreement"][row])
            if n == 15:
                value += (COEFFICIENTS["full"][kind]
                          + COEFFICIENTS["copy"][kind] * math.log(max(m, 15) / 15))
            probabilities.append(1 / (1 + math.exp(-value)) if pos < length else 0.0)
        output.append(probabilities)
    return output


def selection_gate(torch, device="cuda"):
    from specforge.hybrid.ngram import NgramResult
    from specforge.hybrid.selection import select_chains, verify_lengths

    rng = random.Random(139)
    records = []
    for batch in (1, 4, 17):
        mtp = [[rng.randrange(256) for _ in range(5)] for _ in range(batch)]
        mp = [[rng.choice([0.25, 0.5, 1.0]) for _ in range(5)] for _ in range(batch)]
        ng = dict(tokens=[], lengths=[])
        np = []
        for row in range(batch):
            length = [15, 0, 4, 15, 5, 6, 1][row % 7]
            tokens = mtp[row] if row % 3 == 0 else [rng.randrange(256) for _ in range(5)]
            tokens = (tokens + [rng.randrange(256) for _ in range(10)])[:length]
            ng["tokens"].append(tokens + [0] * (15 - length))
            ng["lengths"].append(length)
            np.append([rng.choice([0.25, 0.5, 1.0]) if p < length else 0.0
                       for p in range(15)])
        ng_result = NgramResult(torch.tensor(ng["tokens"], device=device),
            torch.tensor(ng["lengths"], device=device, dtype=torch.int32),
            *[torch.zeros(batch, device=device, dtype=torch.int32) for _ in range(3)])
        mt, mc = (torch.tensor(value, device=device) for value in (mtp, mp))
        nc = torch.tensor(np, device=device)
        for mode in ("hybrid", "mtp", "ngram"):
            for contains in (True, False):
                for price in (0.0, 0.25, 1.0):
                    name = f"selection-b{batch}-{mode}-contains{contains}-price{price}"
                    reference = selection_oracle(mtp, mp, ng, np, price, contains, mode)
                    actual = select_chains(mt, mc, ng_result, nc, mode=mode,
                                           contains=contains, slot_price=price)
                    compare_selection(actual, reference, name, torch)
                    for budget in (None, 0, 1, batch, batch + 1, batch * 15 + 7):
                        lengths = verify_lengths(actual.survival, actual.lengths, budget)
                        assert lengths.cpu().tolist() == allocation_oracle(reference, budget), (name, budget)
                    records.append(name)
    tied = dict(lengths=[15] * 4, survival=[[1.0] * 15] * 4)
    tensor = torch.ones((4, 15), device=device)
    lengths = torch.full((4,), 15, device=device, dtype=torch.int32)
    for budget in (0, 1, 4, 5, 17, 60, 100):
        assert verify_lengths(tensor, lengths, budget).cpu().tolist() == allocation_oracle(tied, budget)
    return records


def graph_gate(scan, torch, seed, length_dtype):
    from specforge.hybrid.selection import estimate_ngram, select_chains, verify_lengths

    rng = random.Random(seed + 2000)
    cases = []
    full = list(range(100, 115))
    for index in range(6):
        histories = [[rng.randrange(2048) for _ in range(2048)] for _ in range(5)]
        if index != 1:
            histories[2][30:45] = full
            histories[2][45:60] = list(range(700 + index, 715 + index))
            if index % 2 == 0:
                histories[2][45:50] = full[:5]
            histories[2][150:165] = full
        lengths = [rng.randrange(16, 1024) for _ in range(5)]
        lengths[2] = 165
        if index == 4:
            lengths[2] = 0  # reused slot after a completed request
        slots = [2, index % 5, -1 if index % 2 else 3, 2]
        cases.append(make_case(f"graph-replay-{index}", histories,
                               lengths=lengths, slots=slots, width=2048))
    device_inputs = to_device(cases[0], torch, length_dtype)
    mtp = [full[:5]] * len(cases[0].slots)
    mp = [[0.75] * 5] * len(mtp)
    mt, mc = (torch.tensor(value, device="cuda") for value in (mtp, mp))
    coefficients = {key: torch.as_tensor(value, device="cuda", dtype=torch.float32)
                    for key, value in COEFFICIENTS.items()}

    def pipeline():
        ng = scan(*device_inputs)
        conditional = estimate_ngram(ng, coefficients)
        selected = select_chains(mt, mc, ng, conditional, slot_price=0.2)
        verify = verify_lengths(selected.survival, selected.lengths, budget=13)
        return ng, conditional, selected, verify

    stream = torch.cuda.Stream()
    stream.wait_stream(torch.cuda.current_stream())
    with torch.cuda.stream(stream):
        for _ in range(3):
            pipeline()
    torch.cuda.current_stream().wait_stream(stream)
    torch.cuda.synchronize()
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        captured, conditional, selected, verify = pipeline()
    records = []
    for case in cases:
        changed = to_device(case, torch, length_dtype)
        for target, source in zip(device_inputs, changed):
            target.copy_(source)
        graph.replay()
        torch.cuda.synchronize()
        ng_reference = expected(case)
        compare(captured, ng_reference, case.name, torch)
        probabilities = estimate_oracle(ng_reference)
        torch.testing.assert_close(conditional.cpu(), torch.tensor(probabilities),
                                   rtol=1e-5, atol=1e-7)
        reference = selection_oracle(mtp, mp, ng_reference, probabilities,
                                     0.2, True, "hybrid")
        compare_selection(selected, reference, case.name, torch)
        assert verify.cpu().tolist() == allocation_oracle(reference, 13), case.name
        records.append(dict(name=case.name, input_sha256=fingerprint(case)))
    return records


def benchmark(scan, torch, cases, repeats):
    records = []
    for case in cases:
        if not case.name.startswith("context-"):
            continue
        for batch in (1, 4):
            inputs = list(to_device(case, torch, torch.int32))
            inputs[2] = inputs[2][:batch]
            for _ in range(5):
                scan(*inputs)
            torch.cuda.synchronize()
            start, end = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
            start.record()
            for _ in range(repeats):
                scan(*inputs)
            end.record()
            end.synchronize()
            records.append(dict(context_length=case.width, batch=batch,
                mean_cuda_milliseconds=start.elapsed_time(end) / repeats,
                repeats=repeats, scope="eager scan calls after compilation and warmup"))
    return records


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--random-cases", type=int, default=128)
    parser.add_argument("--benchmark-repeats", type=int, default=50)
    parser.add_argument("--reference-only", action="store_true")
    args = parser.parse_args()
    assert args.random_cases >= 0 and args.benchmark_repeats > 0
    validate_oracle()
    cases = list(fixtures(args.seed, args.random_cases))
    report = dict(seed=args.seed, scope="request-local ngram scan",
                  oracle_version="visible-endpoints-longest-recent-v1",
                  reference_only=args.reference_only,
                  cases=[dict(name=case.name, requests=len(case.slots),
                              input_sha256=fingerprint(case)) for case in cases])
    started = time.monotonic()
    if args.reference_only:
        for case in cases:
            results = expected(case)
            for row, length in enumerate(results["lengths"]):
                assert 0 <= length <= 15
                assert results["tokens"][row][length:] == [0] * (15 - length)
                assert results["agreement"][row] in (0, 1)
                assert 0 <= results["copy_len"][row] <= 256
        report["reference_passed"] = True
        report["gpu_passed"] = False
    else:
        import torch
        import triton
        from specforge.hybrid.ngram import scan

        assert torch.cuda.is_available(), "a real CUDA device is required"
        references = [expected(case) for case in cases]
        for dtype in (torch.int32, torch.int64):
            for case, reference in zip(cases, references):
                compare(scan(*to_device(case, torch, dtype)), reference,
                        f"{case.name}/{dtype}", torch)
        report["graph_replays"] = {
            str(dtype): graph_gate(scan, torch, args.seed, dtype)
            for dtype in (torch.int32, torch.int64)}
        report["selection_cases"] = selection_gate(torch)
        report["timings"] = benchmark(scan, torch, cases, args.benchmark_repeats)
        report["runtime"] = dict(torch=torch.__version__, triton=triton.__version__,
            cuda=torch.version.cuda, device=torch.cuda.get_device_name(),
            capability=list(torch.cuda.get_device_capability()))
        report["gpu_passed"] = True
    report["elapsed_seconds"] = time.monotonic() - started
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({key: report[key] for key in
                      ("reference_only", "gpu_passed", "elapsed_seconds")}))


if __name__ == "__main__":
    main()
