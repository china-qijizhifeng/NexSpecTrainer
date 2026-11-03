# V4.1 prefill and capture benchmark

This benchmark distinguishes synchronized model-forward diagnostics, HTTP
request latency, and the wall time until validated features are durably written.
Each token is counted once across TP8/EP8. Profiling phases and steady-state
throughput phases are separate. No-publication requests retain the global
auxiliary hidden-state computation; they omit publication and transfer, not
that computation.

The pinned run uses DeepSeek-V4.1-Flash-1017 revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`, SGLang
`38d9e4a3fe4a493b0ae468477a274e83defda81a`, image digest
`c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa`,
and 8 H100s on BP node044. Engram is CPU/per-rank/all-reduce, graphs disabled,
context length 8192, chunked prefill disabled, one running request, full
auxiliary layers 37/38/39 plus final normalized states. Sixteen prepared
4096-token windows are replayed twice per phase; these are independent contexts.

The experiment bundle is
`/gpfs/users/linlizhi/nexspec-v41-prefill-20261103` (mirrored under
`/Users/shom/Downloads/speculative-decoding-training-research/v41/prefill-20261103`).
It retains job YAML, serving command, runtime environment, input receipts,
versions, research, and results. Actual token text, weights and activation
files must not be committed. Training Torch/TE dependencies are not modified.

## Running against the dedicated service

Install the capture plugin from this checkout into the pinned SGLang image,
and add this directory, the checkout and the experiment bundle to PYTHONPATH.
The bundle supplies `collect_hidden_states.validate` and
`offline_capture_layout`; their hashes belong in the run receipt.
The dedicated Mooncake master uses ports 35551/35880 and RDMA `mlx5_00`.

```sh
export SPECCAP_ROOT=/gpfs/users/linlizhi/nexspec-v41-prefill-20261103
export PREFILL_ROOT="$SPECCAP_ROOT"
export PYTHONPATH="$SPECCAP_ROOT/source/benchmarks/v41_prefill:/tmp/nexspec-zstd:$SPECCAP_ROOT/source:$SPECCAP_ROOT"
source "$SPECCAP_ROOT/runtime_env.sh"
python3 "$SPECCAP_ROOT/source/benchmarks/v41_prefill/benchmark.py" \
  --samples "$SPECCAP_ROOT/train-tokenized.jsonl" \
  --output "$SPECCAP_ROOT/new-run" --codec-workers 8
```

Use a new output directory each time. The server's diagnostic hook calls
`instrument.install(scheduler)` after the capture plugin's scheduler install.
For a clean launch this hook imports `runtime_control`; it switches codec
workers and selected kernels between fully drained, sequential phases. The driver
enables DeepInfer scoring and compact candidate scores by default. Pass
`--no-deepinfer-indexer --no-compact-candidates` to use the original scorer, or add
`--fused-indexer` to those two flags for the earlier fused dense-score baseline.
The positive flags remain accepted. These defaults apply to this pinned H100
benchmark driver. `control.json` starts as `{"profile":false,"tag":"startup"}`.
CUDA events enclose the complete model `_forward_raw` call on each rank;
the report uses the maximum rank time. Nested attention/MoE/Engram durations
must not be summed into a model total. Capture copy timing synchronizes the
copy stream and is performed only in diagnostic phases.

`--throughput-only` skips synchronized diagnostic requests. Startup, initial
warmup, file read-back validation and worker setup are excluded from throughput.
Each measured phase includes response, get/decode, verification/hash, optional
encode/fsync, physical sample removal, and request-log overhead. Unique cache
namespaces force a complete prefill. Success requires all files to decode through
the native DSpark normalizer/collator with exact token/mask and tensor hashes.

```sh
python3 benchmarks/v41_prefill/summarize.py EVIDENCE_DIRECTORY \
  baseline-drained codec4-drained codec8-drained codec8-repeat baseline-repeat \
  --output comparison.json
```

`SPECFORGE_ZSTD_WORKERS` is the production opt-in; set it on both server and
collector, independently of these diagnostic hooks. Default remains 1.
The paired run changes it between 1, 4 and 8. Physical removal drain is enabled
in both arms. An initial un-drained run is retained only as diagnosis evidence.

## Continuous 16k and 64k extension

Pass `--tokens 16384` or `--tokens 65536` with matching contiguous inputs.
`--warmup` and `--profile-requests` control diagnostic sample counts; repeats
still apply to each of the three unprofiled phases. The fixed two-source run
uses two repetitions (four requests per phase), so its p95 is just the maximum
observed latency, not a tail-latency estimate.

The isolated long-context bundle is
`/gpfs/users/linlizhi/nexspec-v41-prefill-long-20261103`. It renders the original
80k-token trajectories and selects nested continuous windows from their first
previously validated supervised 4k offset. No independent windows are joined.
Input receipts pin source bytes, token/mask hashes, and offsets.

This run sets context length 69632, max total KV tokens 131072, max prefill
tokens 65536, explicit SWA/full ratio 0.8, and host memory 640 GiB.
The V4.1 pool defaults to 0.1; a small total KV cap must still leave enough
SWA rows for the entire unchunked request plus page headroom. Verify the actual
`token_capacity_swa` after startup, rather than relying on unresolved defaults. It retains full single-forward capture
without chunking/replay/graphs, and includes a 4k calibration under the same
allocation. Compare codecs within this configuration; avoid attributing its
memory or timing differences from the earlier run solely to sequence length.

## Bounded MoE experiment

`--moe-chunk-tokens 8192` bounds the SM90 V4.1 W4A16 fused MoE call after
original top-6 routing. This adopts the bounded-workspace idea from Nextron
`9763a325081d4ff9c2e0df880a339aae33886d88` (`inference/moe.cu`), not its
BF16 grouped-GEMM kernel. Nextron's Qwen H2048/I512/E256/top8 and GLM
H4096/I2048/E288/top8 APIs do not directly accept V4.1 H5120/I2304/E384/top6
MXFP4 weights, activation clamp10 or EP8. Original routing weights, quantized
weights/scales, activation, shared expert and collective order remain intact.
The experiment only limits local CUTLASS token batches; attention and capture
still span the complete sequence in one model forward.

Default is zero (original path). Different GEMM M dimensions can change
rounding or tactics; compare complete final/aux outputs before accepting
performance. The long bundle's `compare_moe.py` reports exact equality plus
relative L2, cosine, and max absolute error; tokens/masks must match exactly.
This is an experimental harness patch, not a production backend selection.

## Fused H100 prefill indexer

`--fused-indexer` enables `fused_indexer.py` for the pinned SGLang BF16
prefill scorer (32 heads, head dimension 128). A Triton dot tile applies the
original BF16 rounding after the dot and weighted product, reduces heads in
FP32, and rounds the sum to BF16 before emitting FP32 scores. The intermediate
`[query rows, heads, keys]` tensor never reaches global memory. Fully future
tiles are skipped. Candidate consumers gather keys using each row's published
block IDs before scoring; excluded blocks are not multiplied. Original top-k,
candidate publication, pool-slot writeback and decode implementations remain.

The kernel retains the existing score row boundaries and final dense score
layout. This bounds score scratch without splitting model prefill or attention.
It is an independent BF16 implementation informed by DeepInfer Hopper PR #92
at `e205d42e4b28185e50ce30f6b7204cd01e59974c`, not a port of its FP8/Gluon
reencoding and asynchronous WGMMA pipeline.

Run the differential GPU check inside the pinned serving image, with this
directory on `PYTHONPATH`:

```sh
python3 benchmarks/v41_prefill/validate_indexer.py --tune --output indexer-validation.json
```

It uses the real SGLang scorer and candidate top-k as oracles. Coverage includes
signed weights, arbitrary BF16 and FP4-valued operands, zero causal visibility,
partial key/block tiles, and invalid candidate IDs. Optional tile tuning includes
both score shapes used by the continuous 16k/64k experiments. Kernel timing is
not an end-to-end throughput result.

`SPECFORGE_INDEXER_AUDIT=1` adds a live comparison of sampled logits and top-k
once per layer and key length. Use one complete warmup request before timing;
the audit synchronizes and must not be included in measured phases. Audit
records are saved under `PREFILL_ROOT/indexer-audit-rank*.jsonl`. Complete
captured aux/final tensors still require paired numerical validation.

## Hash-free capture pipeline

The current benchmark uses `bf16_fields_zstd_v2` wire objects and `NSACTZ02`
files: no object/header SHA256, no Zstd frame checksum, and no activation,
input-file or output-file fingerprints. The matching serving environment is
`SGLANG_SPEC_CAPTURE_WIRE_ENCODING=bf16_fields_zstd_v2`. Both sides of the
experiment install a guard that raises if a capture codec attempts SHA256.
Legacy V1 reading remains available outside this experiment.

`--pipeline-depth 2` overlaps the next HTTP capture with the current request's
fetch/decode/validation/compression/fsync. A single producer and consumer own
their respective stages, with at most two captures outstanding. Depth 1 is
the serial no-hash control. The production writer and
`scripts/prepare_server_hidden_states.py --pipeline-depth 2` use the same
bounded scheduler. Capacity checks reserve the worst-case encoded bytes of
two captures; they do not assume a compression ratio.

Shape/dtype, complete finite-value scans and exact token/mask checks remain
inside throughput timing. Native training read-back and direct comparisons
of every captured tensor element run after timing. JSON shape metadata alone
is not treated as proof of content equality by `summarize.py`. The paired
16k/64k experiment uses eight requests per phase, including pipeline fill and
drain, and stores its direct-comparison receipts in
`/gpfs/users/linlizhi/nexspec-v41-prefill-pipeline-20261103`.

The earlier reports describe their historical checksummed benchmark version;
do not attribute the full difference from those runs solely to pipelining.
Use the new serial/pipeline pair to isolate overlap, and report the historical
comparison separately as the combined removal-of-hashing and pipeline change.


## DeepInfer Hopper pipeline port

The driver enables `--deepinfer-indexer` by default, using the BF16 adaptation of DeepInfer
Hopper PR92's keys-major Gluon kernel. It preserves persistent visible-tile
scheduling, TMA K prefetch, async WGMMA and packed BF16 epilogues. It replaces
full scoring and candidate publishing; candidate consumers use compact scores
by default. Pass `--no-deepinfer-indexer` to disable the DeepInfer scorer.
Use it with the pinned SM90 image; it is not an automatic portable backend selector.

Validate with `python3 benchmarks/v41_prefill/validate_deepinfer.py --output result.json`.
FP32 head-sum order may change intermediate scores; synthetic top-k and complete
model output must be validated. On the tested two traces all full outputs were
bitwise identical. See `DEEPINFER_RESULTS.md` for the same-run16k/64k results:
forward latency−0.74%/−2.45%, full capture−4.14%/+1.54%. Small scorer shapes
regress; use the disable flag for baseline comparisons rather than treating the
kernel microbench as a universal throughput improvement. Upstream attribution/license is retained
in `deepinfer_indexer.py` and `third_party/DeepInfer-LICENSE`.

## Compact candidate scores

The default `--compact-candidates` keeps candidate-consumer scores in candidate-list
order, without allocating or filling a full `[query rows, all keys]` score
matrix and gathering it again. Each query tile writes at most 64 MiB of FP32
candidate scores (16 MiB for the tested 64k input). This is a per-score-tensor
bound, not a bound on all selection scratch or model memory. Full-topk and
candidate-publisher scoring remain on the selected backend. Pass
`--no-compact-candidates` to disable compact storage independently; this selects
the prior DeepInfer configuration when `--deepinfer-indexer` remains enabled.

The compact columns preserve the original candidate order and `torch.topk`
tie behavior. A Triton epilogue maps selected slots back to logical keys; the
existing SGLang request and pool-slot writeback contract is reused. Nextron
HyScale informed the bounded-query approach, but its sampled thresholds,
tie rule and score arithmetic are not copied.

Run `python3 benchmarks/v41_prefill/validate_compact.py --output result.json`
inside the pinned image before full-model paired validation. It compares every
compact score and selected key against the existing fused dense/gather path,
including real request planning/writeback, partial and invalid blocks, empty
candidates, signed weights, BF16 ties, noncontiguous row strides and query tiles.
