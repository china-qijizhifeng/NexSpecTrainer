# DeepInfer Hopper BF16 indexer port

Status: completed; GPU micro and full16k/64k validation passed. Owned jobs deleted and absence verified.

Actual run: node044, 8×H10080GB HBM3. Fixed image/model and full continuous16k/64k capture; no hashes.

Adapted upstream PR92 `e205d42e4b28185e50ce30f6b7204cd01e59974c` keys-major Gluon kernel. Retained persistent visible-tile scheduling, TMA key prefetch, explicit asynchronous WGMMA and packed BF16 epilogues. Changed operands from FP8 to existing BF16, query block32→16, stages3→2 and output BF16→FP32 dense layout. Candidate consumers retain the existing gathered BF16 scorer. No FP8 reencoding or candidate/top-k semantics changes.

## Micro results

| Rows × keys | Visibility | Current ms | Port ms | Speedup | Top-k exact |
|---|---|---:|---:|---:|---|
| 1024 × 8192 | causal ramp | 0.2091 | 0.1687 | 1.24× | True |
| 1024 × 8192 | late | 0.3377 | 0.2733 | 1.24× | True |
| 512 × 16384 | causal ramp | 0.2054 | 0.1635 | 1.26× | True |
| 512 × 16384 | late | 0.3512 | 0.2671 | 1.31× | True |
| 512 × 32768 | causal ramp | 0.3906 | 0.2834 | 1.38× | True |
| 512 × 32768 | late | 0.6893 | 0.4868 | 1.42× | True |
| 256 × 65536 | causal ramp | 0.3895 | 0.2882 | 1.35× | True |
| 256 × 65536 | late | 0.6981 | 0.4851 | 1.44× | True |

All12 cases passed. Small shapes regress due to launch/descriptor overhead; this remains an opt-in long-prefill experiment. Signed weights, arbitrary BF16, causal zero length and partial tiles covered.

Scores are not all bitwise identical: maximum relativeL2 across tested FP4-valued inputs 1.05e-6, all top-k sets identical. First run stopped at an exact-score assertion; CPUFP64 diagnosis isolated2 BF16-boundary differences among7,864,832 visible scores to sequential FP32 head accumulation. User accepted continuing. Acceptance retained strict synthetic top-k and original full-output numerical bounds (relativeL2<=.005, cosine>=.99998, exact token/mask).

PTX confirms 256 async WGMMA instructions,32 wait-group instructions,35 TMA load instructions and128 packed BF16 multiply instructions per emitted variant. This is an actual kernel adaptation.

## Reproduction

`--deepinfer-indexer` selects the new scoring path in `benchmarks/v41_prefill/benchmark.py`; `--fused-indexer` selects the previous optimized baseline. Keep codec8 and pipeline-depth2 for paired capture. All working changes are in the isolated checkout; no commit/push.

Micro timing includes the scorer wrapper/allocation/launch costs, averaged over5 CUDA-event repetitions after warmup. Full capture measures separate unprofiled requests and includes wire decode, validation, encoding/fsync and cleanup. Full-model profile is2 requests, max rank CUDA time per request.

## Full model paired results

Same node044 / same server process / same two real trajectory windows / codec8 / pipeline-depth2.
Four unprofiled requests per phase (two traces × two repeats), two independent diagnostic profile requests.
This differs from the previous pipeline report's eight requests; compare only the paired rows here.

| Continuous input | Forward current → port | Forward latency reduction | Capture current → port | Capture change |
|---|---:|---:|---:|---:|
|16k|668.390 →663.440 ms|0.74%|9872.84 →9464.40 tok/s|−4.14%|
|64k|3212.056 →3133.363 ms|2.45%|9491.35 →9637.75 tok/s|+1.54%|

64k indexer full-topk methods295.389→250.203ms; candidate-publisher262.321→232.356ms.
Combined557.709→482.559ms (−13.48%). Candidate consumers606.403→604.344ms, effectively unchanged.
Nested group timings cannot be added to attention_total or forward. MoE558.692→558.984ms, unchanged.
Peak allocated memory16k47.235→47.235GiB;64k67.367→67.367GiB. No model-memory saving claim.

Complete aux37/38/39+final BF16, input_ids and loss_mask are **bitwise identical** for all8 optimized output files against paired baseline. All16 baseline/port files passed native offline reader validation. 256 live audits (8ranks×8indexers×2lengths×2backends), up to32query rows each, have identical top-k sets. Intermediate score roundoff remains; finite test cases do not establish equivalence for every input.

The 16k capture regression means this is not a universal E2E improvement. The64k1.54% capture increase is a small single-pair result, not a sustained throughput guarantee. Keep the port opt-in; use full-model measurements rather than micro speedup when deciding deployment.

## Follow-up priority

Candidate consumers still cost about604ms at64k. Next experiment should remove their dense[rows,keys] score materialization using a compact candidate-score/top-k contract, or adapt DeepInfer's candidate path. It requires its own exact candidate mapping/tie-handling and full-output validation. This experiment did not implement it.

## Evidence and cleanup

- `comparison.json`, `evidence/parity16k.json`, `evidence/parity64k.json`, `evidence/indexer-audit-rank*.jsonl`.
- `evidence/micro-validation-r2.json`, `evidence/rounding-diagnosis.json`, `evidence/compiled-instructions.json`.
- `source-direct-comparison.json`: final kernel, adapter, benchmark and validation scripts equal executed copies by direct bytes, no hashes.
- Black, RuffF/I, compileall and git diff --check passed. No commit/push/main modifications.
- Final GPU job`job-8d1b68f1288c-20261103163718` / alias`nexspec-v41-prefill-deepinfer-20261103-r6`: Completed, deleted, queryempty. Owned server and Mooncake processgroups terminated.
- Earlier pending jobsr1–r5 canceled; auxiliary node043 image-import job completed/deleted. The imported pinned image remains as cache; no unrelated images/jobs modified. Final model execution stayed on originalnode044.
- Full evidence and activation files remain in `/gpfs/users/linlizhi/nexspec-v41-prefill-deepinfer-20261103`; small receipts mirrored under `evidence/`.
