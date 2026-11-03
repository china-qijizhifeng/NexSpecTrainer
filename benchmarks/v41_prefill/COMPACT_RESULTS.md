# Compact candidate indexer results

The final compact candidate path improved full continuous prefill on the tested
16k and 64k inputs. Candidate consumers write only candidate-order scores, avoiding
the dense all-key score allocation/fill/scatter and subsequent gather. The existing
BF16 arithmetic, candidate ordering, torch.topk and logical-key/pool-slot contracts
are retained. Full-source and candidate-publisher scorers still use the prior
DeepInfer path and still materialize dense scores.

## Final paired measurements

8×H100 80GB HBM3, node120, TP8/EP8, CPU Engram and native FlashInfer MXFP4 MoE.
Same pinned image/model as DEEPINFER_RESULTS.md. Baseline is the prior DeepInfer
optimization from main c179ad2a. Both configurations ran in the same server process
on the same two real token/mask windows, codec8, pipeline-depth2. Each phase uses
four unprofiled capture requests and two separate forward profile requests.
No prefill/attention chunking, bounded SWA replay, CUDA graphs or bounded MoE.

| Input | Candidate consumers ms | Forward ms | Forward reduction | Capture tok/s | Capture gain |
|---|---:|---:|---:|---:|---:|
| 16k | 89.200 → 69.300 | 667.320 → 644.866 | 3.36% | 9986.57 → 10126.69 | 1.40% |
| 64k | 600.853 → 453.571 | 3129.062 → 2976.198 | 4.89% | 9668.58 → 9888.57 | 2.28% |

Forward is the mean of two max-rank CUDA profile times. Candidate timings are
rank0 nested method timings; do not add them to attention or forward totals.
Capture includes pipeline fill/drain, transfer, validation, encode/fsync and cleanup.
Small single-pair throughput gains are not a sustained production guarantee.

Peak allocated memory: 16k 47.235→47.360 GiB; 64k 67.492→67.492 GiB. There is no
whole-model memory-saving claim. For the 64k candidate scorer, the FP32 score
allocation changes from 256×65536 (64 MiB) to 256×16384 (16 MiB). At 16k all keys
fit in the candidate list, so the score tensor remains 64 MiB; fill/scatter/gather
work is still removed. Query tiling bounds each compact score tensor to 64 MiB,
not total live selection scratch or peak model memory.

## Correctness and independent review

All eight final compact output files are bitwise equal to their paired baseline:
complete aux37/38/39 and final BF16 fields, input IDs and masks. All 16 baseline/
compact files passed the native offline training reader. All 256 target live
indexer audits have exact top-k sets. Another 64 startup audits at key lengths1/3
also passed. Two independent review rounds found no blocking issues.

GPU microvalidation passed real prefill request planning/consume/writeback and
14 exact score/top-k cases. Cases include FP4-valued and arbitrary BF16 operands,
signed weights, zero visibility, ties, empty/invalid/partial candidate blocks,
k larger than candidate width, noncontiguous row strides and query chunks19.
The compact kernel preserves the previous fused scorer's arithmetic; prior
DeepInfer source/publisher rounding behavior remains as documented separately.
Finite tests do not establish equivalence for every possible input.

Tile128 was fastest among64/128/256. Scorer+selection micro timings for256rows:
16k 0.49990→0.37103ms (1.347×),64k 0.52546→0.37784ms (1.391×). Includes allocation
and launch costs. These are not full-model timings.

## Query tile choice and HyScale

The first16MiB cap increased16k query chunk count fourfold and regressed full
forward665.293→674.704ms despite the microkernel win. A controlled1024-row probe
found4×256 compact chunks took1.48756ms versus1.25947ms for1×1024, with exact top-k.
The final64MiB cap preserves the original16k row batching. At64k the existing row
bound still gives256rows and16MiB. The full pairs above were rerun after this change.
The first64k pair also improved forward3150.396→2976.786ms; use final rows above.

Reference: Nextron9763a325081d4ff9c2e0df880a339aae33886d88,
primitives/dsa_hyscale/{README.md,selection.cu,selection.cuh,VALIDATION.md}.
This implementation borrows bounded query batching and compact storage concepts;
no Nextron source was copied. HyScale's sampled threshold/filter/exact-recovery
selection was not ported: its smaller-logical-key tie rule and scoring arithmetic
differ from this V4.1 path. Existing torch selection semantics are preserved.

## Reproduction and evidence

The benchmark now enables `--deepinfer-indexer --compact-candidates` by default
with the pinned serving image. Use `--no-compact-candidates` to reproduce this
report's prior DeepInfer baseline; add `--no-deepinfer-indexer` for the original
scorer. The experiment recorded here supplied both positive flags explicitly. Run
`python3 benchmarks/v41_prefill/validate_compact.py --output result.json`
for the kernel and integration checks before paired full-model validation.

All capture/input/output/source checks use direct bytes or tensors, without hashes.
Model/image identities remain pinned. Black, Ruff F/I, compileall and diffcheck pass.

Local experiment receipts:
`/Users/shom/Downloads/speculative-decoding-training-research/v41/prefill-compact-20261103`
Remote full outputs and activation scripts:
`/gpfs/users/linlizhi/nexspec-v41-prefill-compact-20261103`

Key receipts: comparison.json; evidence/parity16k-r2.json; evidence/parity64k-r2.json;
evidence/completion.json; evidence/micro-validation.json; evidence/query-tile-probe.json;
source-direct-comparison.json; compact-review.json and review/CPU contract evidence.
Phase1 source and records are retained. Final source files match executed bytes.

The initial cleanup assertion expected256 audits but observed320 due to64 startup
warmup audits after restart. The corrected verifier checks the exact target
rank/length/backend counts separately and requires all320 top-k results to pass.
No numerical failure was hidden or retried.

Owned job `job-dfa58ffdf82f-20261103172427`, alias
`nexspec-v41-prefill-compact-20261103-r3`, reached Completed. Owned server/Mooncake
process groups terminated; job deleted and absence verified. Earlier pending r1/r2
jobs were also deleted. Unrelated jobs were untouched; pinned imported image remains
as node cache. Cleanup receipts: job-terminal.json, job-delete.json, job-absent.json.
