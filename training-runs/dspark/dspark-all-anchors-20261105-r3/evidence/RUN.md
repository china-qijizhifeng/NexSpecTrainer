# DeNovoSWE V4.1 dSpark — all legal anchors

User scope: improve issue #13 anchor coverage and restart the same 1k-scale training. Confirmed exact cohort is the previous 1,326 training traces and 17 disjoint validation instances. User explicitly replaced the proposed 64-anchor policy with every eligible anchor.

Source: `/Users/shom/Downloads/NexSpecTrainer-anchor-coverage-20261105`, branch `codex/anchor-coverage-20261105`, base `aa15f69740e2d6fef486b47f5f5a5663b4d53cd7`. Exact uploaded working-tree files are pinned in `source-provenance.json`; the run does not require a remote branch or PR. No PR or issue comment was posted.

Remote root: `/gpfs/users/linlizhi/dspark-all-anchors-20261105-r3`. Job alias `denovo-all-anchors-20261105-r3`, WeaverRSI / proj-32, 8 H100, 64 CPU, 640 GiB host memory. Project/queue queried before submission. Fixed image is the same validated Weaver v1.14.92 digest as the previous run. Maximum runtime 7 days; no job restart or automatic resubmission.

## Fixed experiment

- Exact previous train/validation feature paths, official target snapshot, full trajectory prefill context and 4096-token windows. No recapture or mutation of existing caches.
- Official MTP initialization; no warm-start from the previous trained checkpoint.
- 15104 training windows (15053 originals + 51 explicitly recorded dataset padding duplicates), 136 validation windows; seed 42, 1 epoch, 118 optimizer updates.
- Batch 1 × 8 ranks × accumulation 16; LR 1e-5, warmup ratio .03 (3 integer optimizer steps), cosine to zero, clipping 1.0, same AdamW defaults.
- K=5; CE .1 + probability L1 .9 + confidence BCE 1.0; position gamma 4; objective chunk 8. Save/evaluate every 25 updates and save final step118.
- Changed `anchor_sampling: all`; `num_anchors: 32` remains solely for compatibility and is ignored in this mode. `anchor_forward_chunk_blocks: 32` controls the temporary forward workload, not the number of selected anchors.

## Coverage and data contract

Every position whose anchor and first target masks are both valid is enumerated once, in sequence order. Each block stops at the first masked target or sequence end. Rank widths are padded to the global maximum at each forward so FSDP executes the same chunk/collective sequence. Padding contributes neither objective nor coverage. A globally empty objective fails; empty individual ranks still participate.

The whole draft+objective chunk is activation-checkpointed so backward recomputes it. This avoids retaining all gathered native context windows at once. Existing objective weights and global per-microbatch normalization are preserved. The previous implementation normalizes each distributed microbatch before accumulation; this run does not change that convention.

The exported DeNovo parquet has binary integer message `loss_mask`, not NexDataQC `use_loss` fields. `audit_data.py` explicitly selects that policy and re-renders all 1343 complete traces using the same pinned official encoder/tokenizer. Every token SHA and mask SHA matches the old capture manifest; no missing cleaning metadata is invented. `v41_supervision.py` supports explicit strict-boolean message/content-part/tool-call `use_loss` and reasoning-only `use_thinking`, with malformed flags rejected. Tool outputs remain full context, including user/merged results, and are not supervised. Cleaning annotation tests run against the official snapshot.

Excluding the 51 dataset duplicate windows: 39,961,204 legal training anchors versus 481,225 sampled anchors in the previous budget, 199,316,732 target exposures, 45,605 eligible tool-call/window intersections, all covered. 1,523 call/window intersections cannot be covered under the existing mask/anchor rules (including overlap prefixes); these are separately excluded from the coverage denominator. Counts are exposure counts, not unique tokens or unique calls. Tool-call IDs are derived for the data audit, not claimed to exist in legacy cache schema. Exhaustive sampling needs only the verified loss mask.

## Validation and runtime evidence

CPU checks: 90 passed, 441 subtests; official template mask tests included. Full vs recomputed chunk loss/gradients match for chunk widths 1/7/32 and uneven/empty rows. Config and recipe documentation checks pass.

GPU job gates: actual-image preflight, 8-rank FULL_SHARD exhaustive-anchor test including an empty rank and uneven candidate counts, then native V4.1 kernel optimizer updates. Only after every gate passes does the public training CLI start. This document does not imply those gates have passed; see `gpu-regression.complete.json` and rank receipts.

Training wrapper checks every consumed feature ID/mask hash and compares selected anchor and per-position exposure counts to the independent full-data audit. Runtime files: `training/worker.log`, `micro-progress-rank*.json`, `steps-rank*.jsonl`, `live-progress-rank*.json`, and final `complete-rank*.json`. Each optimizer record includes actual LR, global grad norm, clipping flag, exposures and peak CUDA allocation. Frozen target embedding/head are excluded from optimizer and checked before/after.

Full NFC checkpoint path is `training/denovo-v41-all-anchors-step118`. Final publication and fresh restoration remain distinct checks. Do not treat job submission, Running, or old baseline restore receipts as proof of this run's completion. Real serving acceptance improvements are not established by training loss or static coverage.

Additional provider suite: 17 passed / 63 subtests. Two pre-existing catalog-count tests fail because they still expect seven algorithms before native deepseek_v41 was registered; the same exact two failures were reproduced in the unchanged aa15f697 baseline checkout. These are recorded, not counted as passing or attributed to the anchor change.

## Current attempt R3 and prerequisite repairs

Current job: `job-0155ac7d3123-20261105122251`, alias `denovo-all-anchors-20261105-r3`. Read-only terminal/cleanup monitor PID52727 records job state and releases only this exact job after its terminal state; it does not resubmit or message anyone.

R1 passed the unequal-rank test but stopped at the native kernel gate before training: later chunks grew the RoPE cache, invalidating earlier activation recomputation metadata. Native forward now always uses an exact-length view. A CPU regression covers interleaved lengths 12/13/7/30; native GPU optimizer tests subsequently passed.

R2 passed all gates and the first full-size formal forward on eight ranks. Exact IDs/masks, every selected anchor and every target-exposure count matched the independent dataset audit. Its first backward then exceeded memory in the monolithic expert-weight gradient; zero formal optimizer updates occurred. R1 and R2 were both released and their evidence is retained in their own directories.

R3 computes the same complete token reduction in groups of eight independent experts, retaining a single reusable FP32 workspace and directly copying into the final BF16 gradient. Anchor coverage, model objective, optimizer, gradient accumulation and batch remain unchanged. CPU repair tests: 11 passed (including the seven existing full-anchor/padding cases and four new cache/gradient cases).

Actual H100 native-size old/new comparison passed, with BF16 gradients bitwise equal for both expert matrices. Extra peak allocation fell from 18,119,393,280 to 6,794,772,992 bytes for [128,4608,5120], and from 9,059,696,640 to 3,397,386,752 bytes for [128,5120,2304]. This is gradient-kernel allocation, not whole-training peak memory. See native-wgrad-acceptance.json. Formal training progress and optimizer completion must be read from runtime receipts; these kernel checks do not count as formal updates.

R3 terminal: failed on first backward with CUDA OOM allocating a 720 MiB workspace; 67.83 GiB allocated and 9.00 GiB reserved-but-unallocated. Zero formal optimizer updates. Monitor retained evidence, deleted the exact job and confirmed absence.
