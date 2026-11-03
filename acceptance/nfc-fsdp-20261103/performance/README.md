# Real dSpark/MTP checkpoint performance

The input is the actual step-2 DeNovo feature-training checkpoint, with
14,225,362,530 draft parameters. It reconstructs the native draft architecture,
8-rank FSDP1 ownership, BF16 model shards, CPU FP32 masters and Adam moments.
Frozen V4.1 target weights are excluded. Logical training tensors total
199,164,951,024 bytes across the eight ranks; native checkpoint files occupy
about 199.17 GB. This benchmark does not run the full target backbone or a
training forward/backward.

## CPU tile clone elimination: 8-H100 A/B awaiting resources

The current NFC dependency is `be06ae228600d8c2eed3c6a15c740163392a8587`.
In the deployed Core `filesystem_async.py:165–166`, `prepare_write_data()`
clones each CPU tensor whose storage extent differs from its logical bytes.
The optimizer's CPU shards are views of larger allocations. This copied the
payload once before NFC copied it again into staging, even with a cached plan.
`shard_tiles()` now creates an exact-length storage alias for each contiguous
CPU tile, retaining the source allocation and its explicit device. Special
logical/noncontiguous views keep the previous path. The capture fence remains
mandatory; this change removes the intermediate clone, not the snapshot copy.

CPU regression: NFC 522 passed / 43 skipped; both default CPU and meta device
contexts verify exact extent, pointer identity and lifetime after owner Tensor
GC. The independent dependency review also passed actual NativeState plus
BF16Optimizer capture/mutate/restore checks. See `../REVIEW.md`.

`cpuview-benchmark.py` alternates cold baseline, baseline1, optimized1,
optimized2, baseline2, optimized3 in one process with the same backend,
runtime, cached plan and 32 GiB/rank pool, four writers/rank. A scoped clone
hook records CPU bytes and time inside Core preparation; each optimized round
requires zero such clone bytes. Every round mutates live state after collective
source release, overlaps synthetic CUDA work with the writer, then restores
the complete state fingerprint. Original input mappings are private/read-only
and generated payload deletion is restricted to the isolated workspace.

The first attempt passed 10 native Core/CUDA regression tests on `4592a54`
before a higher-priority job preempted it at 19:27:52. It produced no 199 GB
measurements and its exit file alone is not a success receipt. Its events,
worker log and exact cleanup are preserved. A scheduling-only second attempt
was removed after inspecting unschedulability. The current third attempt
`nexspec-nfc-dspark-cpuview-20261103-r3` / `job-46b0d32c5ef9-20261103194134`
uses the final dependency source, remains queued for eight H100s and has no
performance results yet. The queue is WeaverRSI / proj-32; resources remain
64 CPU / 768 GiB, with no exclusive-node or reservation bypass.

Repository formatting only reordered standard-library imports in the archived
driver; its executable AST otherwise matches the uploaded driver, as recorded
in `cpuview-source-equivalence.json`.

The archived run/YAML and source hashes are reproduction inputs, not passing
receipts. No real-size speedup, final-source GPU acceptance or zero-pause
claim is made while the A/B test is pending. Earlier results below retain
their original source identity and timing definitions.

## Previous asynchronous save with cached plans

The final production source is `b311a03e9ec54e872bd70e1e7b99779c08272af5`.
The job ran on 8 H100s with 64 CPU and a 768 GiB host-memory limit, using
32 GiB/rank budgets and four writers/rank. The staging arena actually occupied
199,167,029,248 bytes in total (about 23.19 GiB/rank). All times are the maximum
rank measurement, in seconds.

| Save | Caller pause to collective source release | Complete after source mutation and GPU work | Core phase ending at planned |
| --- | ---: | ---: | ---: |
| First | 36.34 | 49.62 | 20.07 |
| Warm 1 | 12.40 | 23.83 | 9.27 |
| Warm 2 | 11.31 | 22.23 | 8.66 |
| Warm 3 | 10.47 | 21.11 | 6.55 |

The caller now returns before native persistence. Each save was collectively
not ready at the immediate nonblocking poll. The test then mutated every
nonempty live model/master/Adam tensor, LR, scheduler and all RNG sources,
and ran synthetic CUDA matrix work before explicitly draining. At collective
capture return, local writes remained pending on 7/7/8/8 ranks respectively;
a fast rank may finish while it waits for other ranks' captures. All four
checkpoints were restored and matched the original complete state fingerprint
on all eight ranks. A changed fingerprint was required before each restore.
The complete column includes that live mutation and synthetic work, so it is
an end-to-end completion measurement, not isolated disk latency or training
throughput. The large benchmark does not run draft/target forward/backward.

All warm ranks report `validated_cache_reuse=true` and `pool_reused=true`.
Core's cached structure/metadata planning is enabled; remaining Core work up
to the `planned` mark still costs 6.5–9.3 seconds at the slowest rank. The phase includes `prepare_write_data()` and its CPU view clones; it is
not isolated planner computation. The newer exact-storage alias dependency
removes those redundant clones without changing source capture ownership. First-save timing includes
runtime/planner/pool setup. The earlier synchronous measurements below ran
separately on a shared filesystem and must not be used to infer a controlled
speedup for this job.

The same final source first passed 40 small-model rank receipts: three
parallel modes with four full restores and the next update bitwise equal,
public Trainer restart equivalence, and rank-0 write-delay/failure injection.
In the delay test, every rank advanced the real optimizer while rank 0's disk
worker was held; partial readiness did not publish best/latest. Failure
injection reached every rank and preserved the previous same-step version.

Final worker exit was 0. The 32 large-state save/restore records and all eight
pass receipts are included in `async-results.json` / `async-passed.json`;
small-model receipts are in `async-small-model-results.json`. Each generated
199 GB payload was removed; the original nine input files kept their sizes
and modification times. The exact completed job was removed and read back
absent. An earlier scheduling-only attempt was cancelled after its node list
could not fit the request; its cleanup receipt is included and it contributes
no timings. The final attempt added a normal available GPU node and checked
H100 identity at startup, without changing the resource budget.

`async-source-sha256.json` verifies all relevant trainer/config/driver source
hashes against the final uploaded source. The archived benchmark is AST
identical to the tested benchmark after import-spacing formatting. CPU
lifecycle/config/resume/export checks: 249 passed, 7 skipped, 470 subtests.

## Historical synchronous pool comparison

8 H100, 64 CPU, 768 GiB host-memory limit; pinned Torch 2.10/mcore image in
`pool-job.yaml`. Each native configuration uses four writers per rank.
All times below are the maximum across the eight ranks, in seconds.

| Configuration | First save | Warm save 1 | Warm save 2 | Warm source release |
| --- | ---: | ---: | ---: | --- |
| 2 GiB/rank budget | 41.22 | 14.80 | 16.38 | 13.17 / 13.54 |
| 24 GiB/rank budget | 23.38 | 14.96 | 15.11 | 6.22 / 6.45 |
| Legacy full-state gather + save + explicit fsync | 64.02 | 79.10 | — | — |

In this historical baseline the native caller waited through full recoverability, including
durability, metadata finalization and pointer publication. Its caller pause
therefore includes background I/O. The two warm-save medians are 15.59 and
15.04 seconds. The legacy median is 71.56 seconds in this run, including its
real FSDP full-draft gather. These are two observations per native pool and
two legacy observations on a shared filesystem, not a statistical throughput
guarantee. "First" means a cold runtime/pool, not cold filesystem caches.

Increasing the pool changes the source-release boundary substantially. The
2 GiB layout recycles bounded slots while writing; the 24 GiB budget fits
the approximately 24.90 GB local payload and captures the complete rank
before the writer drains it. Actual staging allocation is approximately
2 GiB or 23.19 GiB per rank, respectively. A larger budget alone does not
remove that version's wait for recoverability. The current integration uses
asynchronous capture-return and cached planning; its results are recorded separately.

For both pool sizes, the last warm checkpoint was restored after mutating
every nonempty model/master/Adam tensor, optimizer LR, scheduler position and
all four RNG sources. The changed-state fingerprint was required to differ
before restore; every restored state fingerprint then matched the original,
on all eight ranks. All 64 save records and 16 full-state restore checks
passed; worker exit was 0. Generated checkpoint payloads were deleted after
each measurement, and the original nine input files retained their sizes and
modification times. The exact completed GPU job was deleted and read back
absent.

## Historical synchronous writer comparison

The same 199 GB state, 32 GiB/rank budget and eight H100s were used to vary
writers per rank. These runs still used the synchronous caller and uncached
planning. All times are the maximum rank time in seconds.

| Writers/rank | First save | Warm 1 | Warm 2 |
| --- | ---: | ---: | ---: |
| 4 | 48.18 | 15.11 | 14.99 |
| 8 | 21.72 | 14.19 | 14.75 |
| 16 | 20.56 | 14.71 | 14.29 |

The small differences between warm saves do not justify increasing the
production default above four writers/rank. The first configuration also pays
process-wide initialization, so cold values are not a controlled concurrency
comparison. All 72 save records and 24 full-state restore checks passed;
`writer-results.json` contains the rank receipts. The worker exited 0 and the
exact completed job was removed and read back absent.

## Reproduction and limitations

`pool-benchmark.py`, `pool-run.sh`, `pool-job.yaml`, runtime identity,
source fingerprints, rank receipts and summary are included. The config JSON
is alongside the benchmark. Keep the original input checkpoint read-only and
use a fresh isolated output directory. The source-equivalence receipt in the
parent directory relates the GPU-tested trainer source to the PR source.

The first attempt lacked a config file and failed before measurement. A later
attempt hit GPFS quota after writing about 182 GB; it is excluded from these
timings. The partial payload was removed, quota was inspected, and a real
220 GB preallocation succeeded before resuming measurement. One attempt was
cancelled while waiting for resources. No failed or queued attempt is counted
as a performance result. All exact owned attempts were cleaned up.

The earlier generic snapshot timings belong to a different implementation and
are not included in this comparison. This test measures checkpoint state
serialization and recovery at real dSpark scale; the small-model acceptance
in the parent directory separately proves the next optimizer step and public
Trainer restart equivalence.
