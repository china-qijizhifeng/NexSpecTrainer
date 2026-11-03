# Real dSpark/MTP checkpoint performance

The input is the actual step-2 DeNovo feature-training checkpoint, with
14,225,362,530 draft parameters. It reconstructs the native draft architecture,
8-rank FSDP1 ownership, BF16 model shards, CPU FP32 masters and Adam moments.
Frozen V4.1 target weights are excluded. Logical training tensors total
199,164,951,024 bytes across the eight ranks; native checkpoint files occupy
about 199.17 GB. This benchmark does not run the full target backbone or a
training forward/backward.

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
