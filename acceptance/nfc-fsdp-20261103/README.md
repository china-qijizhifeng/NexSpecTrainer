# FSDP / BF16Optimizer native NFC acceptance

NexSpecTrainer base: `01ff8af94a447041e9377590ee3db5754b083523`.
NFC dependency: PR [#49](https://github.com/china-qijizhifeng/NexFlashCheckpoint/pull/49),
`79d838fc19ba1f179fc6c1dc36147e5183d82ac8` (not merged at validation time).
Runtime and source hashes are recorded alongside this receipt.

## Current asynchronous implementation

`save_checkpoint()` now returns after collective source capture, with disk
persistence continuing while the optimizer advances. Native saves always use
cached Core metadata plans; the V4.1 recipes budget 32 GiB/rank. A training-lane
nonblocking collective poll finalizes ready saves, and final/close explicitly
drain them before reporting success. Completion markers, best/latest and
rotation follow successful persistence.

CPU integration/config/export/resume and lifecycle checks: **249 passed,
7 skipped**, 470 subtests. Independent review rounds 4–5 verified source-release
safety, timeout retry/error propagation, cache invalidation, best rollback,
installation/retention directory sync ordering and umask preservation. The
asynchronous GPU and real-size performance receipts are recorded separately
under `performance/`.

## Historical synchronous integration results

- CPU integration/config/export/resume: **230 passed, 7 skipped**, 465 subtests.
- Package architecture: **17 passed**. One existing main failure concerning
  `scripts/prepare_server_hidden_states.py` importing training code is recorded
  separately; both relevant files are byte-identical to the base revision.
- GPU: **8 × NVIDIA H100 80GB**, all 32 rank result records passed, process exit 0.
- Job `nexspec-nfc-native-20261103-r5`, actual project `WeaverRSI`, queue `proj-32`:
  Completed, then precisely deleted; read-back is `[]`.
- Formatting, import, secret and standard repository pre-commit checks passed.

| Case | Evidence |
| --- | --- |
| FULL_SHARD + CPU master offload | 8 ranks × 4 checkpoint restores; complete state and next step bitwise equal |
| SHARD_GRAD_OP + GPU masters | 8 ranks × 4 checkpoint restores; complete state and next step bitwise equal |
| DDP + CPU master offload | 8 ranks × 4 checkpoint restores after two reducer warmup updates; complete state and next step bitwise equal |
| Fresh wrapped model and master placement switch | Saved FP32 masters preserved; next forward loss equal |
| Native weights-only CPU export | Every exported tensor slice equals its owning rank's model tensor |
| Public `Trainer.fit()` restart | On all 8 ranks: continuous 3 steps equals 1 step + checkpoint + resumed training to step 3, including model/masters/Adam/scheduler/RNG |
| Completion and online resume metadata | CPU regressions reject failed saves, preserve tuple contracts, read online step without a collective, and reject missing export keys |

FSDP cases include step-0 lazy Adam state, empty/uneven original-parameter
shards, nested block wrappers, a dedicated FP32 wrapper, replicated integer and
empty buffers, and frozen ignored target weights. Every checkpoint uses the
native pipeline. The driver makes FULL_STATE_DICT calls fatal.

DDP rebuilds its gradient reduction buckets after the first backward. Comparing
its cold first update with a replay on a warmed reducer produced rounding
differences despite bitwise-identical restored inputs/state. The DDP test now
performs two warmup updates and asserts that bucket rebuilding has completed
before the four checkpoint comparisons. This is not a claim of bitwise cold
DDP reducer migration.

## Scope and reproducibility

The current driver adds disk-delay/error injection and asynchronous source
isolation to the earlier cases. Run `tests/gpu/nfc_fsdp_acceptance.py` with 8
torchrun ranks in the image pinned
in `gpu-job.yaml`. `gpu-run.sh` records runtime versions, Core API/source hash,
GPU identity and the worker exit code. `gpu-results.json` holds all rank records.
The accepted GPU source is recorded in `source-sha256.json`; the final
`checkpoint.py` additionally improves docstrings/comments without changing its
executable statements. Full worker logs remain under the task's GPFS evidence
root and the local `dspark-denovo-20261103/evidence/nfc-integration` directory.

Earlier attempts: r1 identified the FSDP first-forward buffer dtype issue; r2
and r4 were cancelled while waiting for occupied pinned nodes; r3 passed both
FSDP cases and exposed the DDP reducer warmup distinction; r5 completed all
cases. All five exact owned aliases were removed and read back absent. No
unrelated jobs were stopped.

This small-model test proves correctness. Real dSpark/MTP checkpoint
throughput (approximately 14.23B draft parameters with masters and Adam, about
199 GB) is a separate measurement; frozen V4.1 target weights are excluded.
Complete native GPU checkpointing on Torch 2.13 has not been
validated; use the pinned Torch 2.10/mcore runtime. The original 199 GB
checkpoint remains untouched. Formal 40-GPU training has not been launched.
