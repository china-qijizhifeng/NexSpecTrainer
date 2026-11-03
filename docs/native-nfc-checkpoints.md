# Native NFC checkpoints for FSDP

Set `training.nfc_checkpoint: {}` to save resumable training state through
NexFlashCheckpoint's native sharded pipeline. Both DeepSeek V4.1 native example
recipes enable it. Missing dependencies or unsupported layouts fail explicitly;
this path never falls back to a CPU snapshot or the independent `torch_dcp`
backend. Existing recipes without this option retain their legacy checkpoint
format, and old `.pt` checkpoints remain readable.

```yaml
training:
  nfc_checkpoint:
    max_shared_memory_bytes: 2147483648
    pipeline_max_inflight_bytes: 2147483648
    staging_tile_bytes: 134217728
    writer_thread_count: 4
    load_worker_count: 16
```

Both byte limits are **per rank**. NFC streams tensor tiles through a pool
bounded by their minimum; the pool need not hold a rank's complete checkpoint.
Eight ranks at the defaults use at most 16 GiB of staging memory in total,
plus runtime overhead. There is no `pipeline: false` setting.

The V4.1 recipes use a 32 GiB/rank budget to fit the approximately 23.19 GiB
rank payload and capture the entire rank before the writers drain it. Actual
allocation follows payload size. Eight ranks need about 185.5 GiB of pinned
staging memory, in addition to training/optimizer memory; reserve host RAM
accordingly. The generic 2 GiB default still streams through bounded slots,
but then source release depends on writer progress.

The controlled 14.23B/199 GB dSpark A/B measured warm caller pauses of
14.36 / 10.45 seconds before the alias fix and 3.02 / 2.54 seconds after it,
in one process with identical runtime/pool/cache. The two-observation medians
fell from 12.41 to 2.78 seconds (77.6% less pause). Core CPU clones fell from
163.07 GB across ranks to zero; `prepare_write_data()` fell to about 2 ms.
The remaining pause primarily captures roughly 24.9 GB/rank into the immutable
pool (slowest capture 2.92 / 2.44 seconds), with zero slot waiting. Persistence
remained asynchronous on all ranks. Complete time including live mutation and
synthetic GPU work was 11.75 / 12.47 seconds. These are checkpoint measurements,
not complete training throughput.

Five rounds restored the complete state on all ranks (40 checks). The final
third optimized save was preempted before restore and is excluded from those
counts and the comparison; whole-job success is not claimed. Final-source
Core/CUDA regression tests and small-model Trainer receipts passed before the
large measurements. The previous 6.5–9.3-second phase ending at `planned`
included CPU view cloning and was not isolated planner computation.

Historical synchronous pool and writer comparisons are recorded separately;
increasing from four to eight or sixteen writers showed only small warm-save
differences. See exact rank receipts, parameters and limits in
[`acceptance/nfc-fsdp-20261103/performance/`](../acceptance/nfc-fsdp-20261103/performance/).

## Dependencies and verified runtime

This integration requires [NexFlashCheckpoint PR #49](https://github.com/china-qijizhifeng/NexFlashCheckpoint/pull/49),
revision `be06ae228600d8c2eed3c6a15c740163392a8587`, with its built Rust extension,
and a compatible Megatron Core **mcore async** implementation. The newer NVRx
async API is not interchangeable. Install NFC without its `checkpoint` extra
or use `--no-deps`; that extra pins Torch 2.9 and must not replace the trainer's
existing Torch installation.

GPU acceptance uses the pinned image
`harbor.local.clusters/bp/weaver-trainer-slim:20261016-6ec59d8d8079@sha256:b8a311e4c399369117c89164a1304d8110cfacb5038483c045d57a45dd5eea32`,
Torch `2.10.0a0+b558c986e8.nv25.11` and Core serialization source SHA-256
`5f745e742b63b4d4f4c406edda9c09605746bd52faa24c1b5fb4446676799a65`.
The repository's Torch 2.13 development environment is used for CPU contract
checks; a complete native save on Torch 2.13 has not passed. Independent CPU DCP
checks found an existing `Tensor.names` incompatibility in NFC serialization;
this integration remains limited to the pinned Torch 2.10/Core GPU runtime.

Contiguous CPU optimizer tiles use exact-length storage aliases. This avoids
Core cloning a view of a larger backing allocation before NFC captures the
same bytes into its staging pool. The aliases retain the original allocation;
training must still wait for collective `source_released` before mutation.
Noncontiguous, conjugate and negative views keep their existing behavior.

## State and ownership

CUDA training registers `cpu:gloo,cuda:nccl` on WORLD. CPU object collectives
(including Core's save-plan gather) use Gloo; CUDA FSDP tensor collectives keep
using NCCL. This avoids routing host checkpoint metadata through CUDA gather
and retains the default-group contract required by native Megatron.

The adapter supports FSDP1 with `use_orig_params=True`, the WORLD process group,
and `BF16Optimizer`. It reads FSDP's original-parameter shapes and local slice
offsets, including uneven and empty shards. Each model weight, FP32 master and
Adam moment becomes a native Core `ShardedTensor` with its actual flattened
global coordinates. Replicated buffers and DDP state have explicit replica IDs.
There is no model `FULL_STATE_DICT` call or rank-0 model gather.

FSDP's normal lazy initialization runs before collecting views, including its
first-forward buffer casting. Loading preserves saved runtime buffer dtypes.
Private FSDP metadata is an explicit compatibility boundary, covered by the
multi-GPU acceptance driver. A changed world size, parameter layout, dtype or
optimizer ordering is rejected; changing CPU master offload placement is
supported.

Small rank-local `ShardedObject` sidecars contain Adam step counters, optimizer
param groups, scheduler state, gradient clipping policy, and Python, NumPy,
Torch CPU and bound CUDA/NPU RNG state. Masters and Adam moments are never
packed into those object sidecars. The checkpoint also records the strategy,
provider/model provenance contracts, epoch/sample progress and global step.

## Lifecycle and export

Saving flushes the durable data acknowledgement first, then captures the
native shards. `save_checkpoint()` returns a `Checkpoint` with
`metadata["durability"] == "source_released"` once all ranks have safely
released their model/optimizer sources. Its final path is not yet a resume
target. Training can mutate those sources while NFC workers finish local
writes and fsync. Only one save is in flight; the next save drains it before
reusing the staging pool.

At optimizer boundaries the training thread collectively polls the ticket
with `timeout=0`. A rank that is still writing postpones finalization; any
rank's write error fails all ranks. Core finalizers, distributed collectives
and directory publication always run on the training thread. The runtime
always enables Core's save-plan cache and reuses its staging pool. NFC
compares shard signatures across ranks and invalidates cached plans on layout
changes, including lazy Adam initialization; new tensor values are bound on
every save. Changing step counters or RNG does not reuse old payload data.

A completed step directory contains:

- `nfc/`: native DCP tensor shards, rank objects and Core's small `common.pt`.
- `training_state.json`: completion identity and SHA-256 of `common.pt`.

The marker is published only after every rank's native save is recoverable.
Resume first reads the small typed common metadata with `weights_only=True`,
validates training contracts, then restores rank-local tensors after wrapping
the model. Tuple-valued provider contracts retain their types. Every rank
agrees on the common metadata checksum before loading.

Run-scoped latest/best links and retention change only after every rank's
write succeeds. A save uses a private directory, preserving the previous
complete checkpoint throughout capture and disk writing, including a same-step
replacement. Directory installation is synced before rollback/rotation can
remove older versions, and final pointer/retention changes are synced again.

Call `trainer.wait_for_checkpoint()` collectively when a saved path must be
consumed immediately. `Trainer.fit()` drains before reporting success or
publishing consumer completion; controller fit and runtime close also drain.
`last_checkpoint_step` tracks the most recently submitted capture and prevents
a duplicate final save; it is not a durability notification.

The existing export and weights-only warm-start readers recognize the native
format. A single CPU process materializes only the draft weights through DCP,
without reading optimizer tensor payloads or requiring the original world size.
Missing declared weight keys are fatal. SGLang export remains a separate
weights artifact; it does not replace the resumable training checkpoint.

## Acceptance

Run the self-contained small-model test in the verified runtime:

```sh
python -m torch.distributed.run --standalone --nproc_per_node=8 \
  tests/gpu/nfc_fsdp_acceptance.py --output /absolute/isolated/evidence
```

It exercises nested FSDP, FP32 modules, ignored frozen target weights, uneven
and empty shards, lazy Adam state, repeated pool reuse, FULL_SHARD,
SHARD_GRAD_OP, DDP, CPU masters, saved metadata contracts, fresh model restore,
placement changes and weights-only export. DDP is warmed for two updates so
its one-time reduction-bucket rebuild does not confound checkpoint comparison.
The FSDP cases include step-0 lazy Adam state. It compares the complete saved
state and the next step's loss/model/master/Adam/scheduler/RNG bitwise within
the same placement. A placement switch checks saved masters separately because
fused and unfused Adam kernels can round differently.

This is checkpoint correctness acceptance on a small model. Real dSpark/MTP
save throughput requires measuring the approximately 14.23B-parameter draft,
FP32 masters and Adam moments (about 199 GB). Frozen V4.1 target weights are
excluded. The planned 40-GPU training run has not started.
