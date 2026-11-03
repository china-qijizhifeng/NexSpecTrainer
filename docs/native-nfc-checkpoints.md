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

## Dependencies and verified runtime

This integration requires [NexFlashCheckpoint PR #49](https://github.com/china-qijizhifeng/NexFlashCheckpoint/pull/49),
revision `79d838fc19ba1f179fc6c1dc36147e5183d82ac8`, with its built Rust extension,
and a compatible Megatron Core **mcore async** implementation. The newer NVRx
async API is not interchangeable. Install NFC without its `checkpoint` extra
or use `--no-deps`; that extra pins Torch 2.9 and must not replace the trainer's
existing Torch installation.

GPU acceptance uses the pinned image
`harbor.local.clusters/bp/weaver-trainer-slim:20261016-6ec59d8d8079@sha256:b8a311e4c399369117c89164a1304d8110cfacb5038483c045d57a45dd5eea32`,
Torch `2.10.0a0+b558c986e8.nv25.11` and Core serialization source SHA-256
`5f745e742b63b4d4f4c406edda9c09605746bd52faa24c1b5fb4446676799a65`.
The repository's Torch 2.13 development environment is used for CPU contract
checks; a complete native GPU round trip on Torch 2.13 is not claimed.

## State and ownership

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

Saving flushes the durable data acknowledgement first. The caller waits for
NFC durability before advancing checkpoint state or publishing latest/best.
NFC overlaps tile capture with rank-local storage writes internally; this
initial integration does not overlap subsequent optimizer steps with saving.

A completed step directory contains:

- `nfc/`: native DCP tensor shards, rank objects and Core's small `common.pt`.
- `training_state.json`: completion identity and SHA-256 of `common.pt`.

The marker is published only after every rank's native save is recoverable.
Resume first reads the small typed common metadata with `weights_only=True`,
validates training contracts, then restores rank-local tensors after wrapping
the model. Tuple-valued provider contracts retain their types. Every rank
agrees on the common metadata checksum before loading.

Run-scoped latest/best links, retention, rollback and final-save behavior use
the same completed-directory check. Normal `Trainer.fit()` cleanup closes the
runtime; checkpoints returned by `save_checkpoint()` are already durable.

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
