# YiCloud 32 H200 all-anchor AdamW recipe

This is the isolated launcher used for the November 6 LR experiments, with peak
LR set to **2e-4**. It does not change library defaults. The 1e-5 and 1e-4 BP runs
completed; YiCloud 5e-4 was stopped at update 39 at the user's request. The 2e-4
replacement starts from seed 42, not the stopped checkpoint.

## Runtime and inputs

Use the pinned image in `payload.example.json`, four replicas with eight H200s
and eight active IB ports each, and the existing YiCloud `training_run.py`
prepare/submit workflow. The entry command is in `command.json`. It must run
from the clean source checkout with the runtime-provided `RUN_DIR`,
`KUBEBRAIN_*`, `WORKER_ALL_SOCKET_ADDRS`, and `NCCL_IB_HCA` environment.
`RUN_DIR` must be a new, empty run directory; never reuse another run's outputs.

- `MIGRATION_ROOT/runtime-deps/train-core` and `runtime-deps/nfc-src` provide the
  previously validated Core/NFC sources. The preflight checks the Core file hash
  and real image APIs. No packages are installed during launch.
- `INPUT_ROOT` contains `input-compatibility.json`, `data-audit.json`,
  `train-features/` and `validation-features/`. The compatibility receipt must
  describe 1,326 training and 17 validation traces. This experiment has 15,104
  training windows (including 51 explicit padding windows) and 136 unpadded
  validation windows. The distributed validation sampler pads to 160 samples.
- `YICLOUD_RENDEZVOUS_NAMESPACE` selects the platform service namespace. Bootstrap
  uses the interface matching this replica's DNS address, not an assumed bond0
  address or the order of membership entries. Data transport remains IB.
- `prepare_node.py` resolves the paths in `recipe.yaml` from these roots. Paths
  `/INPUT_ROOT` and `/RUN_DIR` in the template are placeholders; do not launch the
  template directly through the CLI.

Prepare with the existing platform helper, inspect the generated CreateJob
request, then submit that exact run. The example project, quota and inputs are
historical deployment values; check live capacity and access before submission.
The helper and credentials are not bundled here. Credentials must not enter a
payload, source tree, run directory or log. No code here stops other jobs.

## Training and acceptance

The fixed experiment uses AdamW, FULL_SHARD, CPU optimizer offload, batch 1 per
rank, accumulation 4, global batch 128, 118 updates, all legal anchors, chunk64,
and FP32 reduction. `num_anchors: 32` is inactive when `anchor_sampling: all`.
Warmup is `int(0.03 * 118) = 3` updates followed by the existing cosine schedule.
The existing step-4 handoff overshoot (about 0.01866%) is preserved for comparison.
Changing peak LR also requires updating its assertion in `prepare_node.py`.

The launcher checks real train/validation features on each node, runs the
32-rank streamed-versus-whole FSDP gate, then audits every actual training batch:
finite loss/gradient, exact eligible-anchor coverage, no per-window duplicate
anchors, frozen target tensors, optimizer update counts and GPU memory. Fixed
reduction settings do not guarantee every custom kernel is bitwise deterministic.

Logs and receipts live under `nodes/<node>/training` and `training`. A
`micro-progress` started record is in flight; JSONL microbatch records are
completed batches. `live-progress` changes only at optimizer updates. Success
requires all 32 completion receipts (118 updates / 472 training microbatches),
all four exit codes, the platform terminal state and final checkpoint metadata.
`finalize.py` does not perform a restore test. Validation/checkpoints run every
25 steps; checkpoint118 has no additional final evaluation.

See [experiment evidence](../../acceptance/dspark-adamw-lr-20261106/README.md)
for exact executed revisions and the limits of cross-platform comparisons.
