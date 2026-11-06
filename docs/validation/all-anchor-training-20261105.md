# Native V4.1 exhaustive-anchor training — 2026-11-05

This implements the exhaustive-anchor portion of [issue #13](https://github.com/china-qijizhifeng/NexSpecTrainer/issues/13).
Every legal anchor is trained once per window; chunk size controls memory rather
than coverage. The existing uniform sampling remains the default.

## Training behavior

Set `training.anchor_sampling: all` with `training.strategy: deepseek_v41`.
`num_anchors` is ignored in this mode. An anchor is legal when its position and
first target are supervised; subsequent targets retain the existing continuous
mask. Ranks pad to a common width so empty ranks still execute every collective.
Metrics report eligible, selected, excluded and padded anchors, duplicates, and
target exposure at each speculative position.

`training.anchor_forward_chunk_blocks` defaults to 32. Each chunk completes a
root FSDP forward/backward before the next starts. The whole microbatch's global
weighted denominator is shared by all chunks, and gradients accumulate in FP32
local shards before rounding once per microbatch. FSDP prefetch is disabled in
this mode. Optimizer boundaries still follow `accumulation_steps`.

Native grouped expert weight gradients now allocate FP32 workspace for eight
experts at a time, retaining each expert's complete token reduction. RoPE cache
views keep their requested length during recomputation. Overlapping context
gathers accumulate their backward contributions in FP32.

`specforge.data.v41_supervision.encode_supervision` is a separate, opt-in utility
for applying existing `use_loss` / `use_thinking` flags to the official encoder's
unchanged token stream. It also emits tool-call IDs. Callers must supply an
explicit missing-flag policy and a private encoder module because hooks are
temporary. This utility is not automatically wired into capture and does not
retroactively change existing training data or implement cleaning heuristics.

## Measured memory and performance

The experiment reused 1,326 traces, eight H100s, FULL_SHARD and CPU optimizer
offload. It retained batch size one per rank, accumulation 16, seed 42, LR 1e-5,
three warmup steps, cosine horizon 118, and the previous objective weights.
The GPU image used Torch `2.10.0a0+b558c986e8.nv25.11`; local CPU validation used
Torch 2.12.1. These are distinct from the repository's Torch 2.13 CI environment.

The earlier full-graph attempt reached 76.28 GiB and ran out of memory on the
second backward, before an optimizer update. Streaming completed 17 real
microbatches, including an Adam update and the following backward, at 35.50 GiB
peak before the later FP32 accumulation changes.

The following final comparisons use the same full-size real microbatch, the
second execution after warmup, and the slowest of eight ranks:

| Policy | Seconds / microbatch | Peak allocated GiB |
| --- | ---: | ---: |
| Ordinary chunk32, inner and outer checkpoint | 73.306 | 40.942 |
| Ordinary chunk32, outer checkpoint only | 61.611 | 40.944 |
| Fixed settings, chunk32, outer checkpoint only | 83.139 | 44.132 |
| Fixed settings, chunk64, outer checkpoint only | 44.077 | 44.316 |

Fixed chunk64 improved throughput 1.66× over the ordinary chunk32 baseline and
1.89× over fixed chunk32. Extrapolating this single microbatch gives about 23.1
hours of pure training; window variation, evaluation and checkpoint time are
not included. Disabling `gradient_checkpointing` in the native draft config
disables the inner stage checkpoints; exhaustive training still retains the
outer complete-chunk checkpoint.

## Numerical acceptance and limits

All ranks preserved anchor counts and position exposure. Fixed chunk64 versus
fixed chunk32 had maximum relative loss error 0.02855% and gradient relative L2
error 4.6234%, below the predeclared 5% full-model threshold. Fixed chunk32
self-repeat gradient error was 0.1969%. Ordinary chunk64 attempts failed the
same threshold (up to 5.47%); the threshold was not relaxed. These measurements
do not establish identical training trajectories or convergence.

The separate real eight-rank FSDP/controller regression passed 32 microbatches
and two optimizer updates per policy. Maximum gradient relative L2 difference
was 0.41999%; FP32 master-update difference was 0.02644%. This smaller fixture
tests streaming versus whole-graph execution, not full-model chunk32 versus
chunk64. CPU regressions cover missing/empty supervision, unequal rank coverage,
denominator precision, optimizer boundaries, lost BF16 increments, overlapping
gather gradients, expert reduction and RoPE recomputation.

The final local regression run passed 160 tests and 435 subtests; 12 tests
requiring unavailable environment features were skipped. The checked-in fixed
policy also passed the two-process CPU FSDP gate (32 microbatches and two updates
per policy). GPU acceptance above was performed separately on the pinned image.

The [Nex-Megatron reference](https://github.com/china-qijizhifeng/Nex-Megatron/commit/56a1aa958c43dddc4fe7598800d73959bcbb0e31)
informed the numerical comparisons and retained failure evidence. Its QuACK
activation kernel was not transplanted into this different native kernel stack.

## Reproducing the fixed reduction experiment

Fixed settings are an opt-in experimental launch policy, not the default CLI
configuration. The exact tested helper is retained in
[`tests/gpu/fixed_reduction.py`](https://github.com/china-qijizhifeng/NexSpecTrainer/blob/main/tests/gpu/fixed_reduction.py). It changes
FSDP construction process-wide and must be installed once, before wrapping any
model. Custom kernels still prevent a bitwise-determinism guarantee.

Set these variables before starting the worker processes:

```bash
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export NVIDIA_TF32_OVERRIDE=0
export NVTE_ALLOW_NONDETERMINISTIC_ALGO=0
export NCCL_ALGO=Ring
export NCCL_PROTO=Simple
```

The helper enables PyTorch deterministic algorithms, disables CUDA matmul and
cuDNN TF32 and cuDNN benchmarking, and replaces FSDP mixed-precision
`reduce_dtype` with FP32. Save this gate entry point as `fixed_gate.py`:

```python
from tests.gpu.fixed_reduction import install

install()
from tests.gpu.all_anchor_fsdp_acceptance import main

main()
```

Run it from the repository with the repository on `PYTHONPATH`:

```bash
PYTHONPATH="$PWD" python -m torch.distributed.run --standalone --nproc_per_node=8 \
  fixed_gate.py --output /path/to/fixed-gpu-regression
```

For the small CPU FSDP regression use two processes and `--device cpu`.
For training, use the same entry point with `from specforge.cli import main`
and pass `train --config /path/to/train.yaml`. Set `anchor_sampling: all`,
`anchor_forward_chunk_blocks: 64`, and draft `gradient_checkpointing: false`.
Revalidate numerical acceptance on another hardware/software stack before
assuming the measured chunk64 behavior transfers.

## Retained evidence

Evidence roots on the experiment host are under `/gpfs/users/linlizhi/`:

- `dspark-stream-memory-20261105-r2`: real microbatch memory acceptance.
- `dspark-perf-20261105-r4`: checkpoint comparison.
- `dspark-perf-20261105-r5`: fixed chunk32/64 comparisons, per-rank gradients,
  `performance-selection.json` and source fingerprints.
- `dspark-all-anchors-20261105-r9`: full-run configuration, source/runtime
  fingerprints, fixed GPU regression, per-microbatch and per-update receipts.

Full-run job `job-7b468f7cfd17-20261105162116` restarted with the selected policy.
At restart validation its second real microbatch took 44.049 seconds at
44.316 GiB peak. This report records startup acceptance, not completion of all
118 optimizer steps, final checkpoint restoration or downstream model quality.
