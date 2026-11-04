# dSpark stage1 and held-out test run fixes

The executed source fixes are now on GitHub `main`:

- `952e70c3e8109e5295436141b6c6336897448bc2` pads masked anchor slots so every
  rank executes the same number of objective chunks. Valid anchor selection,
  loss and gradients are preserved.
- `e925d251e9eece36339b3bc2a888ad19e15727ef` registers CUDA WORLD as
  `cpu:gloo,cuda:nccl`, keeping native NFC host object collectives on Gloo and
  CUDA FSDP tensors on NCCL. This does not establish a hardware root cause for
  the earlier save failure.

This directory preserves the additional fixes previously present only in the
run directories. `source-provenance.json` records their original locations and
both executed and archived SHA-256. Repository formatting and import ordering
are applied to the archived copies; their non-import AST is identical. Removed
imports are unused standard-library names and one redundant protobuf import
that is still loaded through its retained `Type` import.
They retain their pinned run paths and are historical operational artifacts,
not a portable benchmark CLI. Do not launch the archived controllers or submit
their YAML again: these aliases and single-use intents already belong to the
recorded attempts.

## Training wrapper

`stage1-wrapper/audit_train.py` reads the public
`trainable_module().training` state instead of an absent `self.model` attribute,
and counts training and evaluation microbatches separately.
`wrapper_regression.py` extracts the exact nested wrapper by AST, reproduces
the old error with the previous run's script, and exercises train/eval/train
and finite backward using the real native strategy.

The source regression suite passed 48 CPU checks. The subsequent completed
stage1 run executed 118 formal updates, 1,888 training microbatches per rank
on eight ranks, and verified the input sequence and frozen table hashes.
The copied CPU receipt is historical and predates those GPU checks.

## Test export and runner

`test-r2/common.py::checkpoint_metadata` checks the actual NFC layout:
`training_state.json` plus `nfc/common.pt`, the metadata checksum, step/format
and 32 tensor shards. Both the controller's pre-submission check and exporter
use it. The first attempt checked a nonexistent root-level `common.pt`; it
failed before any export or GPU evaluation and was precisely cleaned up.
`r1-failure-analysis.json` records the static review, fix and bounded new attempt.

`check_runner.py` tests the valid nested layout, corrupt metadata rejection,
root-level decoy rejection, SSE framing and cumulative/incremental token
counts, plus verification-weighted metric aggregation. These checks pass
without Torch or GPU dependencies:

```sh
python3 acceptance/dspark-stage1-test-20261104/test-r2/check_runner.py
```

The fixed protocol compares target-only, official MTP, stage1 step25 and
step118 on 247 held-out trajectories / 36 instances. Each trajectory supplies
one complete-history prefix up to 64,512 tokens, followed by 1,024 generated
tokens. The 176 originally longer trajectories are represented by earlier
prefixes; this is not full long-context or closed-loop SWE task evaluation.
Inputs were frozen before outputs; no test-driven selection or tuning occurs.

The earlier FP4 AR/verify greedy discrepancy remains unresolved. The bounded
diagnostic matrix records A/A stability, first disagreement and common-prefix
logprobs. A mismatch must prevent a lossless speedup claim even when the
measurement run completes. The evaluation result is still pending when this
archive is created; published training checkpoints and CPU runner regressions
do not constitute test scores or evaluation completion.

## Run locations and validation

Stage1 source: `e925d251e9eece36339b3bc2a888ad19e15727ef`.
The test worker uses the same pinned source archive, independent of later
repository commits. Its source and execution checksums remain unchanged.

- Training: `/gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6`.
- Current test attempt: `/gpfs/users/linlizhi/dspark-test-20261104-r2`.
- Failed test preflight: `/gpfs/users/linlizhi/dspark-test-20261104-r1`.

The native final checkpoint was freshly restored on eight ranks, exercised
with a diagnostic Adam update at the unchanged terminal LR=0, and restored
again to the initial restored hashes. This does not prove a nonzero-LR weight
update or a final-save-before/restore-after state hash comparison. The formal
training covers 1,326 complete train trajectories <=64K; 2,862 longer train
trajectories remain deferred.

The formatted archived Python files pass compilation, shell entrypoints pass
`bash -n`, and the archived CPU runner checks pass. They require the original
private data manifests, complete target snapshot, native checkpoints, pinned
images, dependency overlays and original run receipts for actual execution.
Those datasets, weights, checkpoints and credentials are not part of this archive.
