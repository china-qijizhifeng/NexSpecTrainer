# Native DeepSeek V4.1 Flash MTP training

This candidate strategy trains the released three-stage DSpark module under
`mtp.*`. It initializes from the native checkpoint, freezes the target embedding
and output head, and exports a separate checkpoint directory for SGLang.
**Full-checkpoint eight-H100 training and native export passed a short
real-trajectory validation. Exported SGLang DSPARK inference ran, but strict
target-output preservation did not pass.** See the
[2026-11-03 validation record](validation/deepseek-v41-20261103.md) for the
two-call smoke run, RDMA/compression measurements and unresolved inference limit.
All 17 small checkpoint, kernel and training acceptance tests
passed on H100 with the preserved Weaver Torch 2.10 / TE 2.20.2 stack. The initial learning rate is a conservative
starting point, not a reproduced training recipe.

## Kernel and model contract

| Component | Implementation |
| --- | --- |
| Expert forward/dgrad/wgrad | DeepGEMM BF16 M-grouped / K-grouped GEMM |
| Expert dispatch/combine | Megatron `permute` / `unpermute`, mandatory TE fused path |
| Router, clamped weighted SwiGLU | TileKernels forward/backward |
| mHC split, Sinkhorn, pre/post mix | TileKernels forward/backward |
| RMSNorm | TileKernels; wide FP32 mHC normalization uses TE |
| RoPE / inverse output RoPE | TileKernels, interleaved tail dimensions |
| Attention with trainable sink | Megatron packed CSA: FlashMLA forward + cuDNN DSA backward |
| Dense projections | PyTorch linear/bmm dispatch to cuBLAS |

PyTorch supplies module parameters, autograd bindings, indexing, checkpoint I/O
and metadata. There is no eager numerical fallback for these training kernels.
Expert matrices are stored stacked; native checkpoint conversion maps back to
individual `experts.{i}.w1/w2/w3.weight` entries. Empty expert routes remain
represented in grouped wgrad. Training uses BF16 dequantized weights; it does
not reproduce inference FP4/FP8 rounding or implement quantization-aware training.

The draft has approximately 14.23B parameters: 5120 hidden width, three stages,
128 routed experts/top-3, one shared expert, four mHC streams, five draft tokens,
Markov rank 256. It captures **layer inputs** 37/38/39, averaging the four mHC
streams per layer and concatenating to width 15360. The target final hidden state
has width 5120 and must be post-final-norm. Confidence consumes the draft's
pre-final-norm collapsed hidden state. A block at anchor `a` reads target
positions `[a-128, a)` and has private, bidirectional draft positions `a..a+4`.

FSDP wraps attention and MoE separately and retains mHC/router/sink/confidence
parameters in FP32 units. The example uses eight trainers and CPU Adam state
offload. DSpark keeps `num_anchors` slots on every rank even when a sample has
fewer supervised anchor positions. Extra slots are masked out of loss and
metrics. This keeps objective-head FSDP collectives in the same order during
forward and activation-checkpoint replay; selected valid anchors are unchanged.
There is no expert parallelism or trainer tensor parallelism in this
initial adapter. Every rank initially materializes the full BF16 draft before
FSDP sharding; allow over 30 GB just for draft and frozen head/embedding storage,
plus checkpoint conversion, activations, optimizer and communication memory.
The larger training recipe is not a measured memory guarantee. The separate
[bounded smoke recipe](../examples/configs/offline/colocated/deepseek-v41-flash-smoke.yaml)
used four anchors and two-block objective chunks and peaked at 38.33 GiB of
allocated CUDA memory per rank in that validation.

## Install the trainer stack

Prefer the pinned Weaver v1.14.92 base with Python 3.12, CUDA 13.0,
Torch `2.10.0a0+b558c986e8.nv25.11` and TE `2.20.2+6ea2a74`.
Keep these Torch/TE binaries and build DeepGEMM 2.8.2 and FlashMLA against that
Torch ABI; see `docker/deepseek-v41/Dockerfile.weaver`. Import
`transformer_engine.pytorch` before probing `transformer_engine_torch`.

Pinned TileKernels declares `torch>=2.13`, but its required operators import
with TileLang 0.1.15 on Weaver's original Torch. H100 numerical checks passed
for RMSNorm (512/5120), TE wide RMSNorm (20480), mHC, SwiGLU and RoPE in both
directions. This validates that kernel subset, not all TileKernels features.
Use a source overlay for this compatibility path; do not let the general
trainer or TileKernels dependency resolver replace the base Torch/TE.

The base DeepGEMM 2.3 rejects the prefix-sum layout on SM90, so symbol presence
alone is insufficient. The base Megatron also lacks packed CSA; attention
still needs the pinned CSA implementation, FlashMLA and compatible cuDNN DSA.
The small acceptance suite also passed fused MoE dispatch, private-window
attention forward/backward and two DSpark AdamW steps with recomputation.
The subsequent full-checkpoint, multi-GPU smoke run is recorded above; longer
training and convergence remain unvalidated.

The K-grouped wgrad binding uses aligned host K lengths and an FP32
accumulator required by SM90. Each MoE dispatch synchronizes padded counts to
the host once; both expert GEMMs reuse them. Other GPU architectures are rejected. Keep the SGLang capture server in a separate environment.

Use the offline Weaver build/assembly workflow in
[`docker/deepseek-v41/README.md`](https://github.com/china-qijizhifeng/NexSpecTrainer/blob/main/docker/deepseek-v41/README.md). It builds
DeepGEMM commit `5081f8513af57e61696de4f4f04b0e707ffb503e` and FlashMLA
`b7643bd54521f563b839b98289b5cd048c062ba2` against the existing Torch ABI,
then adds the pinned TileKernels/Megatron source and TileLang/CuTe DSL overlays.
The assembler checks every input hash and the original Torch/TE binary paths.
Do not use an unrestricted `pip install -e .` or the general TileKernels
requirements on this base: those dependency declarations can replace Torch.

Preflight checks version and API contracts before loading the model. Numerical
and training acceptance uses the following commands:

```bash
python -c 'from specforge.modeling.draft.deepseek_v41.validation import preflight; preflight()'
python -m pytest -q tests/test_modeling/deepseek_v41
```

## Prepare target activations

Use a complete, immutable local HF snapshot of either
`deepseek-ai/DeepSeek-V4.1-Flash` at
`314113f8a7dadb388215396ba5fbee2c610a04d6`, or
`deepseek-ai/DeepSeek-V4.1-Flash-1017` at
`1bae0e1088993b9b82de08f59de5be5a34e42170`.
The configuration is checked against native target fields before training.
Do not mix activations from one target revision with weights from the other.

Use the V4.1 checkpoint's official `encoding/encoding.py` to prepare token IDs
and an assistant-token loss mask; online JSONL records contain `input_ids` and
`loss_mask`. The online example uses `data.prompts_path` to avoid applying the
older V4 chat template. Responses from an API or HF dataset still need a local
teacher-forcing pass to obtain hidden states.

For the separate SGLang server, use the inspected revision in
[the source notice](https://github.com/china-qijizhifeng/NexSpecTrainer/blob/main/NOTICE.deepseek-v41.md), with the
[capture plugin](https://github.com/china-qijizhifeng/NexSpecTrainer/blob/main/plugins/sglang-spec-capture/README.md). The capture flags are:

```text
--skip-tokenizer-init
--chunked-prefill-size -1
--speculative-algorithm DSPARK
--enable-return-aux-hidden-states
--return-aux-hidden-state-layer-ids 37 38 39
--return-hidden-states-mode full
```

Set `SPECFORGE_SPEC_CAPTURE=1` and the Mooncake endpoints as documented by the
plugin. Add V4.1's hardware-specific target TP, FP4 expert and Engram settings
from the SGLang deployment that already runs on your cluster. Stock trainer
SGLang 0.5.18 is not the V4.1 capture environment.

Use the [kernel cache launcher](sections/examples/kernel-cache.md) around that
server command to retain Triton, TileLang, DeepGEMM, and FlashInfer caches across
jobs. It enables sampled DeepGEMM prewarm by default; keep separate namespaces
for the preserved Weaver trainer and the V4.1 serving image. The guide also
covers local NVMe mounts and the DSPARK MoE prefill autotune limitation.

**Capture acceptance remains required:** send a complete prompt+response with
`max_new_tokens=0`; check full token coverage, 15360-wide layer-input features
and 5120-wide final states. V4.1's CED bounded replay must not truncate deep-layer
features. Merely starting an inference server does not establish this contract.
Persist `input_ids`, `loss_mask`, `hidden_states`, `target_last_hidden_states`
using SpecForge's manifest format and `strategy=deepseek_v41` for offline use.
The plugin/native V4.1 combination passed capture of the prepared 4k windows
described in the validation record; longer full-episode capture remains a
separate acceptance target. For device allocation, transport settings and the
larger startup window, see the [Mooncake RDMA recipe](sections/examples/mooncake-rdma.md).

## Train and export

Adjust target/data paths in either example, then run:

```bash
specforge train --config examples/configs/offline/colocated/deepseek-v41-flash-native.yaml
# Or, with a separate capture server and Mooncake already running:
specforge train --config examples/configs/online/disaggregated/external/deepseek-v41-flash-native.yaml
```

For the published V4.1 backbone Muon algorithm plus AdamW fallback, select
`training.optimizer: v41_flash_muon` or use
`examples/configs/offline/colocated/deepseek-v41-flash-muon.yaml`.
The [Muon contract](muon.md) describes head/expert boundaries, CPU offload,
checkpoint compatibility, and the explicit DSpark Markov fallback. AdamW remains
the default; this option is an experimental DSpark training recipe.

Export a runtime checkpoint from a trusted training run:

```bash
python -m specforge.modeling.draft.deepseek_v41.export \
  --checkpoint outputs/deepseek-v41-native-offline/<run-step-directory> \
  --target /models/DeepSeek-V4.1-Flash-1017 \
  --output /models/DeepSeek-V4.1-Flash-1017-trained-mtp
```

Export re-quantizes each MTP tensor to the native template format, preserving
FP8 32x32 scales and packed FP4 per-32-column scales. It writes new MTP shards
and an index, and symlinks unchanged target files. Keep the source snapshot
available; dereference links when transferring the export. Existing output
directories are rejected. Use the new directory as the native model path in
your working SGLang DSPARK launch. Verify load, draft logits, confidence,
acceptance rate and throughput before using the trained module in production.
