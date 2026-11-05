# Muon + AdamW

AdamW remains the default. Native DeepSeek V4.1 DSpark can opt into
`training.optimizer: v41_flash_muon`; see the
[offline recipe](../examples/configs/offline/colocated/deepseek-v41-flash-muon.yaml).
This names the **published backbone Muon algorithm**, with explicit DSpark
parameter choices below. It does not claim to reproduce an unpublished DSpark
training recipe or to improve acceptance rate without a controlled experiment.

## Algorithm and sources

- [V4.1 report §2.5 and §4.2.2](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash/blob/main/DeepSeek_V41_Tech_Report.pdf): head-wise Query/Key Muon, Nesterov momentum 0.95, decoupled decay 0.1, update RMS scale 0.18; AdamW `(0.9, 0.95)`, epsilon `1e-20`, decay 0.1 for norm weights and no decay for biases/scales.
- [V4 §2.4 / Algorithm 1](https://arxiv.org/html/2606.19348v1#S2.SS4), inherited by V4.1: `M = 0.95 M + G`, orthogonalize `G + 0.95 M`, eight quintic iterations with `(3.4445, -4.7750, 2.0315)` followed by two with `(2, -1.5, 0.5)`.
- The published rescaling is `direction *= 0.18 * sqrt(max(rows, columns))`, **not** a fresh division by the measured RMS. With a finite number of NS iterations or deficient rank, measured RMS can differ from 0.18. Zero gradients produce zero directions, while weight decay still applies.
- [TorchTitan DistMuon at a97b83b](https://github.com/pytorch/torchtitan/blob/a97b83bbbfebb9bf121641890f48e9986f7f3c78/torchtitan/distributed/flex_shard/dist_muon.py) supplies the numerical implementation reference. Its generic defaults (EMA momentum, five NS iterations, `0.2 * sqrt(max(m,n))`) are not the V4.1 profile. This implementation uses the paper's unnormalized momentum state and 8+2 schedule. BF16 NS computation and norm-clamp epsilon `1e-7` are explicit implementation choices; FP32 NS is also supported.

The named profile rejects changes to published algorithm constants. For ablations,
use `optimizer: muon_adamw` and `muon.profile: generic`. `MuonConfig` exposes
momentum convention, Nesterov, NS iteration count/coefficient schedule/dtype,
epsilon, scaling, both group LR multipliers and decays, and Adam betas/epsilon.
All parameters share the existing pre-update global clipping operation and LR
schedule. The recipe's `1e-5` base LR, warmup and data budget are project choices.

## Native logical matrices

Roles and original shapes are captured **before wrapping**; local shard dimensions
are never used to infer a role. Every trainable parameter has a recorded name,
shape, role and matrix shape. Frozen target tables are excluded.

| Parameter | Update and matrix boundaries |
| --- | --- |
| `wq_b` | Muon independently on each `[head_dim, q_lora_rank]` Q head; the native row layout keeps each whole head contiguous |
| `wkv` | Muon on `[head_dim, hidden_size]`; native KV is one shared head, not fused K/V halves |
| `wq_a`, `wo_b`, `main_proj` | One Muon matrix per projection |
| `wo_a` | One `[o_lora_rank, heads * head_dim / o_groups]` matrix per output group |
| routed `gate_up` | Each expert's gate and up are separate `[intermediate, hidden]` matrices; never orthogonalize the expert axis |
| routed `down`, shared expert projections | One matrix per expert/projection |
| router weight, mHC `fn`, confidence projection | Muon on each linear weight; mHC `fn` is treated as one projection |
| norm weight | AdamW, decay 0.1 |
| bias, sink, mHC base/scale | AdamW, decay 0 |
| Markov embedding and Markov prediction head | Explicit DSpark AdamW fallback, decay 0; **not** the main model's Sinkhorn-balanced optimizer |
| frozen target embedding / prediction head | Excluded; no optimizer state |

The report does not publish a separate DSpark optimizer mapping. Markov fallback
and the mHC projection boundary are explicit local choices. Q heads are not
additionally split into RoPE/non-RoPE regions; the V4.1 report specifies splitting
by head. The generic profile recognizes module-owned Linear weights and uses
AdamW for embeddings, vectors and unknown parameters (recorded as
`unknown_adamw`), rather than guessing from names.

## FSDP and memory contract

Supported: FSDP1 `FULL_SHARD`, `use_orig_params=True`, a single WORLD group;
replicated DDP `NO_SHARD`; and an unwrapped model. `SHARD_GRAD_OP`, subgroups,
hybrid sharding and implicit TP/EP adaptations are rejected. This PR does not
migrate the backend or change loss normalization, accumulation or anchor sampling.

FP32 master weights, Muon momentum and AdamW slots live on the CPU when
`optimizer_cpu_offload: true`, otherwise on the parameter device. NS always runs
on the model device. For every logical matrix, each rank places its owned slice
of the prepared momentum into a zero-filled FP32 matrix, all-reduces it, computes
the complete matrix direction and applies only its own slice. This favors simple
correctness over communication efficiency: all ranks compute the same NS result.
Empty shards participate in the same order; FSDP padding is excluded. Shared
clipping reduces disjoint shard norms once; DDP norms are not summed again.

Only one matrix is reconstructed at a time, including within stacked experts.
`max_matrix_elements` (default 128 Mi elements) rejects oversized matrices before
training. Temporary memory is proportional to `rows*columns` plus
`min(rows,columns)^2` for NS Gram matrices. Persistent storage is a FP32 master
plus one momentum shard for Muon, or two moment shards for AdamW, in addition to
model/gradient storage. CPU offload incurs one host-to-device prepared update and
one device-to-host direction per locally owned matrix slice. Real budgets and
step time must be measured with the chosen model, checkpoint staging pool and
accumulation configuration.

## Checkpoints

NFC stores FP32 masters, `momentum_buffer`, Adam moments/step counters, scheduler,
RNG and the existing data cursor. The saved optimizer identity contains resolved
configuration, implementation version, pinned reference, canonical names/roles,
logical shapes, shard offsets/counts, topology and schedule. Restore validates
identity **before loading model payloads**. Lazy step-zero state is valid.

Resume requires the same world size and FSDP wrapping/layout. CPU placement may
change. AdamW ↔ Muon or a profile/group/layout change requires an explicit
weights-only warm start through the model initialization path, not `resume_from`.
Existing AdamW NFC layouts without the new identity field remain readable.

## Reproducing validation

CPU tests: `python -m pytest tests/test_optimizer tests/test_runtime/test_nfc_checkpoint.py tests/test_config/test_schema.py`.

For actual FSDP/NFC on eight CUDA GPUs with the native NFC runtime installed:

```sh
torchrun --standalone --nproc_per_node=8 tests/gpu/muon_fsdp_acceptance.py \
  --output /path/to/isolated-results --phase write --reference /path/to/pinned/dist_muon.py
torchrun --standalone --nproc_per_node=8 tests/gpu/muon_fsdp_acceptance.py \
  --output /path/to/isolated-results --phase resume
```

The reference file is verified against its Git blob hash before executing the
actual TorchTitan numerical helpers. The script compares complete-matrix updates
to real FSDP updates with accumulated gradients and clipping, including empty and
uneven shards. Predetermined master tolerances are `atol=rtol=2e-6` for FP32 NS
and `2e-4` for BF16 NS; same-placement fresh-process resume must be bitwise.
The full-model pilot, controlled validation acceptance/serving comparison and
long-budget convergence are distinct from this small numerical test.

### Full native configuration: static storage estimate

The checked-in 3-layer / 128-expert configuration assigns 14,159,123,712 elements
to Muon and 66,238,050 to AdamW. It contains 192 independent Q-head matrices,
768 routed gate/up matrices and 384 routed down matrices. The largest matrix is
`main_proj` at 5,120 × 15,360 (78,643,200 elements), below the default bound.
FP32 master/momentum/Adam slots total about **106.23 GiB across all ranks**,
excluding gradients, model weights, temporary workspace and NFC staging. This is
a shape-derived estimate, not a measured peak. Full-matrix all-reduce payload is
about 52.75 GiB per rank per optimizer step before accounting for collective
transport overhead; optimizing that communication is separate work.
