# CP prefill numerical parity

`NORTHGATE_CP_NUMERICAL_PARITY=1` enables `cp_numerics.install()` through the
capture plugin after the model loads. It is an explicit compatibility mode for
the pinned SGLang 38d9e4a3 V4.1 runtime on Hopper, TP8/EP8 with CP8 and attention
TP1. It has no dependency on captured reference activations or diagnostic files.
Model parameters are not replaced or modified. `install()` returns an undo
callback for resident debugging.

The original 4096-token CP capture differed from TP8 by 34.2747% relative L2 in
final hidden states. Small arithmetic differences changed sparse-key and MoE
routing decisions, which amplified through the model. Real-operand GPU
replays isolated shape-dependent failures:

- Indexer head-weight BF16 GEMM: full 4096 rows versus eight 512-row shards
  differed in 55 elements. Four tokens changed selected keys in layer 2.
- mHC combine/norm: the original four-warp prefill and eight-warp small-batch
  kernels differed on identical rows. The regression fixture reproduces 497
  differing elements; the fixed four-warp form is exact across tested row counts.
- mHC BF16x3 coefficients: the 64-row tile used for 1784-row CP shards was not
  repeatable on identical operands. The 128-row prefill tile matched the full
  14226-row reference for all eight shards and five repeats each. This identifies
  a failing kernel variant; it does not establish a specific compiler defect.
- Above 65536 global rows, Hopper TP switches mHC coefficient calculation to
  `hc_mix_stats` plus `hc_split_sinkhorn`. CP8's local 16384 rows still selected
  the BF16x3 fused path for a 131072-token request. The resulting small FP32
  coefficient differences amplified to 33.51% final-hidden relative L2. Selecting
  the coefficient path from the global sequence length alone restored exact
  aux/final agreement with the immutable TP capture in two full 128K replays.

The compatibility mode also preserves TP's ascending SWA key order and sparse
attention kernel, TP8 wo_b weight slicing and real all-reduce, and sequence-order
MoE reduction. Indexer head weights use the current full logical sequence's GEMM
geometry. All-gather operands are padded to equal *physical* CP lengths before
communication, including indexer operands whose local padding was removed.
These choices add communication and workspace; throughput and long-context
memory use require separate validation.

## Reproduce the GPU regression

Use the pinned serving image and the exported real fixtures. Run from this
checkout, without loading the 510 GB model:

```bash
python -m ops.northgate_capture.cp_numerics_regression \
  --index-fixture /path/to/index-weights.pt \
  --mhc-fixture /path/to/mhc-combine.pt \
  --stats-fixture /path/to/mhc-stats.pt
```

`check_collectives(parallel)` additionally runs on every live CP rank and checks
real NCCL gather/scatter ordering with unequal logical lengths. These kernel
checks supplement HTTP → capture plugin → Mooncake → native Brotli10/Zstd →
exact tensor read-back. They do not substitute for full-model comparison.

## Acceptance scope

The r7 resident job `j-1793927136-479755` compares two independent eight-H200
servers with unchanged model parameter pointers and versions. Original TP
outputs remain the reference. The full 14226-token real input and the 4097-token
probe pass with exact aux/final tensors and exact repeatability; the 4096-token
probe also matches the original immutable baseline exactly. Capacity-edge
15871/15872-token diagnostic inputs each pass two replays in both layouts with
exact tensor hashes and native roundtrips. Fixed numerical
thresholds remain global L2 ≤1%, cosine ≥0.99995, per-token P95 L2 ≤1%, and P05
cosine ≥0.99995.

The configured context is 16384, but the pinned scheduler's input admission
limit is 16378 (`min(context_len - 1, kv_capacity - 1) - 5`). The independent KV/SWA admission budget is stricter: with 256-token pages and
one reserved page, the live budget accepts at most 15872 tokens. Both layouts
left a 16377-token request queued until its HTTP client timed out; 16383 was
rejected immediately. No context or KV capacity was changed. Capacity-edge
fixtures cycle the real input and are diagnostic only; they are not production
training trajectories. The final acceptance report records their results.
This work does not release the full CP dataset or establish correctness and
capacity beyond the tested lengths. The unrelated non-CP short-capture job is
not part of this change.

## 128K dispatch regression

The standalone real-operand regression reuses the recorded r7 residual fixture
and reads only `layers.0.hc_attn_fn` from the checkpoint. It does not load the
full model. Run it in the pinned image on one Hopper GPU with `RUN_DIR` and
`NORTHGATE_ROOT` set to the experiment output and evidence root:

```bash
python -m ops.northgate_capture.cp128k_mhc_regression
```

It cyclically extends captured rows to 65537 and 131072, verifies the checkpoint
weight against the fixture, reproduces coefficient differences for all eight
logical shards, and requires the global-length path to match the full reference
and its repeat exactly. This is an operator regression, not a claim that the
cycled input is a complete production trajectory.

The 128K full-model ablation uses context/KV capacity 135168, full unchunked
prefill, and the same address-only `mhc_index64` repair on the reference and
candidate. The exact 131072-token diagnostic input is a prefix of a longer real
record. Two raw captures match every immutable TP tensor hash with zero cached
tokens; parameters remain unchanged on all eight ranks. The repaired output
also passes native Brotli10/Zstd exact read-back, and both diagnostic jobs finish
Succeeded with no active replicas. This establishes the
cause for that input, not acceptance of every long trajectory or full CP data.
