# DeepInfer PR #92 → H100 V4.1 adaptation

Source: https://github.com/deepseek-ai/DeepInfer/pull/92 (open), pinned head `e205d42e4b28185e50ce30f6b7204cd01e59974c`, base `774ce244b87cc0eb8d4b94ab9f5de2ab22049f01`. Exact local reference: `/Users/shom/Downloads/deepinfer-pr92-sm90-20261105`. Reviewed source, not independently reproduced DeepInfer's H20 results.

## Measurement contract

PR reports 64K/128K TTFT 1.058/2.114 seconds on 8×H20-3e **with** `DEEPINFER_YOCO_PREFILL_TRUNCATION_MODE=compact`, keep=140, DSpark enabled. The original H100 series kept full prefill and no DSpark. Those numbers are not a controlled hardware/kernel comparison. On 2026-11-05 user explicitly authorized YOCO Compact, superseding the earlier full-prefill-only scope for new, separately labeled variants.

## What can be reused

- **MoE** (`sm90_ops/moe.py`, `v4_modules/mlp.py`): keep DeepEP V2 EPBuffer dispatch/combine; `do_expand=True` gives expert-aligned psum segments, then on-device psum→indexed tiles (no CPU count sync). Humming resident MXFP4 W4A8; max M tile=32, stream-K off. FP8 dispatch group=32, power-of-two scales. `swiglu_forward_and_per_token_cast` fuses gate/up activation, routing weight and group-32 quantization before GEMM2. Alignment gaps are skipped. This is a complete EP path, unlike all-gathering every token onto each expert rank.
- **Workspace**: PR still uses `torch.empty` per call for output, gate/up and index arrays; only tuning configs are cached. `num_max_tokens_per_rank` bounds dispatch capacity, and dead activation references are released early. It does **not** provide an existing reusable SM90 workspace pool to transplant. A pool needs stream/event ownership and graph address lifetime; cannot retain one growing buffer per layer without accounting for 40 layers.
- **Numerics**: Humming FP8-activation relative exponent representation may round groups beyond span=11. PR reports 28 MTP w13 matrices affected and no main-layer matrix, with small relative errors. Its optional FP8-resident backend checks exact MXFP4 conversion at load; costs ~1.9× expert memory. Do not equate these two conversions.
- **Attention**: FlashMLA ba89a34 sparse-prefill stays BF16 KV; vendor patch changes sparse-decode cache reads for V41 528B and V41_FP4 288B formats. Thus the patch is not automatically a full-prefill zero-dequantization speedup. Compare the actual current dispatch and whole KV chain first.
- **Indexer**: keys-major dense implementation plus compact candidate outputs; non-identity candidates use explicit gather scoring. The PR itself lists faster candidate-row scorer as follow-up, and does not port SM100's candidate-sparse MQA. Preserve candidate identity/prefix semantics before replacing ours.
- **mHC**: existing TileKernels path, no Hopper mega_mhc. Our post/combine/norm adaptation already reduces full-prefill overhead; under CP the local tensors are much smaller and its opt-in gate correctly stays off.

## Actual SGLang gaps

Current base `38d9e4a3f` already contains Humming runner + MXFP4 adapter + DeepEP v1 normal/LL conversion, including group-128 FP32 activation scales and masked SiLU quant. Its DeepEP v2 runner guard currently requires deep_gemm, so removing that guard would not create the psum adapter or fix incompatible scale layouts. DeepInfer uses group-32 packed/FP32 scales. A direct transplant must address both layout and weighting (DeepInfer applies routing before GEMM2; SGLang combines after GEMM2). Existing FlashInfer Humming option is distinct from standalone humming-kernels.

CP with A2A=none gathers all rows for MoE. Native allocator full64K attempts OOM at ~14.8GiB workspace. Run008 limits the entire prefill chunk to16K and preserves complete computation: cold64,326-token median **1.544297s**, samples1.644984/1.505558/1.544297, zero cached tokens. Rank0 whole-prefill GPU span1.530s, busy1.406s, NCCL kernel sum383ms (all ranks383–436ms), scorer36.5–36.9ms. Communication remains material; an EP implementation must be evaluated including dispatch, quantization, permutation, GEMMs and combine.

## YOCO Compact mapping

DeepInfer truncates at `n_prefill_layer`, preserves full source KV, gathers residual/mHC/candidate state and rebuilds candidate scheduling. CP uses contribution gather plus boundary remap. `KEEP_TOKENS >= window_size + MAX_BACKTRACKING_TOKENS` is required; published configuration keeps140.

SGLang already implements analogous bounded late-layer computation with `--enable-decoder-swa-bounded-replay`: write all source KV at layer20, retain last128 query rows for layers20–39, remap CP metadata and mHC state. It is not bitwise/numerically equivalent to full prefill and it is not an exact port of DeepInfer's140-row policy. Requests for prompt logprobs or full hidden states use complete late-layer computation. Eager CP is implemented; CP+prefill graph is unsupported. Run009 separately tests full CP16K control with output-only generation, compact128 CP16K, compact128 breakable-graph16K, with HBM sampling and whole-prefill traces. Production kernels unchanged by this configuration experiment.
