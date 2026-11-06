# 32-GPU AdamW startup accepted

Job `job-0a081d7a6923-20261105192255`, alias `dspark-all-anchors-32gpu-20261105-r1`; WeaverRSI / proj-32. Started 2026-11-05 19:22:55 Beijing. Four nodes: node022, node023, node027, node028; eight H100 per node. NCCL logs confirm IB/RoCE on all four nodes.

Old AdamW r9 was stopped by user request after 14 updates; exact deletion succeeded and readback was absent. Its logs and restart receipt remain retained. No checkpoint was available; this run starts from the same initialization. Muon `job-3a1b09d413fd-20261105171629` was preserved.

Configuration remains 1,326 traces, all legal anchors, chunk64, FP32 reduction, AdamW LR1e-5 and 118 optimizer updates. Accumulation changed 16 → 4 for world8 → world32, preserving global batch128. Same hash-verified r9 source, draft config and source/data audit. Validation and NFC checkpoint every25 updates. Cross-topology numeric identity is not claimed.

## Validation

- All four real-image/GPU/API preflights passed.
- 32-rank synthetic FSDP reference-vs-streamed gate passed, accumulation4, eight micros/two updates: maximum gradient relative L2 0.7583% (limit3%), master-update 0.1757% (limit10%).
- All32 ranks completed one actual AdamW update and six training microbatches, including post-update backward. Loss and gradient finite; every selected input covers all legal anchors with zero anchor duplicates; frozen target tables excluded from optimizer.
- Maximum allocated GPU memory33.29GiB; allocator reserved maximum74.18GiB. These describe different memory measures.
- No durable checkpoint yet. Checkpoint restoration and model-quality improvement are not verified.

## Early timing

First micro71.30s; subsequent non-update micros49–51s; update micro62.80s. First optimizer step including warmup233.76s. Early steady-step estimate212.0s, compared with r9 median727.6s (~3.43×). At that rate118 updates take about7.0h of pure training, excluding evaluation/checkpoint/init. This is an early estimate, not a completed-run benchmark.

Existing heartbeat `dspark-anchor-r9` now watches this exact32-GPU job. Exact-identity terminal cleanup monitor PID45414 retains receipts before cleanup and cannot submit jobs. Per-node logs/exits live under `nodes/<node>/training/`; rank progress lives under `training/`.
