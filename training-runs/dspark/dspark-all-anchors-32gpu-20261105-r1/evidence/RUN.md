# AdamW 32-GPU restart

Requested in the original chat on 2026-11-05. New alias: `dspark-all-anchors-32gpu-20261105-r1`.

- Four nodes, eight H100 each, WeaverRSI / proj-32, mandatory IB/RDMA.
- Identical r9 source snapshot, data audit, model initialization, pinned image and optimizer recipe.
- 1,326 training traces; global batch 128 = 32 × 1 × 4. Target 118 updates, 472 training microbatches per rank.
- All legal anchors, chunk64, fixed FP32 FSDP reduction, AdamW LR 1e-5, warmup/cosine unchanged.
- Restart from initialization. No 8-to-32 checkpoint resume is supported, and r9 had no durable checkpoint at restart preparation.
- Validation and checkpoint every 25 optimizer steps, final checkpoint at 118.
- Different rank grouping/reduction order means no bitwise or exact optimization-trajectory equivalence claim.

Each node writes launcher/environment/worker/exit evidence under `nodes/<node>/`; rank metrics remain under `training/`. JIT caches and composed runtime modules are isolated per node. The actual 32-rank synthetic FSDP gate uses 8 microbatches, accumulation4, 2 updates, the original gradient3%/master-update10% tolerances. Native training then audits loss/gradients, exhaustive coverage, memory, updates and frozen tensors. Gate success does not imply full-training or checkpoint-restore success.

The Muon job `job-3a1b09d413fd-20261105171629` is separate and must be preserved.
