# AdamW peak learning rate comparison

User requested peak LR 1e-4. Baseline dspark-all-anchors-32gpu-20261105-r1 completed 118 steps at peak 1e-5. Fresh run from the same initialization and seed42; no checkpoint resume.

32 H100, global batch128, accumulation4, 1326 traces, 118 steps, warmup3, cosine decay, all legal anchors, chunk64, fixed FP32 reduction, same loss weights and CPU optimizer offload. Validation/checkpoint every25 steps; final checkpoint118. Only learning_rate changes as a training parameter. Isolated output/cache paths and rendezvous ports29852/29853.

LR1e-5 was inherited from original stage1 r6 (which itself cites r4) under the instruction to keep optimization parameters unchanged. No explicit user choice of 1e-5 or LR tuning evidence has been found.
