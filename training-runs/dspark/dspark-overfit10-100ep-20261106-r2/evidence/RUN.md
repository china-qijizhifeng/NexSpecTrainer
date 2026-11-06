# DSpark ten-sample memorization experiment

Reference: SpecForge `scripts/gates/README.md`, fixed tiny dataset repeated over epochs.
Model: existing DeepSeek-V4.1-Flash-1017 native DSpark, fresh load of native target MTP weights (seed 42), no fine-tuned checkpoint resume.
Data unit: one first <=4096-token cached feature window from each of ten distinct DeNovo training traces, selected with Python random seed 42; existing offline features retained. This is not ten complete multi-call trajectories, and these traces were not regenerated at temperature zero for this experiment.
Training: 100 epochs, two GPUs, microbatch 1/rank, accumulation 1, global batch 2, five optimizer steps/epoch, 500 optimizer steps. Every selected window must be consumed exactly 100 times globally; no distributed sampler padding is necessary.
AdamW peak LR 1e-4, warmup 0.03, cosine schedule, norm clip 1, same CE/L1/confidence loss weights, all legal anchors, chunk64, gradient checkpointing enabled, FP32 reduction and optimizer CPU offload.
Evaluation: exact same ten samples before training and every five epochs (25 optimizer steps). Save per-sample ratios, full-set accuracy/loss and draft-position/walk diagnostics. >=99% same-set target accuracy is the declared memorization criterion; report partial convergence if below. Hard-label walk metrics are offline diagnostics, not measured SGLang serving acceptance.
Checkpoints: epoch50 and epoch100; validate completion/metadata, do not claim restore without testing it.
Platform: BP Northjob project WeaverRSI, queue proj-32, pinned existing v1.14.92 image, two GPUs, 32 CPU, 256Gi memory, independent output directory.
Remote: /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r2

Communication fix: first baseline failed during NCCL RDMA plugin initialization because this single-node job has no RDMA device. Same-image real 2-GPU all_gather/all_reduce reproduces double free under default plugin and passes after only NCCL_NET_PLUGIN=none. The revised job gates training on both regression outcomes. Job ID job-7a794b903336-20261106161542.
