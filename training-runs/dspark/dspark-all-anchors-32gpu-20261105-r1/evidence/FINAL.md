# AdamW 32 GPU training completed

Job: job-0a081d7a6923-20261105192255. Terminal monitor recorded Completed with all four node exits 0; exact-alias API readback is absent after existing monitor cleanup.

All 32 rank completion receipts passed: 118 optimizer steps and 472 training microbatches each; frozen parameters unchanged. All recorded losses and gradient norms finite. Per-input legal anchor coverage complete with zero anchor duplicates. Total eligible anchor exposures: 40,097,642; target exposures: 199,997,123.

Median step interval 216.90 seconds. Training start to final optimizer update elapsed 7.236 hours, including intervening evaluation/checkpoint overhead. Peak allocated memory 33.294 GiB; peak reserved memory 74.184 GiB per GPU.

Final checkpoint: /gpfs/users/linlizhi/dspark-all-anchors-32gpu-20261105-r1/training/denovo-v41-all-anchors-step118

Verified step118 training_state.json, NFC publication receipt (snapshot 108892f0e1b74ad0d0616fe12028a842), .metadata, 128 nonempty shards covering 32 ranks, and common.pt SHA256 matching training_state.json. Checkpoint restore has not been tested.

Latest/best recorded validation was step100: avg_loss 0.9135519767141073, avg_acc 0.7806236773475701. No step118 evaluation or model-quality improvement is claimed.

Automation dspark-anchor-r9 deleted after terminal verification. Independent YiCloud Muon training untouched.
