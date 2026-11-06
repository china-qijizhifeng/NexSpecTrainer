# YiCloud AdamW peak LR2e-4

User requested stopping5e-4 and replacing it with2e-4 on2026-11-06. Old job j-1793931458-946720 stopped after39 optimizer updates and156 completed train microbatches on all32 ranks. All4 platform replicas confirmedStopped; checkpoint25 retained (128 shards). This was a requested cancellation, not a training failure.

New job: j-1793940402-922604; UUID0008929f-5488-4d37-8f2c-cc2d111fc6c7. Submitted2026-11-06 12:46:42 Beijing.
SSH yicloud-dev. Run /mnt/work/shom/training_runs/dspark-adamw-lr2e4-32gpu-20261106-r1. Work /mnt/work/shom/dspark-adamw-lr2e4-32gpu-20261106-r1.
Source commit b2e6e77f8a45ccc7d14cd3b3c1f85a2c37b41f4d; isolated clone of prior run, changed only recipe peak learning_rate5e-4 to2e-4 and matching launch assertion. Original trainer/model/optimizer code unchanged.

4×8 H200,fdj-infra/nex_bp,priority9,pinned image sha256:f2e360fea06554224cf3070d8c5c65bba15762ffad7cc6a7f7d7a3719297c960. Same verified YiCloud1326 training traces and17 validation traces,input feature tensors and padding scheme as previous5e-4. Fresh seed42 initialization,no checkpoint resume. AdamW,118 steps,472 train micros/rank,globalbatch128,accum4,warmup3/cosine118,all legal anchors,chunk64,fixed FP32 reduction,FULL_SHARD,CPU optimizer offload. Evaluation/checkpoint every25steps;finalcheckpoint118 without finalevaluation.

All4 replicasRunning at initialreadback. Actual gate and optimizer-update status retained in startup-status.json; Running alone is not training acceptance. Exact-run terminal reconciler PID73332, no resubmission. Existing app heartbeat adamw-lr-1e-4 now tracks this2e-4 job and retains BP1e-5/1e-4 and cancelledYiCloud5e-4 as comparison evidence.

BP H100 inputs were independently captured from YiCloud H200 inputs; cross-platform comparisons are not strict LR-only comparisons. No checkpoint restore or inference-quality improvement has been tested.

Startup verification: all4 environment and native input checks passed; all32 ranks passed numerical FSDP/AdamW gate. Four resolved node training configs differ from5e-4 only in learning_rate. Training is initializing,0 real optimizer updates at verification; see config-parity.json/startup-status.json.
