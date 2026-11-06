# YiCloud AdamW peak LR5e-5

Submitted2026-11-06 13:25:02 Beijing at user request, alongside existing2e-4.
JobId j-1793942702-213881; UUID c363a7e3-a7cf-46a6-8cc2-6897b0dfda40.
Run /mnt/work/shom/training_runs/dspark-adamw-lr5e5-32gpu-20261106-r1.
Work /mnt/work/shom/dspark-adamw-lr5e5-32gpu-20261106-r1; SSH yicloud-dev.

4x8H200,fdj-infra/nex_bp,priority9,pinned trainer image from2e-4. Source commit dd680101ad3cd238a84831abca9e2520c7679965,cloned from2e-4 experimental commit b2e6e77f8a45ccc7d14cd3b3c1f85a2c37b41f4d. Only changes are peak LR2e-4 to5e-5 in recipe and matching assertion. Keep this experimental source fixed independently of GitHubmain so comparison uses identical runtime code.

Same YiCloud feature files:1326 train/17validation traces,15104 train windows with51explicit padding windows,136validation windows before distributed padding. Fresh initialization seed42,no resume.118updates/472micros per rank,globalbatch128,accum4,allanchors,chunk64,FP32reduction,FULL_SHARD,CPU optimizer offload,warmup3/cosine118. No schedule implementation change. Checkpoint/evaluation every25steps,checkpoint118without finalevaluation.

Submission readback recorded platformStarting with4assigned nodes. Actual startup must pass4native input/environment checks,32rank numerical gate,and real optimizer updates. Running alone does not establish update acceptance. Exact-run reconciler PID78067;combined heartbeat adamw-lr-1e-4 tracks2e-4and5e-5. No other jobs stopped or changed. Historical BPcomparisons have independent feature capture/hardware differences. No checkpoint restore or model quality claim.

Startup checks passed on all4nodes;32rank numeric gate passed. All4 resolved training configs differ from2e-4 only in learning_rate. Formal training initialization in progress;0 actual updates at verification. See config-parity.json and startup-status.json.
