# AdamW peak LR 5e-4 — YiCloud migration

User requested moving the queued BP 5e-4 run to YiCloud on 2026-11-06. BP job `job-ba62ea5f8a6f-20261106100111` was Pending with no training; cancelled by exact alias and read back absent. BP 1e-4 remains independent.

YiCloud JobId: `j-1793931458-946720`; UUID: `b756ae89-74cf-44f7-8f57-f41de963f6c2`.
Run: `/mnt/work/shom/training_runs/dspark-adamw-lr5e4-32gpu-20261106-r1`.
Work: `/mnt/work/shom/dspark-adamw-lr5e4-32gpu-20261106-r1`.
SSH: `yicloud-dev`. Project `fdj-infra`, quota `nex_bp`, 4×8 H200, priority 9. Snapshot quota before submission showed 136 free GPUs; no unrelated jobs were changed.

Recipe: AdamW, peak LR5e-4, seed42 fresh initialization, warmup3/cosine118 (including existing small step4 overshoot), 118 updates and 472 train micros/rank, global batch128=32×1×4, FULL_SHARD, CPU optimizer offload, all legal anchors, chunk64, fixed FP32 reduction, evaluation and checkpoint every25 steps. No terminal step118 evaluation configured.

All1395 original source files verified byte-identical to BP source manifest. Isolated launch adapter committed as `9499520cbb4a8fbc1c6c70811bd565509f8268b7`; no trainer/optimizer/schedule implementation changed. Image pinned to sha256:f2e360fea06554224cf3070d8c5c65bba15762ffad7cc6a7f7d7a3719297c960. Four resolved node training configs exactly equal BP5e-4 recipe.

Inputs reuse the previous verified YiCloud mapping: original1326 train and17 validation traces,15189 original windows with matching full token/mask hashes,15104 training windows including51 explicit padding windows,136 unpadded validation windows. Independently captured BF16 features and H200 hardware differ from BP; this is NOT a strict LR-only input-tensor-controlled comparison. Anchor duplicate count0 is per-window anchor coverage, not a claim that dataset padding is absent.

Startup: four real-image API and native train/eval input preflights passed; all32 ranks passed the AdamW streamed-versus-whole FSDP gate. Actual training progress is in startup-status.json; a Running job alone is not optimizer-update acceptance. No checkpoint restore or model quality improvement is claimed.

Read-only helpers at work root: `api.py GetJob`, `api.py ListJobReplicas`, `read_status.py`.
Terminal reconciler PID60466 uses exact run identity, no resubmission. Shared app heartbeat `adamw-lr-1e-4` now follows BP1e-4 plus this YiCloud5e-4; notify first real update/first validation/errors/completion, quiet routine progress. Cancelled BP5e-4 absent is intentional.
