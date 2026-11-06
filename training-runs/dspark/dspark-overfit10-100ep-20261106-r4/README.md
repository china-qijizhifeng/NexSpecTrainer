# dspark-overfit10-100ep-20261106-r4

状态：**已完成（历史验收）**。
实时检查时间：—；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 过拟合实验 |
| 平台 | BP / H100 |
| SSH | linlizhi-bp-dev |
| Job ID | job-01c79ce0f5e7-20261106162729 |
| Alias | dspark-overfit10-100ep-20261106-r4 |
| UUID | — |
| 已完成步数 | 200 |
| 远端运行目录 | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r4 |
| Checkpoint 目录 | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r4/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | AdamW (default) |
| learning_rate | 0.0001 |
| epochs | 100 |
| max_steps | 200 |
| total_steps | 200 |
| gpus | 8 |
| micro_batch | 1 |
| accumulation_steps | 1 |
| global_batch | 8 |
| anchor_sampling | all |
| anchor_chunk | 64 |
| seed | 42 |
| eval_interval | 10 |
| save_interval | 100 |
| feature_path | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r4/features |
| validation_feature_path | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r4/features |

## 最近一次验证

step **200**；loss **0.027147766799524166**；accuracy **0.9715498938428875**（0–1）。
验证步数与训练最终步数可能不同。过拟合、稀疏 anchor、全量 anchor、独立采集的 BP/仪电特征不能直接混排。

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/dspark-overfit10-100ep-20261106-r4`
- [evidence/RUN.md](evidence/RUN.md)
- [evidence/FINAL.md](evidence/FINAL.md)
- [evidence/train.yaml](evidence/train.yaml)
- [evidence/submitted-job.json](evidence/submitted-job.json)
- [evidence/job-identity.json](evidence/job-identity.json)
- [evidence/final-verification.json](evidence/final-verification.json)
- [evidence/RESULT.json](evidence/RESULT.json)
- [evidence/result.json](evidence/result.json)

## 说明

- Global batch=8表示8个计算槽位；每epoch两步合计10个真实样本+6个零监督padding槽位，不是16个真实样本。

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
