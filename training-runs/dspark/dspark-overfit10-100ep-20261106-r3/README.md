# dspark-overfit10-100ep-20261106-r3

状态：**已停止，替换为8卡r4**。
实时检查时间：—；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 过拟合实验 |
| 平台 | BP / H100 |
| SSH | linlizhi-bp-dev |
| Job ID | job-5c54beb13e96-20261106162221 |
| Alias | dspark-overfit10-100ep-20261106-r3 |
| UUID | — |
| 已完成步数 | — |
| 远端运行目录 | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r3 |
| Checkpoint 目录 | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r3/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | AdamW (default) |
| learning_rate | 0.0001 |
| epochs | 100 |
| max_steps | 200 |
| total_steps | 200 |
| gpus | 5 |
| micro_batch | 1 |
| accumulation_steps | 1 |
| global_batch | 5 |
| anchor_sampling | all |
| anchor_chunk | 64 |
| seed | 42 |
| eval_interval | 10 |
| save_interval | 100 |
| feature_path | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r3/features |
| validation_feature_path | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r3/features |

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/dspark-overfit10-100ep-20261106-r3`
- [evidence/RUN.md](evidence/RUN.md)
- [evidence/train.yaml](evidence/train.yaml)
- [evidence/submitted-job.json](evidence/submitted-job.json)

## 说明

- 替代运行：dspark-overfit10-100ep-20261106-r4；旧目录与证据保留。

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
