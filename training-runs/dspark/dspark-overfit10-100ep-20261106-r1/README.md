# dspark-overfit10-100ep-20261106-r1

状态：**失败**。
实时检查时间：—；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 过拟合实验 |
| 平台 | BP / H100 |
| SSH | linlizhi-bp-dev |
| Job ID | job-7cb0017758b3-20261106155850 |
| Alias | dspark-overfit10-100ep-20261106-r1 |
| UUID | — |
| 已完成步数 | — |
| 远端运行目录 | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r1 |
| Checkpoint 目录 | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r1/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | AdamW (default) |
| learning_rate | 0.0001 |
| epochs | 100 |
| max_steps | 500 |
| total_steps | 500 |
| gpus | 2 |
| micro_batch | 1 |
| accumulation_steps | 1 |
| global_batch | 2 |
| anchor_sampling | all |
| anchor_chunk | 64 |
| seed | 42 |
| eval_interval | 25 |
| save_interval | 250 |
| feature_path | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r1/features |
| validation_feature_path | /gpfs/users/linlizhi/dspark-overfit10-100ep-20261106-r1/features |

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/dspark-overfit10-100ep-20261106-r1`
- [evidence/RUN.md](evidence/RUN.md)
- [evidence/DIAGNOSIS.md](evidence/DIAGNOSIS.md)
- [evidence/train.yaml](evidence/train.yaml)
- [evidence/source-provenance.json](evidence/source-provenance.json)
- [evidence/submitted-job.json](evidence/submitted-job.json)
- [evidence/monitor-terminal.json](evidence/monitor-terminal.json)
- [evidence/RESULT.json](evidence/RESULT.json)
- [evidence/result.json](evidence/result.json)

## 说明

- 历史monitor终态详见 evidence/monitor-terminal.json；不从提交状态推断当前状态。

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
