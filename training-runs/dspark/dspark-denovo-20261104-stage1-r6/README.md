# dspark-denovo-20261104-stage1-r6

状态：**已完成（历史验收）**。
实时检查时间：—；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 训练 |
| 平台 | BP / H100 |
| SSH | linlizhi-bp-dev |
| Job ID | job-261dab7030ba-20261104064734 |
| Alias | denovo-specforge-stage1-train-20261104-r6 |
| UUID | — |
| 已完成步数 | 118 |
| 远端运行目录 | /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6 |
| Checkpoint 目录 | /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | AdamW (default) |
| learning_rate | 1e-05 |
| epochs | 1 |
| max_steps | 118 |
| total_steps | 118 |
| gpus | 8 |
| micro_batch | 1 |
| accumulation_steps | 16 |
| global_batch | 128 |
| anchor_sampling | config default |
| anchor_chunk | — |
| seed | 42 |
| eval_interval | 25 |
| save_interval | 25 |
| feature_path | /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/train-features |
| validation_feature_path | /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/validation-features |

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/dspark-denovo-20261104-stage1-r6`
- [evidence/RUN.md](evidence/RUN.md)
- [evidence/train.yaml](evidence/train.yaml)
- [evidence/source-provenance.json](evidence/source-provenance.json)
- [evidence/training-job.json](evidence/training-job.json)
- [evidence/training-terminal.json](evidence/training-terminal.json)
- [evidence/stage1-final-acceptance.json](evidence/stage1-final-acceptance.json)

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
