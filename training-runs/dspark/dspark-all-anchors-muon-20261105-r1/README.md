# dspark-all-anchors-muon-20261105-r1

状态：**失败（数值门禁）**。
实时检查时间：—；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 训练 |
| 平台 | BP / H100 |
| SSH | linlizhi-bp-dev |
| Job ID | job-bf9be038498b-20261105170339 |
| Alias | dspark-all-anchors-muon-20261105-r1 |
| UUID | — |
| 已完成步数 | — |
| 远端运行目录 | /gpfs/users/linlizhi/dspark-all-anchors-muon-20261105-r1 |
| Checkpoint 目录 | /gpfs/users/linlizhi/dspark-all-anchors-muon-20261105-r1/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | v41_flash_muon |
| learning_rate | 1e-05 |
| epochs | 1 |
| max_steps | 118 |
| total_steps | 118 |
| gpus | 8 |
| micro_batch | 1 |
| accumulation_steps | 16 |
| global_batch | 128 |
| anchor_sampling | all |
| anchor_chunk | 64 |
| seed | 42 |
| eval_interval | 25 |
| save_interval | 25 |
| feature_path | /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/train-features |
| validation_feature_path | /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/validation-features |

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/dspark-all-anchors-muon-20261105-r1`
- [evidence/RUN.md](evidence/RUN.md)
- [evidence/train.yaml](evidence/train.yaml)
- [evidence/source-provenance.json](evidence/source-provenance.json)
- [evidence/runtime-provenance.json](evidence/runtime-provenance.json)
- [evidence/submitted-job.json](evidence/submitted-job.json)

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
