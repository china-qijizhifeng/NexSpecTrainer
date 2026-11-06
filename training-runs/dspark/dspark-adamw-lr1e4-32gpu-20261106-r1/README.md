# dspark-adamw-lr1e4-32gpu-20261106-r1

状态：**已完成（历史验收）**。
实时检查时间：—；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 训练 |
| 平台 | BP / H100 |
| SSH | linlizhi-bp-dev |
| Job ID | job-d3b052f034c2-20261106043222 |
| Alias | dspark-adamw-lr1e4-32gpu-20261106-r1 |
| UUID | — |
| 已完成步数 | 118 |
| 远端运行目录 | /gpfs/users/linlizhi/dspark-adamw-lr1e4-32gpu-20261106-r1 |
| Checkpoint 目录 | /gpfs/users/linlizhi/dspark-adamw-lr1e4-32gpu-20261106-r1/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | AdamW (default) |
| learning_rate | 0.0001 |
| epochs | 1 |
| max_steps | 118 |
| total_steps | 118 |
| gpus | 32 |
| micro_batch | 1 |
| accumulation_steps | 4 |
| global_batch | 128 |
| anchor_sampling | all |
| anchor_chunk | 64 |
| seed | 42 |
| eval_interval | 25 |
| save_interval | 25 |
| feature_path | /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/train-features |
| validation_feature_path | /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/validation-features |

## 最近一次验证

step **100**；loss **0.906610415719248**；accuracy **0.7827979223880724**（0–1）。
验证步数与训练最终步数可能不同。过拟合、稀疏 anchor、全量 anchor、独立采集的 BP/仪电特征不能直接混排。

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/dspark-adamw-lr1e4-32gpu-20261106-r1`
- [evidence/RUN.md](evidence/RUN.md)
- [evidence/FINAL.md](evidence/FINAL.md)
- [evidence/train.yaml](evidence/train.yaml)
- [evidence/source-provenance.json](evidence/source-provenance.json)
- [evidence/submitted-job.json](evidence/submitted-job.json)
- [evidence/terminal-verification.json](evidence/terminal-verification.json)
- [evidence/curves.json](evidence/curves.json)

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
