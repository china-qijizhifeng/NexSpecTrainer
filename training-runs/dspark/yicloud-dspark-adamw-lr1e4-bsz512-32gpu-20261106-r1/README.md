# yicloud-dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1

状态：**运行中**。
实时检查时间：2026-11-06T20:56:02.900109+08:00；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 训练 |
| 平台 | 仪电 / H200 |
| SSH | yicloud-dev |
| Job ID | j-1793949241-199779 |
| Alias | dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1 |
| UUID | 8d089b34-6ca3-4af5-80da-f57bfc72be0d |
| 已完成步数 | 25 |
| 远端运行目录 | /mnt/work/shom/training_runs/dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1 |
| Checkpoint 目录 | /mnt/work/shom/training_runs/dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | AdamW (default) |
| learning_rate | 0.0001 |
| epochs | 1 |
| max_steps | 30 |
| total_steps | 30 |
| gpus | 32 |
| micro_batch | 1 |
| accumulation_steps | 16 |
| global_batch | 512 |
| anchor_sampling | all |
| anchor_chunk | 64 |
| seed | 42 |
| eval_interval | 6 |
| save_interval | 6 |
| feature_path | /mnt/work/shom/dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1/inputs/train-features |
| validation_feature_path | /mnt/work/shom/dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1/inputs/validation-features |

## 最近一次验证

step **24**；loss **0.912995667485424**；accuracy **0.7809727485043418**（0–1）。
验证步数与训练最终步数可能不同。过拟合、稀疏 anchor、全量 anchor、独立采集的 BP/仪电特征不能直接混排。

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/yicloud-dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1`
- [config/0-config.yaml](config/0-config.yaml)
- [config/1-recipe.yaml](config/1-recipe.yaml)
- [evidence/snapshot.json](evidence/snapshot.json)
- [evidence/RUN.md](evidence/RUN.md)
- [evidence/data-plan.json](evidence/data-plan.json)

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
