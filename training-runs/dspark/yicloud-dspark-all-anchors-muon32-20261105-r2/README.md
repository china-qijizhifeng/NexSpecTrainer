# yicloud-dspark-all-anchors-muon32-20261105-r2

状态：**已完成（结果passed）**。
实时检查时间：2026-11-06T20:56:06.783262+08:00；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 训练 |
| 平台 | 仪电 / H200 |
| SSH | yicloud-dev |
| Job ID | j-1793880042-617264 |
| Alias | dspark-all-anchors-muon32-20261105-r2 |
| UUID | 69e9394b-b815-47d2-b4b3-111784cd0d56 |
| 已完成步数 | 118 |
| 远端运行目录 | /mnt/work/shom/training_runs/dspark-all-anchors-muon32-20261105-r2 |
| Checkpoint 目录 | /mnt/work/shom/training_runs/dspark-all-anchors-muon32-20261105-r2/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | v41_flash_muon |
| learning_rate | 1e-05 |
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
| feature_path | /mnt/work/shom/dspark-muon32-20261105-r1/train-features |
| validation_feature_path | /mnt/work/shom/dspark-muon32-20261105-r1/validation-features |

## 最近一次验证

step **100**；loss **0.928088360990563**；accuracy **0.7772910056175979**（0–1）。
验证步数与训练最终步数可能不同。过拟合、稀疏 anchor、全量 anchor、独立采集的 BP/仪电特征不能直接混排。

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/yicloud-dspark-muon32-20261105-r1`
- [config/0-config.yaml](config/0-config.yaml)
- [config/1-recipe.yaml](config/1-recipe.yaml)
- [evidence/snapshot.json](evidence/snapshot.json)
- [evidence/RUN.md](evidence/RUN.md)
- [evidence/STATUS-20261106.md](evidence/STATUS-20261106.md)

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
