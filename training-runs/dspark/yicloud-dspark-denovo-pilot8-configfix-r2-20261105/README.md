# yicloud-dspark-denovo-pilot8-configfix-r2-20261105

状态：**平台成功**。
实时检查时间：2026-11-06T20:56:11.081792+08:00；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 训练探针 |
| 平台 | 仪电 / H200 |
| SSH | yicloud-dev |
| Job ID | j-1793810576-946821 |
| Alias | dspark-denovo-pilot8-configfix-r2-20261105 |
| UUID | 8c73d3f0-2813-4fcb-b658-113d8cbef0dc |
| 已完成步数 | 3 |
| 远端运行目录 | /mnt/work/shom/training_runs/dspark-denovo-pilot8-configfix-r2-20261105 |
| Checkpoint 目录 | /mnt/work/shom/training_runs/dspark-denovo-pilot8-configfix-r2-20261105/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | AdamW (default) |
| learning_rate | 1e-05 |
| epochs | 4 |
| max_steps | 3 |
| total_steps | 3 |
| gpus | 8 |
| micro_batch | 1 |
| accumulation_steps | 1 |
| global_batch | 8 |
| anchor_sampling | config default |
| anchor_chunk | — |
| seed | 42 |
| eval_interval | 0 |
| save_interval | 0 |
| feature_path | /mnt/work/shom/dspark-v41-denovo-migration-20261104/pilot-features/train |
| validation_feature_path |  |

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/yicloud-dspark-migration-20261104`
- [config/0-config.yaml](config/0-config.yaml)
- [evidence/snapshot.json](evidence/snapshot.json)

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
