# yicloud-dspark-denovo-full-capture-wide-r4-20261105

状态：**已停止**。
实时检查时间：2026-11-06T20:56:07.922398+08:00；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 数据采集 |
| 平台 | 仪电 / H200 |
| SSH | yicloud-dev |
| Job ID | j-1793819908-584020 |
| Alias | dspark-denovo-full-capture-wide-r4-20261105 |
| UUID | 05a1489a-0f70-46ad-a1d0-86cde069b664 |
| 已完成步数 | — |
| 远端运行目录 | /mnt/work/shom/training_runs/dspark-denovo-full-capture-wide-r4-20261105 |
| Checkpoint 目录 | /mnt/work/shom/training_runs/dspark-denovo-full-capture-wide-r4-20261105/training |

## 配置

| 字段 | 值 |
|---|---|
| optimizer | AdamW (default) |
| learning_rate | 1e-05 |
| epochs | 1 |
| max_steps | 3 |
| total_steps | 3 |
| gpus | 32 |
| micro_batch | 1 |
| accumulation_steps | 1 |
| global_batch | 32 |
| anchor_sampling | config default |
| anchor_chunk | — |
| seed | 42 |
| eval_interval | — |
| save_interval | 0 |
| feature_path | /mnt/work/shom/training_runs/dspark-denovo-full-capture-wide-r4-20261105/capture-w0/features |
| validation_feature_path | — |

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/yicloud-dspark-migration-20261104`
- [config/0-config.yaml](config/0-config.yaml)
- [evidence/snapshot.json](evidence/snapshot.json)

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
