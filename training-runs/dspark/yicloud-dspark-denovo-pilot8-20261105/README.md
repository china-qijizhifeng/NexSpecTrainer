# yicloud-dspark-denovo-pilot8-20261105

状态：**已停止**。
实时检查时间：2026-11-06T20:56:10.387327+08:00；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 训练探针 |
| 平台 | 仪电 / H200 |
| SSH | yicloud-dev |
| Job ID | j-1793809241-735498 |
| Alias | dspark-denovo-pilot8-20261105 |
| UUID | 7e25e9ef-aa07-44a6-898b-4b5019f5b37e |
| 已完成步数 | — |
| 远端运行目录 | /mnt/work/shom/training_runs/dspark-denovo-pilot8-20261105 |
| Checkpoint 目录 | /mnt/work/shom/training_runs/dspark-denovo-pilot8-20261105/training |

## 配置

| 字段 | 值 |
|---|---|
| gpus | 8 |

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/yicloud-dspark-migration-20261104`
- [evidence/snapshot.json](evidence/snapshot.json)

## 说明

- 未发现落盘resolved config；不得将prepared计划认作已训练。源码commit和执行入口已记录。

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
