# dspark-native-mtp-all-eval-20261106-r1

状态：**评测结果已验收；平台Failed**。
实时检查时间：—；归档时间：2026-11-06T20:57:55+08:00。

本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。

| 字段 | 记录 |
|---|---|
| 类型 | 评测 |
| 平台 | BP / H100 |
| SSH | linlizhi-bp-dev |
| Job ID | job-0828a6b23265-20261106185151 |
| Alias | dspark-native-mtp-all-eval-20261106-r1 |
| UUID | — |
| 已完成步数 | 0 |
| 远端运行目录 | /gpfs/users/linlizhi/dspark-native-mtp-all-eval-20261106-r1 |
| Checkpoint 目录 | — |

## 配置

| 字段 | 值 |
|---|---|
| gpus | 8 |
| max_steps | 0 |

## 最近一次验证

step **0**；loss **0.94442**；accuracy **0.774319**（0–1）。
验证步数与训练最终步数可能不同。过拟合、稀疏 anchor、全量 anchor、独立采集的 BP/仪电特征不能直接混排。

## 证据

- [结构化记录](run.json)
- [来源与 SHA256 清单](evidence-manifest.json)
- 原始本地目录：`/Users/shom/Downloads/dspark-native-mtp-all-eval-20261106-r1`
- [evidence/RUN.md](evidence/RUN.md)
- [evidence/FINAL.md](evidence/FINAL.md)
- [evidence/final-verification.json](evidence/final-verification.json)
- [evidence/aligned-result.json](evidence/aligned-result.json)

## 说明

- GPU评测完成后CPU汇总因ID命名空间失败；CPU修复重算通过。8卡前向重建32rank权重，非32卡逐位复现。

证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。
