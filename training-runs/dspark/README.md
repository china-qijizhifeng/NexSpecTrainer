# DSpark training runs

归档 55 条记录：训练 35、过拟合实验 4、数据采集 6、训练探针 8、评测 2。

本地2026-11-04至11-06 DSpark正式训练目录及仪电training_runs登记项；采集/探针/评测单独标注。性能microbenchmark和仅CPU回归不算训练。

每个运行一个目录，包含 README、run.json、配置/结果快照和带 SHA256 的来源清单。平台 Job ID 与 run UUID 分开保存；准备、失败、取消、替代运行分别登记。

## 当前训练

各行时间见明细；只把实时平台 Running 的正式训练列在这里。

| Run | 状态 | GPU | LR | Global batch | 步数/预算 | 最近验证 step / acc / loss |
|---|---|---:|---:|---:|---:|---|
| [yicloud-dspark-adamw-lr1e4-2epoch-32gpu-20261106-r1](yicloud-dspark-adamw-lr1e4-2epoch-32gpu-20261106-r1/README.md) | 运行中 | 32 | 0.0001 | 128 | 22/236 | — / — / — |
| [yicloud-dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1](yicloud-dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1/README.md) | 运行中 | 32 | 0.0001 | 512 | 25/30 | 24 / 78.0973% / 0.912996 |
| [yicloud-dspark-adamw-lr2e4-32gpu-20261106-r1](yicloud-dspark-adamw-lr2e4-32gpu-20261106-r1/README.md) | 运行中 | 32 | 0.0002 | 128 | 114/118 | 100 / 78.2545% / 0.912526 |
| [yicloud-dspark-muon-lr1e4-32gpu-20261106-r1](yicloud-dspark-muon-lr1e4-32gpu-20261106-r1/README.md) | 运行中 | 32 | 0.0001 | 128 | 80/118 | 75 / 78.2868% / 0.904063 |

## 已完成训练

| Run | 状态 | GPU | LR | Global batch | 步数/预算 | 最近验证 step / acc / loss |
|---|---|---:|---:|---:|---:|---|
| [dspark-adamw-lr1e4-32gpu-20261106-r1](dspark-adamw-lr1e4-32gpu-20261106-r1/README.md) | 已完成（历史验收） | 32 | 0.0001 | 128 | 118/118 | 100 / 78.2798% / 0.906610 |
| [dspark-all-anchors-32gpu-20261105-r1](dspark-all-anchors-32gpu-20261105-r1/README.md) | 已完成（历史验收） | 32 | 1e-05 | 128 | 118/118 | 100 / 78.0624% / 0.913552 |
| [dspark-denovo-20261104-stage1-r6](dspark-denovo-20261104-stage1-r6/README.md) | 已完成（历史验收） | 8 | 1e-05 | 128 | 118/118 | — / — / — |
| [dspark-overfit10-100ep-20261106-r4](dspark-overfit10-100ep-20261106-r4/README.md) | 已完成（历史验收） | 8 | 0.0001 | 8 | 200/200 | 200 / 97.1550% / 0.027148 |
| [yicloud-dspark-adamw-lr5e5-32gpu-20261106-r1](yicloud-dspark-adamw-lr5e5-32gpu-20261106-r1/README.md) | 已完成（结果passed） | 32 | 5e-05 | 128 | 118/118 | 100 / 78.2439% / 0.905707 |
| [yicloud-dspark-all-anchors-muon32-20261105-r2](yicloud-dspark-all-anchors-muon32-20261105-r2/README.md) | 已完成（结果passed） | 32 | 1e-05 | 128 | 118/118 | 100 / 77.7291% / 0.928088 |
| [yicloud-dspark-denovo-tuned-train-20261105](yicloud-dspark-denovo-tuned-train-20261105/README.md) | 已完成（结果passed） | 8 | 1e-05 | 128 | 796/796 | — / — / — |

## 历史尝试与未提交计划

| Run | 状态 | GPU | LR | Global batch | 步数/预算 | 最近验证 step / acc / loss |
|---|---|---:|---:|---:|---:|---|
| [dspark-adamw-lr1e4-2epoch-32gpu-20261106-r1](dspark-adamw-lr1e4-2epoch-32gpu-20261106-r1/README.md) | 已取消，迁移仪电 | 32 | 0.0001 | 128 | 0/236 | — / — / — |
| [dspark-adamw-lr5e4-32gpu-20261106-r1](dspark-adamw-lr5e4-32gpu-20261106-r1/README.md) | 已取消，迁移仪电 | 32 | 0.0005 | 128 | 0/118 | — / — / — |
| [dspark-all-anchors-20261105](dspark-all-anchors-20261105/README.md) | 失败 | 8 | 1e-05 | 128 | 0/118 | — / — / — |
| [dspark-all-anchors-20261105-r2](dspark-all-anchors-20261105-r2/README.md) | 失败 | 8 | 1e-05 | 128 | 0/118 | — / — / — |
| [dspark-all-anchors-20261105-r3](dspark-all-anchors-20261105-r3/README.md) | 失败 | 8 | 1e-05 | 128 | 0/118 | — / — / — |
| [dspark-all-anchors-20261105-r4](dspark-all-anchors-20261105-r4/README.md) | 失败 | 8 | 1e-05 | 128 | 0/118 | — / — / — |
| [dspark-all-anchors-20261105-r5](dspark-all-anchors-20261105-r5/README.md) | 历史记录，终态未核验 | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-all-anchors-20261105-r6](dspark-all-anchors-20261105-r6/README.md) | 历史记录，终态未核验 | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-all-anchors-20261105-r7](dspark-all-anchors-20261105-r7/README.md) | 历史记录，终态未核验 | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-all-anchors-20261105-r8](dspark-all-anchors-20261105-r8/README.md) | 历史记录，终态未核验 | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-all-anchors-20261105-r9](dspark-all-anchors-20261105-r9/README.md) | 已停止，替代运行另列 | 8 | 1e-05 | 128 | 14/118 | — / — / — |
| [dspark-all-anchors-muon-20261105-r1](dspark-all-anchors-muon-20261105-r1/README.md) | 失败（数值门禁） | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-all-anchors-muon-20261105-r2](dspark-all-anchors-muon-20261105-r2/README.md) | 已停止 | 8 | 1e-05 | 128 | 16/118 | — / — / — |
| [dspark-denovo-20261104-stage1](dspark-denovo-20261104-stage1/README.md) | 历史记录，终态未核验 | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-denovo-20261104-stage1-r2](dspark-denovo-20261104-stage1-r2/README.md) | 历史记录，终态未核验 | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-denovo-20261104-stage1-r3](dspark-denovo-20261104-stage1-r3/README.md) | 失败 | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-denovo-20261104-stage1-r4](dspark-denovo-20261104-stage1-r4/README.md) | 失败 | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-denovo-20261104-stage1-r5](dspark-denovo-20261104-stage1-r5/README.md) | 历史记录，终态未核验 | 8 | 1e-05 | 128 | —/118 | — / — / — |
| [dspark-overfit10-100ep-20261106-r1](dspark-overfit10-100ep-20261106-r1/README.md) | 失败 | 2 | 0.0001 | 2 | —/500 | — / — / — |
| [dspark-overfit10-100ep-20261106-r2](dspark-overfit10-100ep-20261106-r2/README.md) | 失败 | 2 | 0.0001 | 2 | —/500 | — / — / — |
| [dspark-overfit10-100ep-20261106-r3](dspark-overfit10-100ep-20261106-r3/README.md) | 已停止，替换为8卡r4 | 5 | 0.0001 | 5 | —/200 | — / — / — |
| [yicloud-dspark-adamw-lr5e4-32gpu-20261106-r1](yicloud-dspark-adamw-lr5e4-32gpu-20261106-r1/README.md) | 已停止 | 32 | 0.0005 | 128 | 39/118 | 25 / 76.5460% / 1.007512 |
| [yicloud-dspark-all-anchors-muon32-20261105-r1](yicloud-dspark-all-anchors-muon32-20261105-r1/README.md) | 失败 | 32 | 1e-05 | 128 | —/118 | — / — / — |
| [yicloud-dspark-denovo-full-epoch-20261105](yicloud-dspark-denovo-full-epoch-20261105/README.md) | 未提交（prepared） | 8 | — | — | —/— | — / — / — |
| [yicloud-dspark-denovo-full-epoch-moe-r3-20261105](yicloud-dspark-denovo-full-epoch-moe-r3-20261105/README.md) | 未提交（prepared） | 8 | — | — | —/— | — / — / — |
| [yicloud-dspark-denovo-full-epoch-swa-r2-20261105](yicloud-dspark-denovo-full-epoch-swa-r2-20261105/README.md) | 未提交（prepared） | 8 | — | — | —/— | — / — / — |
| [yicloud-dspark-denovo-full-epoch-wide-r4-20261105](yicloud-dspark-denovo-full-epoch-wide-r4-20261105/README.md) | 未提交（prepared） | 8 | — | — | —/— | — / — / — |
| [yicloud-dspark-denovo-full-stream-20261105](yicloud-dspark-denovo-full-stream-20261105/README.md) | 已停止 | 8 | 1e-05 | 128 | —/796 | — / — / — |

## 采集、探针与评测

| Run | 状态 | GPU | LR | Global batch | 步数/预算 | 最近验证 step / acc / loss |
|---|---|---:|---:|---:|---:|---|
| [yicloud-dspark-denovo-full-capture-20261105](yicloud-dspark-denovo-full-capture-20261105/README.md) | 未提交（prepared） | 32 | — | — | —/— | — / — / — |
| [yicloud-dspark-denovo-full-capture-moe-r3-20261105](yicloud-dspark-denovo-full-capture-moe-r3-20261105/README.md) | 未提交（prepared） | 32 | — | — | —/— | — / — / — |
| [yicloud-dspark-denovo-full-capture-swa-r2-20261105](yicloud-dspark-denovo-full-capture-swa-r2-20261105/README.md) | 未提交（prepared） | 32 | — | — | —/— | — / — / — |
| [yicloud-dspark-denovo-full-capture-wide-r4-20261105](yicloud-dspark-denovo-full-capture-wide-r4-20261105/README.md) | 已停止 | 32 | 1e-05 | 32 | —/3 | — / — / — |
| [yicloud-dspark-denovo-longest234k-20261105](yicloud-dspark-denovo-longest234k-20261105/README.md) | 已停止 | 8 | 1e-05 | 8 | —/3 | — / — / — |
| [yicloud-dspark-denovo-longest234k-moe-r3-20261105](yicloud-dspark-denovo-longest234k-moe-r3-20261105/README.md) | 失败 | 8 | 1e-05 | 8 | —/3 | — / — / — |
| [yicloud-dspark-denovo-longest234k-swa-r2-20261105](yicloud-dspark-denovo-longest234k-swa-r2-20261105/README.md) | 失败 | 8 | 1e-05 | 8 | —/3 | — / — / — |
| [yicloud-dspark-denovo-longest234k-wide-r4-20261105](yicloud-dspark-denovo-longest234k-wide-r4-20261105/README.md) | 平台成功 | 8 | 1e-05 | 8 | —/3 | — / — / — |
| [yicloud-dspark-denovo-paired-eval-20261105](yicloud-dspark-denovo-paired-eval-20261105/README.md) | 平台成功 | 8 | 1e-05 | 128 | —/796 | — / — / — |
| [yicloud-dspark-denovo-pilot8-20261105](yicloud-dspark-denovo-pilot8-20261105/README.md) | 已停止 | 8 | — | — | —/— | — / — / — |
| [yicloud-dspark-denovo-pilot8-configfix-20261105](yicloud-dspark-denovo-pilot8-configfix-20261105/README.md) | 未提交（prepared） | 8 | — | — | —/— | — / — / — |
| [yicloud-dspark-denovo-pilot8-configfix-r2-20261105](yicloud-dspark-denovo-pilot8-configfix-r2-20261105/README.md) | 平台成功 | 8 | 1e-05 | 8 | 3/3 | — / — / — |
| [yicloud-dspark-denovo-pilot8-tagged-20261105](yicloud-dspark-denovo-pilot8-tagged-20261105/README.md) | 失败 | 8 | 1e-05 | 8 | —/3 | — / — / — |
| [yicloud-dspark-denovo-tuned-capture-20261105](yicloud-dspark-denovo-tuned-capture-20261105/README.md) | 平台成功 | 32 | 1e-05 | 32 | —/3 | — / — / — |
| [yicloud-dspark-full-trajectory-capture-20261105](yicloud-dspark-full-trajectory-capture-20261105/README.md) | 平台成功 | 8 | 1e-05 | 8 | —/3 | — / — / — |
| [dspark-native-mtp-all-eval-20261106-r1](dspark-native-mtp-all-eval-20261106-r1/README.md) | 评测结果已验收；平台Failed | 8 | — | — | 0/0 | 0 / 77.4319% / 0.944420 |

## 比较口径

- BP 与仪电特征独立采集、硬件不同；跨平台对照不能仅归因于学习率。
- 近期全量 anchor 调参组：1,326 train traces、17 validation traces；15,104 train windows 含 51 个既有重复 padding；32 rank 验证补齐为 160 windows。
- Batch 512 组补齐至 15,360 windows，共 30 步；应按消费窗口量或 epoch 比较，不能与 batch 128 同步数直接对比。
- 早期 stage1 为采样 anchor；全量 4,188 trace / 796 步、10 样本过拟合和未微调 MTP 评测另有口径，不能并入同一排名。
- 日志中的准确率是离线指标，不等同于 serving 接受率或任务分数。

## 更新

修改对应目录的 `run.json`，添加带检查时间的证据并更新 `evidence-manifest.json`，然后运行：

```bash
python3 training-runs/dspark/refresh_index.py
```

只读监控按精确 Job ID 和 UUID 更新；不要用启动时的 run state=verified 代替平台状态。checkpoint、特征、完整日志和运行脚本继续保留在原目录，避免影响仍运行的任务。

[范围与清单](inventory.json) · [来源聊天](codex://threads/01a1b06c-ef27-73b0-a81b-d4e54a0788bc)
