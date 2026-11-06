# Muon + AdamW 全量 anchor 训练已启动

当前作业 `job-3a1b09d413fd-20261105171629`，alias `dspark-all-anchors-muon-20261105-r2`，8 H100。
固定源码 `e480b1a28167ae3165e77b9621712427e871dddc`（PR #15 + #16）。

复用 1,326 条 trace、全量合法 anchors、chunk64；118 optimizer steps、batch1/rank、accumulation16、LR1e-5、warmup3、cosine，seed42。使用 v41_flash_muon：矩阵 Muon + 指定参数 AdamW，按 PR 的完整配方；原 r9 AdamW 作业保留。

截至北京时间 2026-11-05 17:23，全部 8 ranks 完成前 2 个正式微批次，loss 有限，覆盖匹配、重复数0。第二微批次最慢44.511秒、峰值44.318GiB。当前正式 optimizer 更新0次，首步仍在积累；不把这写成真实全模型 Muon 更新或收敛验收。

GPU 小模型组合门禁32微批次/2更新通过：相同梯度完整矩阵与分片 Muon+AdamW 更新差异0。整图 vs 流式 backward 的梯度最大相对L2为0.41999%，其 master update 差异13.912%未通过旧等价门限。r1在该门禁失败后退出，保留证据并清理；CPU实测回放确认非线性Muon对两种BF16梯度输入产生不同方向，r2使用相同梯度oracle验收实现，未声称跨路径训练轨迹等价。详情 DIAGNOSIS.md。

正式训练逐微批次核对mask/覆盖/finite/显存，逐更新记录norm/LR/optimizer时间；第17微批次另保存首次真实更新及post-update backward的startup-acceptance。后台终态清理monitor已启动，只清理本次精确identity，不重投。完整训练与checkpoint恢复、下游质量尚未验收。
