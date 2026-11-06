# 仪电 AdamW LR1e-4 Global Batch512

用户要求启动 AdamW LR1e-4、bsz512。解释为全局batch512；沿用32H200，每卡batch1，梯度累积16，seed42 fresh初始化，全量anchors、chunk64、FP32 reduction、FULL_SHARD、CPU optimizer offload。相同1326训练trace、17验证trace；原15053窗口+51既有padding，再显式补齐256窗口至15360（不删减原始样本），1epoch=30updates，每rank480微批。warmup1步/cosine30，每6步验证保存。额外padding清单见data-plan.json。验证集仍为原136窗口、32rank补齐160。

源码基于已验证仪电 AdamW2e-4 b2e6e77f8a45ccc7d14cd3b3c1f85a2c37b41f4d，仅改实验配方、预算验收常量和GPU小模型门禁的累积16；没有修改训练器或优化器业务实现。用户已明确确认保持1个epoch、约30步。

UUID：8d089b34-6ca3-4af5-80da-f57bfc72be0d。
远端工作根：/mnt/work/shom/dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1。
远端运行根：/mnt/work/shom/training_runs/dspark-adamw-lr1e4-bsz512-32gpu-20261106-r1。

与batch128对比应按消费窗口量或epoch进度，不直接按optimizer step比较。warmup与评测间隔随预算调整，额外padding带来小量数据重复，报告时保留这些差异。尚无新分数。

已提交 JobId `j-1793949241-199779`。源码 `4fdccaefb630db9b7eeac4e8901461bc49819a32`，终态 reconciler PID85465，只跟踪本作业、不重投。
