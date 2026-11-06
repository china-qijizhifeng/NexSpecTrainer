# 仪电 DSpark AdamW1e-4 两轮训练

用户要求启动2epoch，随后明确转到仪电。BP已排队作业job-65b3a28c23d2-20261106191543取消并回查空，未开始训练。

仪电4×8H200，fdj-infra/nex_bp；沿用已验证AdamW仪电2e-4源码b2e6e77f8a45ccc7d14cd3b3c1f85a2c37b41f4d、固定镜像、相同1326 train/17 validation特征。官方MTP fresh初始化seed42，无resume。AdamW峰值1e-4，globalbatch128/micro1/accum4，all anchors/chunk64，FP32 reduction/FULL_SHARD/CPU optimizer offload。

num_epochs2/max_steps236/total_steps236，warmup_ratio0.03（7步），cosine236；每25步验证保存，末尾236保存。15104窗口每轮（15053原始+51既有重复padding），每rank472微批/轮，合计944微批；逐微批核对torch sampler seed42+epoch顺序，保存两个epoch的消费回执。除LR/预算及对应验收脚本外不修改训练器、模型、优化器。现有微小LR overshoot保留，不混入调度修复。

仪电特征为独立采集，不能将相对BP最好配方的差异仅归因于训练轮数。预计13–16小时，以实际进度为准。

提交成功：j-1793964028-161646；UUID ebff47ff-1db3-42da-9fdb-2eee6709d407；源码7f6954a7d4338a05051926cadcf40d8a44edb3ef。2026-11-06 19:20北京时间提交。
