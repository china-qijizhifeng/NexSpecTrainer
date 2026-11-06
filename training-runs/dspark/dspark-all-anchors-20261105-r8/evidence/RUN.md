# 1,326 条 trace / 全部 anchors 完整训练

用户在显存优化期间授权“没事了，继续跑”。本次完整训练复用原 1,326 条 trace，全部合法 anchor；32 是计算分块大小，不是采样上限。

保持原 optimizer、LR 1e-5、warmup 3、cosine 118-step horizon、batch 1/rank、8 卡、累积 16、K=5 和损失系数。FULL_SHARD + 原 optimizer CPU offload；逐 anchor chunk 完成反向归约。数据缓存只读复用，官方原生初始化重新开始，有限显存测试的临时 Adam 状态不作为训练 checkpoint。

目录 `/gpfs/users/linlizhi/dspark-all-anchors-20261105-r8`，alias `dspark-all-anchors-20261105-r8`，WeaverRSI / proj-32。原配置保留 save/eval interval 25，最终第 118 步保存 NFC。72 小时作业上限，禁止自动重试。

已启动一次性接续进程，只有 `dspark-stream-memory-20261105-r2` 的 17 微批次/1 optimizer step 显存测试正常退出、原生 loss 对照通过、源码快照完全相同才提交。意图在 API 请求前落盘，提交结果不明确时不重提。提交身份由 submitted-job.json 记录；该文件存在也不等于训练完成。

每个训练窗口验证 input/mask 指纹和实际 anchors/各位置曝光数；每微批次记录峰值显存和用时，每更新记录 LR/梯度范数；全程检查冻结 embedding/head 未进入 optimizer，最终校验哈希。终态监控只清理本次精确作业，保留日志和 checkpoint。
