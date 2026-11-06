# DSpark AdamW1e-4 两轮训练

用户2026-11-06授权启动2epoch。沿用BP最好已完成配方：32H100/4nodes、相同1326 train+17 validation traces、15104 train windows（15053原始+51既有重复padding）、globalbatch128/micro1/accum4、all合法anchor/chunk64、FP32 reduction/FULL_SHARD/CPU optimizer offload、seed42、相同loss、源码和固定镜像。从官方DeepSeek-V4.1-Flash-1017 MTP fresh初始化，无resume。

变更：num_epochs=2，max_steps=total_steps=236；warmup_ratio保持0.03，由3步变7步；cosine236。其余训练参数与已完成1epoch配方相同。每25步验证与保存，最终checkpoint236。两轮每rank944训练微批，每轮472；全局30208窗口消费，含每轮51个原有重复padding。逐微批对照torch DistributedSampler seed42+epoch的独立参考顺序，记录每轮完整消费回执；第二轮重新洗牌，非第一轮重复同序。

项目WeaverRSI，Volcano queue proj-32，独立alias和目录。提交前检查已有作业，独立双步32rank数值门禁通过后开始正式模型。预计约14–15小时，实际受分配节点与保存耗时影响。监控仅处理本次精确job身份，保存终态证据后释放资源；不自动重投。

提交JobId：job-65b3a28c23d2-20261106191543；2026-11-06 19:15北京时间，初始Pending，尚未确认实际更新。

用户随后明确改到仪电。本BP作业仍Pending时取消，精确alias回查空，无实际训练更新。替代仪电job j-1793964028-161646，详见../yicloud-dspark-adamw-lr1e4-2epoch-32gpu-20261106-r1/RUN.md。
