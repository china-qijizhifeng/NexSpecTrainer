# 8 H100：chunk64 + 固定通信配置重启

用户要求不加卡、优化后重启，并补充要求 chunk64 固定规约。完整原 r8 已停止，0 optimizer 更新、无 checkpoint。复用原 1,326 条 trace、118 optimizer steps、全部合法 anchors，batch1/rank×8卡×accum16、LR1e-5、warmup3、原损失权重和数据顺序保持。

选择：anchor chunk64、保留外层完整 chunk checkpoint、关闭内层逐层 checkpoint。CUBLAS_WORKSPACE_CONFIG=:4096:8，TF32关闭，NVTE_ALLOW_NONDETERMINISTIC_ALGO=0，NCCL Ring/Simple，PyTorch deterministic algorithms，FSDP reduce_dtype FP32。启动器 fixed_reduction.py 固定该运行策略；不是所有自定义 kernel 逐位确定性的承诺。

完整模型同一实际第二微批次8卡对照：固定32为83.139秒，固定64为44.077秒；相比原普通32的73.306秒吞吐1.663倍。所有卡最大loss相对差异0.02855%，完整梯度分片相对L2最大4.6234%，低于事先5%门限；固定32重复运行最大0.1969%。差异仍在，未声称训练轨迹完全等价或收敛不变。峰值44.316GiB。

精度修正：每微批次 chunk 梯度FP32本地分片汇总、重叠上下文gather的FP32反向汇总；前向保持既有TileKernels/DeepGEMM kernel，未切换QuACK。参考 Nex-Megatron 56a1aa958c43dddc4fe7598800d73959bcbb0e31 的数值/生命周期和失败记录原则。失败r1/r2/r3保留，r4是已通过的固定形状checkpoint对照，r5是用户要求的固定设置64对照。

目录 /gpfs/users/linlizhi/dspark-all-anchors-20261105-r9，WeaverRSI/proj-32、8H100，固定原镜像、72小时上限。先跑同设置8卡32微批次/2更新 gate，随后完整训练。save/eval interval25，最后118步NFC。监控精确终态清理，不自动重试；每批覆盖/显存/耗时，每更新LR/grad norm，冻结表哈希检查。

单个微批次外推纯训练约23.1小时，未包含样本变化、评估、保存，不保证完成时间。运行状态以实际回执为准。
