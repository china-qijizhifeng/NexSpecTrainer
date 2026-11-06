# 性能优化与重启结果

在原 8 H100 上完成短测并重启 1,326 条 trace / 118 步全量 anchor 训练。作业 `job-7b468f7cfd17-20261105162116`，alias `dspark-all-anchors-20261105-r9`。状态与实际训练进度见 runtime 回执；此报告不声称长跑已完成。

| 同一真实微批次配置 | 用时 | 峰值活跃显存 |
| --- | ---: | ---: |
| 普通 anchor32、内外两层 checkpoint（r4） | 73.306 秒 | 40.942 GiB |
| 普通 anchor32、仅外层 checkpoint（r4） | 61.611 秒 | 40.944 GiB |
| 固定设置 anchor32、仅外层 checkpoint（r5） | 83.139 秒 | 44.132 GiB |
| 固定设置 anchor64、仅外层 checkpoint（r5，采用） | 44.077 秒 | 44.316 GiB |

采用方案相比普通32基线吞吐约1.66倍，耗时减少39.87%；相同固定设置32→64吞吐1.89倍。每个候选两次执行，取预热后的第二次、8卡最慢用时。原始 r8 的实测也约71–77秒/批。单个微批次外推纯训练约23.1小时，不包含评估/保存和窗口分布差异。

固定设置：cuBLAS workspace=:4096:8、TF32关闭、NVTE确定性选项、PyTorch deterministic algorithms、NCCL Ring/Simple、FSDP梯度通信FP32，以及微批次内FP32梯度分片累加。自定义kernel仍有非逐位确定性；固定设置并不保证完全固定规约顺序。

数值边界：全部卡实际anchors/位置曝光完全一致；固定64相对固定32，最大loss相对误差0.02855%、最大完整梯度分片相对L2误差4.6234%，低于事先5%门限。固定32自身重复误差0.1969%。这仍是可见差异，不证明长期训练轨迹或收敛等价。普通64的失败对照保留，未放宽阈值。

保留原 optimizer/LR/warmup/accumulation/loss weights、数据顺序、FULL_SHARD和CPU optimizer offload，所有合法anchors都执行。关闭内层逐层checkpoint但保留外层完整chunk checkpoint。两处精度修正有最小负对照回归：FP32分片累加避免小梯度被逐块BF16舍入掉；重叠上下文gather在FP32中汇总梯度。

参考 [Nex-Megatron 56a1aa9](https://github.com/china-qijizhifeng/Nex-Megatron/commit/56a1aa958c43dddc4fe7598800d73959bcbb0e31)：采用同调度数值、saved tensor生命周期、失败结果保留的验证方式。当前已有TileKernels融合SwiGLU；QuACK的eager BF16舍入约定不同，因此本轮未直接替换Megatron Core或激活kernel。

本地74项控制器/流式测试通过（1项跳过）、11项精度/上下文/RoPE测试通过；双进程CPU FSDP及真实8卡回归通过。完整模型r4/r5短测成功正常退出，已释放测试作业。重启作业还会以最终固定配置运行8卡32微批次/2更新门禁，并在完整训练逐批记录显存、耗时、覆盖，逐更新记录LR/梯度范数。

源码工作区仍未提交或推送。训练源码、运行脚本、配置指纹分别保存于 source-provenance.json / runtime-provenance.json；性能数据见 performance-selection.json，以及相邻 dspark-perf-20261105-r4 / r5 的逐rank回执。

实际重启已完成前两个训练微批次：首批含预热63.77秒，第二批44.05秒，峰值44.316GiB；此时0 optimizer更新，正在继续积累。最终固定设置8卡回归32微批次/2更新通过，最大梯度L2差异0.41999%、FP32 master更新0.02644%。
