# Muon + AdamW 全量 anchor 对照

- Alias: `dspark-all-anchors-muon-20261105-r2`。
- 固定源码：NexSpecTrainer `e480b1a28167ae3165e77b9621712427e871dddc`，包含 PR #15 和 #16。
- 对照：`dspark-all-anchors-20261105-r9` AdamW；保持原作业运行。
- 复用原 1,326 条训练 trace 的已审计特征、划分、顺序、官方初始化、seed 42；118 步、每卡 batch 1、accumulation 16、LR 1e-5、warmup 3、cosine。
- 全量合法 anchors，chunk64，关闭内层 checkpoint、保留外层 chunk checkpoint；固定 NCCL Ring/Simple、FP32 reduction，与 r9 同一配方。
- `optimizer=v41_flash_muon`：Mu​​on 矩阵更新（momentum .95，8+2 NS，BF16 NS）；指定其他参数 AdamW (.9,.95), eps 1e-20，decay 按 PR 配方，组 LR multiplier 均 1。是完整优化器配方对照，不是只切换算法而保持所有优化器常数。
- 新申请 8 H100 / 64 CPU / 640 GiB，WeaverRSI / proj-32，不固定节点，72 小时 deadline。
- 复用经过验证的固定 Weaver 镜像与隔离 GPFS overlay，启动时核对 Torch/TE/forward/NFC ABI、源码及配置指纹。
- 本地 131 tests + 100 subtests 通过（6 skipped）；双进程 CPU FSDP / Muon 32 micros + 2 updates 通过。
- GPU 先执行同一组合门禁，再进入实际 native 全规模训练。每个微批次核对所有实际 anchors、位置曝光与数据 mask；每步记录 norm、LR、optimizer 时间及 RSS。
- 第 17 个微批次保存 startup-acceptance，证明一次真实更新及下一次 backward 完成、loss 有限、冻结表未变化。不是训练结束或收敛验收。
- 每 25 步评估/保存，最终 step118 NFC。沿用 PR 已披露的非逐位 full-model resume 差异；本次不宣称恢复或下游质量等价。

首次 CPU 组合门禁遗漏了 TrainingConfig 中显式 FULL_SHARD（backend 已设），配置验证拒绝；静态核对所有 Muon config/工厂后补齐，保留失败日志后再跑通过。正式 train.yaml 原本即 FULL_SHARD，无 GPU 失败或重复作业。


本次使用修正的相同梯度 optimizer oracle，详见 DIAGNOSIS.md；r1 GPU 门禁失败证据保留，正式训练未启动。

当前精确作业：`job-3a1b09d413fd-20261105171629`。本地 8 进程修正门禁通过，两种 backward 路径各 32 微批次/2 更新；相同梯度完整矩阵 oracle 的 master/update 差异均为 0。

真实 8 H100 组合门禁已通过：32 微批次 / 2 更新，相同梯度完整矩阵 oracle 更新差异为 0；whole-graph vs streamed 梯度最大相对 L2 0.41999%，跨路径 master update 差异 13.912% 仍不被当作等价性通过。
