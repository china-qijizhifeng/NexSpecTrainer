# DeNovoSWE / SpecForge dSpark stage1 r5

固定源码 e925d251e9eece36339b3bc2a888ad19e15727ef，基于 r4 修复 952e70c3e8109e5295436141b6c6336897448bc2。

22:35 UTC 已唯一提交 alias denovo-specforge-stage1-train-20261104-r5、精确 job job-1d30bb48f926-20261104063523，8 H100，WeaverRSI/proj-32。一次性 controller PID32477、start_ticks2347197907，命令 python3 -u /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r5/controller.py。禁止重启或重复提交；当前状态以 monitor-snapshot.json 为准。

r4 已完成 25/118 次真实更新及第 25 步固定验证（loss 0.6652339372、acc 0.8099694246）。首次 NFC 保存的 Core SavePlan gather_object 经默认 NCCL 报错，未发布 checkpoint，只有 87,143 bytes 的 common.pt 暂存文件。r4 精确 job 已失败并清理，controller 已退出。节点/fabric 或保存时资源压力仍不能仅凭旧日志排除，不能声称原 NCCL 故障的底层原因已经确认。

r5 为 CUDA WORLD 注册 cpu:gloo,cuda:nccl，让 host 对象和保存计划走 Gloo，CUDA FSDP 张量继续走 NCCL；保持 NFC 原生 WORLD 契约、分片 pipeline 和完整状态格式。CPU 相关 48 项测试通过。真实镜像的 GPU 保存/恢复验收须以本 run 回执为准。

唯一 8 H100 作业按顺序运行：环境预检、dSpark 稀疏 anchor 回归、真实 NFC 多种 FSDP 布局/异步发布/bitwise 恢复与下一步回归，然后正式 stage1。正式训练计划先完成第 1 步并额外发布完整 NFC，再以新进程恢复该 checkpoint，逐 rank 核对完整模型/FP32 masters/Adam/scheduler/RNG 哈希后继续第 2 至 118 步。第一阶段只改变 max_steps 早停 cap 为 1，total_steps 与 scheduler 始终为 118；恢复阶段 max_steps=118，仍从原始 epoch 的每 rank 第 17 片继续，无重放或跳过训练片。小模型回归的更新不计入正式 118 步。

仍使用已完整采集的 1326 条训练轨迹、15104 片和 17 个 disjoint validation instances 的 136 片；不重新采集。正式 train.yaml 除 run 路径外与 r4 相同：batch1、8 rank、accum16、LR1e-5、gamma4、anchors32、chunk8、25 步验证/保存、末步完整 NFC。其余 2862 条长轨迹 deferred，stage1 不代表全量完成。

新的 audit 将训练和验证的 micro 计数分开；正式每 rank 应消费 118*16=1888 个训练 micro。初始阶段的 16 个 micro 与第一步 grad norm 持久化后与恢复阶段合并。第一步恢复通过不代表最后一步 118 checkpoint 的恢复通过，分别记录。

单一 controller、独占提交 intent、无自动重试，终态后只精确清理本作业。身份见 controller-process.json、training-job.json。保留 r2/r3/r4 的活跃输入、特征及 cache 依赖，不删除历史已发布 dSpark 或 huangyuqi Weaver checkpoint。预留 850GB 用于完整 latest/best/inflight 和 headroom。
