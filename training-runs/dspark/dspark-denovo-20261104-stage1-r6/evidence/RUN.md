# DeNovoSWE / SpecForge dSpark stage1 r6

固定源码 e925d251e9eece36339b3bc2a888ad19e15727ef，基于 r4 修复 952e70c3e8109e5295436141b6c6336897448bc2。


r4 已完成 25/118 次真实更新及第 25 步固定验证（loss 0.6652339372、acc 0.8099694246）。首次 NFC 保存的 Core SavePlan gather_object 经默认 NCCL 报错，未发布 checkpoint，只有 87,143 bytes 的 common.pt 暂存文件。r4 精确 job 已失败并清理，controller 已退出。节点/fabric 或保存时资源压力仍不能仅凭旧日志排除，不能声称原 NCCL 故障的底层原因已经确认。

r5 为 CUDA WORLD 注册 cpu:gloo,cuda:nccl，让 host 对象和保存计划走 Gloo，CUDA FSDP 张量继续走 NCCL；保持 NFC 原生 WORLD 契约、分片 pipeline 和完整状态格式。CPU 相关 48 项测试通过。真实镜像的 GPU 保存/恢复验收须以本 run 回执为准。

唯一 8 H100 作业先做环境预检和临时wrapper真实策略回归，核对固定SHA后复用r5已经完成的dSpark稀疏anchor及真实NFC多布局/异步发布/bitwise恢复与下一步GPU回归回执，再进入正式stage1。正式训练计划先完成第 1 步并额外发布完整 NFC，再以新进程恢复该 checkpoint，逐 rank 核对完整模型/FP32 masters/Adam/scheduler/RNG 哈希后继续第 2 至 118 步。第一阶段只改变 max_steps 早停 cap 为 1，total_steps 与 scheduler 始终为 118；恢复阶段 max_steps=118，仍从原始 epoch 的每 rank 第 17 片继续，无重放或跳过训练片。小模型回归的更新不计入正式 118 步。

仍使用已完整采集的 1326 条训练轨迹、15104 片和 17 个 disjoint validation instances 的 136 片；不重新采集。正式 train.yaml 除 run 路径外与 r4 相同：batch1、8 rank、accum16、LR1e-5、gamma4、anchors32、chunk8、25 步验证/保存、末步完整 NFC。其余 2862 条长轨迹 deferred，stage1 不代表全量完成。

新的 audit 将训练和验证的 micro 计数分开；正式每 rank 应消费 118*16=1888 个训练 micro。初始阶段的 16 个 micro 与第一步 grad norm 持久化后与恢复阶段合并。第一步恢复通过不代表最后一步 118 checkpoint 的恢复通过，分别记录。

单一 controller、独占提交 intent、无自动重试，终态后只精确清理本作业。身份见 controller-process.json、training-job.json。保留 r2/r3/r4 的活跃输入、特征及 cache 依赖，不删除历史已发布 dSpark 或 huangyuqi Weaver checkpoint。预留 850GB 用于完整 latest/best/inflight 和 headroom。

r5 的相同源码、镜像及环境完成全部 8rank dSpark/NFC GPU 回归，40 份 NFC 回执通过，但正式首步前 audit wrapper 使用不存在的 self.model 导致 AttributeError，零正式更新、无 checkpoint，精确作业已清理。r6 仅修复 wrapper 使用公开 trainable_module().training；CPU 调用原生 NativeStrategy 和准确 AST 抽取的 wrapper 先复现旧错误，再通过 train/eval/train 三次调用及有限 backward。r6 独立源码与 r5 逐文件 SHA 相同，复用 r5 GPU 回归回执，不重复同一 GPU 回归；原 118 步配置和第1步保存/恢复计划保持。

当前唯一身份（22:47 UTC提交）：alias denovo-specforge-stage1-train-20261104-r6，job-261dab7030ba-20261104064734，8 H100，WeaverRSI/proj-32。一次性controller PID62391、start_ticks2347270969，命令python3 -u /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r6/controller.py。不得重启或重复提交。22:52 UTC全部8rank已完成初始16训练micro的前向；尚未发布第1步checkpoint，恢复验收仍待实际执行。

22:58 UTC 更新：第1步完整 NFC 已发布（199,168,059,921 bytes / 38 files），完成标记和 common.pt SHA 核验通过。8rank 在新进程恢复后，全 model/FP32 masters/Adam/scheduler/RNG hash 与保存前逐位一致；8rank 全部完成恢复后的正式更新2，grad norm=0.5967926979，每rank累计32训练micro。见 nfc-repair-validation.json、full-model-restored-rank*.json 和 first-restored-update-rank*.json。初始阶段冻结 embedding/head 完整前后哈希8rank通过。该结果验收第1步checkpoint的保存与恢复，不代表118步训练或最终step118 checkpoint的恢复验收完成。训练正继续，当前实际进度见monitor-snapshot.json。

r4新失败的未发布staging已备份metadata并精确删除，逻辑87,143bytes，见r5/checkpoint-cleanup.json；之前pilot残留也是87,143bytes。其他空闲变化来自共享挂载上其他写入/清理，不能计为本次释放。


2026-11-04 00:14 UTC 正式 stage1 完成：8 rank 全部118次optimizer更新、各1888训练micro。全部15104片（15053原片+51显式padding）的实际input token SHA与不可变整轨迹切片及原生seed42/epoch0 sampler顺序逐项一致。8rank冻结embedding/head前后完整hash相同、不入optimizer且无梯度。最终loss0.7648583651、gradnorm0.1916707754、scheduler终态LR=0。839项execution SHA仍一致。原trainer Completed并精确清理，原controller已退出。详见stage1-completion-audit.json；这里只是1326条<=65536tokens完整训练轨迹，2862条长轨迹仍deferred。

最终NFC已发布：training/denovo-v41-specforge-stage1-step118，199168059737 bytes、38files/32tensor shards，完成标记和common.pt SHA核对通过。latest=step118，best=step25（eval/simulated_acc_len=0.8099694246）；step100已原生轮转。100步固定验证loss0.6863564260、acc0.8019985196；118步按原interval不额外eval。

最终checkpoint fresh restore正在独立验收：../dspark-denovo-specforge-stage1-20261104-r6-final-restore，alias denovo-specforge-stage1-restore-20261104-r6，job-99c87ac1ba46-20261104081123，8H100。只在临时进程以原样终态LR做一次diagnostic optimizer window再重新restore对比完整hash，不写checkpoint，不增加正式118步。尚未通过时不将旧step1恢复回执当最终验收。保留training/result.json restore_verified=false的历史事实，新增回执记录后续验收。


2026-11-04 00:19 UTC 最终验收通过：最终step118 checkpoint在独立8H100进程fresh restore，8rank验证Adam counter118、训练metadata和冻结表hash。各rank随后在临时进程重放原epoch0首16片，原生backward与一次diagnostic optimizer更新有限（loss0.6805745363、gradnorm0.2400847524，Adam118->119），再重新restore最终checkpoint，完整model/FP32masters/Adam/scheduler/RNG hash全部返回首次restore值。诊断原样使用终态LR=0，未覆盖LR，因而不代表非零LR的权重更新验收。未捕获最终保存前全状态hash，本次对比范围是首次fresh restore与诊断更新后再次restore；step1的保存前/恢复后全状态比对另已通过。正式训练仍恰好118步，无额外正式消费，不发布诊断checkpoint。

最终训练与恢复job均Completed且精确删除，两个controller均退出，精确alias API查询为空。839项训练execution及8项验收execution SHA一致，最终checkpoint大小/标记/commonSHA保持，latest118/best25约398GB保留，实际空闲2813589454848 bytes。最终新增验收回执stage1-final-acceptance.json passed；原training/result.json和stage1-completion-audit.json保留各自发布/检查当时restore=false历史，不覆盖它们。全部32份rank恢复/更新/roundtrip/complete回执与consumption-audit位于../dspark-denovo-specforge-stage1-20261104-r6-final-restore。2862条长轨迹仍未训练。
