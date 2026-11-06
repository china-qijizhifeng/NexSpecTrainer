# DeNovoSWE / SpecForge dSpark stage1 r4

修复 commit：`952e70c3e8109e5295436141b6c6336897448bc2`，基于 `66f0cfd81443a6481232d8fdf7b4abaa7c1f63b7`。

## 修复与验收范围

r3 的第 6 次更新、第 12 个累积 micro（全局 micro-step 92）出现跨 rank 通信顺序分歧：rank0 的有效监督只能生成 19 个 anchor、3 个 objective chunk，其他 rank 为 32 个 anchor、4 个 chunk。FP32 ConfidenceHead 是独立 FSDP 子模块，rank0 提前进入 denominator allreduce，其他 rank 仍在进行 head 参数 allgather。8 个 rank 最终只完成 5/118 次更新，未生成 checkpoint。

r4 保留原 sampler 的有效 anchor 及随机采样，只把 DSpark 的 anchor/keep 张量补齐到配置的 32 个槽位。补齐槽位 keep=False，不贡献 loss/metrics；所有 rank 执行 4 个 objective chunk，包括 activation checkpointing 重算。没有删片、截断轨迹或更改训练超参。

CPU 回归验证有效 anchor、loss、metrics 和全部可训练参数梯度保持一致且 finite。8 H100 上的真实 FULL_SHARD、原生 FP32 ConfidenceHead、DSpark objective 和 CPU-offloaded optimizer 回归全部通过；checkpointing=False/True 各运行 32 micro、2 次更新。回归更新不计入正式 118 步。2026-11-03 21:59 UTC，完整模型 8 个 rank 均完成正式第 6 次更新并继续，越过原 micro-step 92；现场快照再次观测到该 micro 的 rank0 仅 19 个有效 anchor。第 6 步 loss=0.6375944614、grad_norm=0.2769145668，见 `repair-validation.json` 和 `original-faultpoint-observed.json`。完整 118 步、最终冻结表哈希、checkpoint 发布和恢复验收仍未完成。

## 唯一运行身份

- BP Dev 根目录：`/gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r4`
- Northjob alias：`denovo-specforge-stage1-train-20261104-r4`
- 精确 job：`job-7fd88bb2852e-20261104054920`
- 资源：8 H100，project `WeaverRSI`，queue `proj-32`
- controller：PID `23544`，start_ticks `2346921637`，命令 `python3 -u /gpfs/users/linlizhi/dspark-denovo-specforge-stage1-20261104-r4/controller.py`

controller 单次提交、无自动重试，终态记录后精确清理该 job。不得重启已有 controller 或原样重复提交。`training-job.json` 保存提交时状态，实时状态由精确 alias API 查询及 `pipeline-status.json` 核对。

## 数据和训练

复用 r3 已完整采集、逐位 codec 验收的特征，不重新采集。stage1 为全部 1326 条 <=65536 tokens 的完整训练轨迹，59,274,736 训练 tokens；15053 原始训练片加 51 个显式补齐片为 15104 片。17 个 disjoint validation instances 选出 136 个固定验证片。8 rank、batch1、accum16，1 epoch=118 次真实 optimizer 更新，LR=1e-5，gamma=4，anchors=32，objective_chunk=8。每 25 步保存/验证，末步完整 NFC 保存。

其余 2862 条长训练轨迹 deferred，尚未启动。stage1 不代表全量训练完成。保持同 snapshot 官方 tokenizer/encoding、thinking、drop_thinking=False、effort75；每条完整轨迹一次 prefill 后切 4K/overlap129。

训练产物须包含 NFC 原生分片模型、FP32 masters、Adam、scheduler、RNG 和消费 metadata。checkpoint 发布、恢复及下一步更新分别验收；当前不宣称 restore_verified。

## 证据和存储

`execution-sha256.json` 固定训练源码及运行脚本，`cpu-regression.json` 和 `gpu-regression/rank*.json` 记录回归。`training/live-progress-rank*.json` 记录真实更新，`training/micro-progress-rank*.json` 记录实际 micro，`training/frozen-baseline-rank*.json` 与最终 `live-audit-rank*.json` 核对冻结表。`training/worker.log`、`controller.log` 和 `monitor-snapshot.json` 保留进度与错误。

仅删除旧 pilot 一个未发布的失败 staging 目录，逻辑大小 87,143 bytes，分配大小 131,072 bytes，详见 `checkpoint-cleanup.json`；其唯一 metadata 已备份。r3 无 checkpoint 可清理。已发布旧 pilot checkpoint、旧 dSpark 历史 checkpoint 和 huangyuqi 的 Weaver checkpoint 均保留，experiments 清理暂停。

必须保留 r3 特征及其依赖的 r2/source、capture-w*/train-full.jsonl、validation-full.jsonl 和 cache。r4/source 是独立修复源码。持续核对 `/gpfs/users/linlizhi` 空闲量，预留 850GB 用于完整 latest/best/inflight 及 headroom。2026-11-03 21:52 UTC 实际空闲约 1.322TB，随其他写入波动。

## 后续终态（2026-11-03 22:14 UTC）

8 rank 最终完成 25/118 次真实更新，第 25 步固定验证成功（loss 0.6652339372、acc 0.8099694246）。首次 NFC 保存规划在 Core 的 gather_object/WORLD NCCL 报错，未发布 checkpoint。controller 已记录 Failed、精确清理 job-7fd88bb2852e-20261104054920 后退出，不能重启。详见 training-nfc-failure.json 与 worker-failed.log。新修复在独立 r5 源码及运行目录进行，不改写 r4 训练证据。
