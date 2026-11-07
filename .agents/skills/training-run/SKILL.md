---
name: training-run
description: 在 NexSpecTrainer 中准备、启动和验收 SpecForge 训练，复用 BP Dev 已验证的 V4.1 Flash 环境，核对数据编码、GPU 运行与 NFC checkpoint。适用于训练运行；不用于只读超参调研。
---

# Training environment and acceptance

**这是临时运行方案。正式训练镜像构建、发布并验收后，应优先使用固定 digest 的镜像，不再继续依赖 GPFS overlay 或任务目录内的 Megatron 组合。**

训练入口统一遵循 [use-nsp](../use-nsp/SKILL.md)。本 skill 记录已验证的环境与运行边界，不代表下面的历史路径仍然存在、镜像已经发布，或本次训练已经完成。以用户当前选择的数据、上下文方式和规模为准。

## 先确认运行方式

1. 检查已有镜像、构建回执及仓库的 [镜像构建说明](../../../docker/deepseek-v41/README.md)。有可拉取且覆盖当前 forward/NFC ABI 的正式镜像时，固定 digest，直接使用镜像内依赖。
2. 正式镜像尚不可用时，BP Dev V4.1 运行读取 [临时环境与启动步骤](references/bp-dev-v41.md)。保持已验证的 Weaver Torch/TE，离线补齐缺少的依赖；不要重新解析并替换整套环境。
3. 固定源码 commit、target snapshot、数据 revision/split、resolved config、运行目录和作业 identity。使用独立输出与缓存目录。pilot 和全量训练分别记录范围，不把少量 optimizer 更新写成正式训练完成。

## 数据与完整上下文

- V4.1 Flash 必须使用与 target 相同 snapshot 的官方 tokenizer 和官方 `encoding/encoding.py::encode_messages`。该 snapshot 没有 HF `chat_template`，不能用通用 `apply_chat_template` 替代；保留用户要求的 reasoning、工具调用和监督区间。
- 用户选择“整条轨迹一次 prefill 后切片”时，复用 `scripts/prepare_server_hidden_states.py` 和原生离线 feature reader。先采集完整轨迹，再切训练片段；4K 训练长度不表示把 target 上下文重置到 4K。超长样本不得静默截断或跳过。
- 采集字段、原生 MTP 初始化与训练配置见 [V4.1 训练契约](../../../docs/deepseek-v41-native.md)；文件入口见 [server capture](../../../docs/sections/examples/server-capture-to-disk.md)。按 instance 划分后再切片。
- 切片检查 IDs/mask、完整长度、层语义、BF16 shape/finite、压缩回读以及 overlap/anchor/标签对应。沿用局部 position IDs 时明确记录，不声称与完整绝对位置参考数值等价。
- 整条轨迹模式先核对采集 buffer、8 GiB decoded-record 限制、最长轨迹显存和存储预算。当前示例入口拒绝超过限制的 record；提高 context-length 参数不能消除该限制。不要在用户暂缓优化后继续实现新的分块采集系统。

## 提交与验收

- BP Dev 使用 SSH `linlizhi-bp-dev`、Northjob project=`WeaverRSI`、queue=`proj-32`，提交前核对当前配置。已有可用的 `northjob-gpu-test-bp-dev` skill 时遵循其认证、提交和精确清理流程。不要改用默认项目或绕过节点专属限制。
- 先做实际镜像中的环境预检，再启动用户授权规模的 GPU 运行。作业 payload 使用 `nsp train --config ... --foreground`（或 `python -m specforge.nsp.cli`）；由 nsp 内部调用分布式 launcher，不在每个 torchrun rank 各提交一个 Run。临时环境示例见 reference，不将历史 3 步 pilot 配额硬套到正式任务。
- 镜像拉取失败、容器 Running、target 健康、采集完成和 optimizer 更新是不同状态。出现一类运行错误后，先核查配置、契约及相邻调用方，修复后再试；不要循环重提同样失败的作业。镜像拉取失败时先释放本次失败作业，查相同 digest 的已验证缓存或镜像源。
- 记录实际训练步数、loss/grad norm、数据消费范围、GPU/host 峰值，检查冻结 target embedding/head 未进 optimizer、没有梯度且未变化。短 pilot 只证明已覆盖的执行链路，收敛、长样本、全量覆盖和推理质量分别报告。
- 保存完整 NFC 原生分片 checkpoint：模型 shard、FP32 masters/Adam、scheduler、RNG 和训练/消费 metadata。等待发布完成后验证 manifest、step/world/layout，恢复及下一步更新另行验证。遵循 [NFC 契约](../../../docs/native-nfc-checkpoints.md) 和当前 `specforge/training/nfc_checkpoint.py`，不要恢复已移除的 generic snapshot/独立 torch_dcp backend。
- 结束后保留 checkpoint、日志、版本与结果回执，按本次精确 alias 清理作业并读回确认。失败立即记录真实状态，不宣称后续训练会自动运行；有一次性接续脚本时，核实其存活、触发条件、停止条件与重复提交防护。
