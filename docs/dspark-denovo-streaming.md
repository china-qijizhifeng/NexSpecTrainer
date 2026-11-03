# DeNovoSWE dSpark 的采集缓冲与 checkpoint

本页记录 [训练计划 #8](https://github.com/china-qijizhifeng/NexSpecTrainer/issues/8)
的资源与缓冲契约。正式训练仍依赖完整前缀分块采集（#5）、online 固定验证集支持和 GPU pilot。
下面的配置片段只描述这些契约，不能作为完整上下文训练的启动配置。

## 资源与数据流

采集池使用 32 张 H100，拆成 4 个独立的 TP=8、EP=8 V4.1 Flash 副本。
训练池使用另外 8 张 H100，DP/FSDP=8，总计 40 张。
4 个采集 endpoint 接入同一个 producer、一个源 channel 和同一份全局缓冲预算。
各副本读取同一份固定版本的 GPFS target 权重。每个 endpoint 初始并发为 1。

必须使用 `deepseek-ai/DeepSeek-V4.1-Flash-1017` snapshot
`1bae0e1088993b9b82de08f59de5be5a34e42170` 中的 `encoding/encoding.py::encode_messages`
和 `tokenizer.json`，并设置 `add_special_tokens=False`。该 snapshot 的
`tokenizer_config.json` 没有 `chat_template`，不能使用通用模板回退。

4096-token capture chunk 表示连续 target 前向的输出块；target 保留此前完整前缀。
它与 trainer 的 4096-token 训练片段分别管理，后者还要处理 129-token overlap
及重复监督屏蔽。当前 producer 仍按一项任务返回一条 ref；本次缓冲改动没有实现 #5 的分块协议。

## 采集前预留容量

`runtime.capture_max_sample_bytes` 开启有界采集模式。每次发出采集请求前，
producer 在所有 worker 之间原子预留 ref 数量和最坏 wire bytes。
预算同时计算已发布、尚未被消费确认的对象，以及尚未返回的采集请求。
容量不足时缩小 lease 或暂停请求。收到有效 refs 后，用实际 wire bytes 记账并释放预留；
失败、取消排空和发布失败也释放预留。源 channel 的消费确认释放已发布对象的预算。
缺失、零或超过声明上限的 `estimated_bytes` 会导致该批对象被清理，且训练 attempt 失败。

开启此模式时，ref 和 byte 的高、低水位都必须容纳一个完整 consumer dispatch quantum，
避免 consumer 等待完整窗口、producer 等待消费确认的互相等待。
CLI 在准备 prompts 前检查配置与 byte 环境覆盖，producer 在 consumer 握手时再次检查。
compressed capture 请求同时携带 `wire_max_total_bytes`，交给支持该契约的服务端检查。
未设置 `capture_max_sample_bytes` 时保留原有软水位行为。

以 batch=1、accumulation=16、DP=8 的 `round_robin` dispatch 为例，quantum=128。
每个 4096-token 样本的四份 BF16 特征约 160 MiB；预留上限取 164 MiB，
为 IDs、mask 和 codec 元数据留出余量。仍需 pilot 验证完整 wire payload 的实际大小。

```yaml
runtime:
  producer_lease: 1
  producer_concurrency: 1
  consumer_dispatch: round_robin
  in_flight_high_watermark: 256
  in_flight_low_watermark: 128
  resident_high_watermark_bytes: 51539607552  # 48 GiB，全池共享
  resident_low_watermark_bytes: 34359738368   # 32 GiB，全池共享
  feature_store_max_resident_bytes: 60129542144  # 56 GiB
  capture_max_sample_bytes: 171966464  # 164 MiB，逐样本上限
training:
  save_interval: 100
  max_checkpoints: 1
```

128 个预留样本需要 20.5 GiB，256 个需要 41 GiB。每个采集 server 的 global
segment 初始设为 64 GiB、local buffer 为 8 GiB，4 个 server 总 segment 配额为
256 GiB。producer 与各 trainer rank 的 client/receive buffer 从每进程 512 MiB
开始验证。压缩配置在服务端使用
`SGLANG_SPEC_CAPTURE_WIRE_ENCODING=bf16_fields_zstd_v1`，producer 使用
`DISAGG_WIRE_ENCODING=bf16_fields_zstd_v1`。

本契约约束声明有效的 wire feature 占用，不涵盖 target KV、临时编码解码张量、
接收池、allocator 开销、失败对象隔离或服务端违反大小契约的情况。
这些内存需要分别计入部署预算。该 CPU 实现检查不能替代 4 server 跨节点 RDMA 验收。

## 保存与恢复

`output_dir` 必须位于持久 GPFS run 目录。每次完整 checkpoint 保存
`training_state.pt` 和 8 份 `training_state_rank{0..7}.pt`，包含 MTP 权重、
各 rank 的 optimizer/scheduler、RNG、步数及消费位置。沿用现有 CheckpointManager，
保持 latest 和受保护的 best；final 是最后一次完整保存，可与 latest 指向同一目录。
同时保存原生 SGLang 导出和配置、模型/tokenizer/template/data revision 与文件哈希。
不得用 weights-only 文件或原生导出替换完整 best/final checkpoint。

已有完整 checkpoint 实测约 199.17 GB；latest 链接不重复占用一份。
其中 BF16 MTP 权重约 28.5 GB，FP32 主权重与 Adam 两份动量约 170.7 GB；
各 rank 保存 optimizer 分片，不包含完整 target 权重。
latest、best、正在写入的新 checkpoint 的瞬时峰值约 597.50 GB。
轮转发生在新保存完成之后，固定验证特征、导出及至少 100 GiB 空闲余量要另外计算。
结束时核对 best/final 的共享文件、全部 rank 文件、哈希和加载结果，再清理本轮作业。

online resume 要求 checkpoint step 与 durable ack step 一致，并保留对应 ledger、
channel/inboxes 和 Mooncake 对象。周期保存不能保证任意崩溃点直接恢复。
状态不匹配时使用新 attempt，从已保存状态重新采集并重放；不得修改 ack 伪造一致性。
正式 pilot 必须验证保存点恢复、第一次恢复更新与未中断参考的一致性，以及失败后的清理。
