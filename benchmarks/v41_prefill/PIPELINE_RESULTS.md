# V4.1 Flash：无 hash 的完整采集流水线

状态：全部完成。16k/64k对照、64k重复、正式writer落盘及resume均通过；作业Completed后已删除，查询确认不存在。

## 实现

新增有界 producer/consumer 调度，正式 `ServerCaptureFileWriter.write_many` 与
benchmark 共用。`--pipeline-depth 2` 将下一条HTTP capture与当前记录的取回、
解码、校验、压缩及fsync重叠。一个producer顺序调用adapter，一个consumer处理CPU
tensor，最多2条capture在途；不并行两个模型forward，不改变attention上下文。
默认depth1保持串行。容量检查按最坏编码大小预留2条服务端对象，不能借预计压缩率超售。

wire新增 `bf16_fields_zstd_v2`，磁盘新增 `NSACTZ02`，明确不计算对象/文件头SHA256，
Zstd frame也不写checksum。激活校验、输入身份、输出验收及源码取证均不计算hash。
服务端和客户端实验codec都安装保护，若尝试SHA256便报错；全部回读要求v2并拒绝
带checksum的frame。旧v1兼容代码保留在其他调用路径，格式不会混用。
正式writer保存并直接比较payload和metadata，capture-plan更新为v2；不会静默续跑旧hash manifest。

仍逐条检查全部activation有限值、shape/dtype和token/mask对齐，使用原生DSpark
normalizer/collator回读文件。输出验收直接对每个tensor的每个字节做比较，不用fingerprint。
结构检查不等同于任意bit flip检测；取消checksum是本次明确要求。

磁盘错误、输入错误、提前关闭迭代器时等待唯一在途producer，abort并drain尚未消费的
capture；当前记录由consumer释放。HTTP响应丢失时额外清理未adopt的服务端写入。
发布仍使用临时文件、fsync、原子rename；异常不会把.partial当作训练样本。

## 固定条件

8×H100，node044，WeaverRSI/proj-32；TP8/EP8，CPU Engram，原版FlashInfer MXFP4 MoE，
启用上一轮已验收fused indexer。模型DeepSeek-V4.1-Flash-1017，revision
`1bae0e1088993b9b82de08f59de5be5a34e42170`；SGLang
`38d9e4a3fe4a493b0ae468477a274e83defda81a`；同一固定镜像。
两条约80k的真实OpenHands轨迹，从offset4096取连续16k/64k，直接复用上轮输入文件。
全aux37/38/39+final BF16；关闭chunked prefill、bounded SWA replay及CUDA graphs。
context69632、max-total-tokens131072、显式SWA ratio0.8、expandable allocator，
服务端/客户端均8个codec workers、8GiB Mooncake buffers。

每个阶段2条输入×4次=8请求；吞吐包含填充、排空、取回/解码、完整校验、落盘fsync及
对象删除。预热、同步GPU诊断、文件回读和paired tensor比较不计入吞吐。
GPU forward诊断取8 ranks最大CUDA event时间。p95仅是8请求最大值。
历史含hash组只有4请求且是另一服务进程，只用于描述合并效果；本轮无hash串行/流水线
在同一服务、输入与配置下对照，才用于归因流水线收益。

## 已完成结果

| 连续输入 | 无hash串行 | 无hash流水线 | 流水线增益 |
|---|---:|---:|---:|
| 16k | 6315 tok/s | 10825 tok/s | 1.71× |
| 64k | 6243 tok/s | 10255 tok/s | 1.64× |
| 64k重复 | 同上 | 10226 tok/s | 1.64× |

16k落盘阶段GPU利用率均值29.4%→52.3%；单请求p50仍约2.49秒，forward约668–670ms。
流水线降低请求之间的空闲，而不是把单条模型forward变快。
16k/64k各8份配对文件的完整aux/final/token/mask均逐位一致；native回读全部通过；
resident_samples、provisional_external、active_leases、resident_bytes及release_pending均为0。

64k落盘阶段GPU利用率均值30.3%→50.6%；单请求p50约10.29秒，forward约3211ms。
8条落盘阶段墙钟时间约84.0秒→51.1秒，包含首条填充与末条排空。

| 64k每请求平均阶段 | 无hash串行 | 无hash流水线 |
|---|---:|---:|
| HTTP含GPU和服务端publication | 5.721 s | 5.802 s |
| Mooncake取回 | 0.629 s | 0.631 s |
| 解码 | 0.969 s | 0.988 s |
| 全量有限值及对齐检查 | 1.586 s | 1.473 s |
| 压缩与fsync | 1.332 s | 1.319 s |

阶段发生重叠，不应将并发阶段相加后当作整组吞吐分母。

上一轮含hash、已开启同一indexer的完整采集为16k 4327 tok/s、64k 4194 tok/s。
本次无hash加流水线分别约2.50×、2.44×；这是跨运行的合并效果，不能全部归因流水线。

## 验收与资源

10项针对性回归已通过，覆盖无hash真实sender→Mooncake→decoder、所有BF16位模式、
frame checksum禁用、并发上限、各种失败清理、writer续跑和容量边界。
Black、Ruff F/I与git diff --check通过。
实际GPFS运行目录的12个关键源码文件与本地worktree直接逐字节比较一致，
未计算源码hash；回执为 `source-direct-comparison.json`。

作业：`job-d1b2aaef69a0-20261103152321`，alias `nexspec-v41-prefill-pipeline-20261103-r1`。
GPFS证据：`/gpfs/users/linlizhi/nexspec-v41-prefill-pipeline-20261103`。
本地代码：`/Users/shom/Downloads/NexSpecTrainer-v41-prefill`，未提交、推送或修改main。

正式writer对真实服务的4份输出（16k、64k各2份）均与串行基准逐位一致；
4次resume正确跳过。回执：`evidence/production-writer-validation.json`。
Benchmark共40份文件通过原生训练reader回读；24份pipeline/repeat输出与各自串行
基准做全量字节比较，全部一致。回执：`evidence/parity-{12,15,17}.json`。
正式writer的原始payload sidecar和activation文件保留在GPFS，不复制到本地证据包。

服务和Mooncake自有进程组已按PID核对并停止，作业终态Completed；按精确alias删除，
查询返回空。回执：`job-terminal.json`、`job-delete.json`、`job-absent.json`。

## 使用

服务端和客户端都设置 `SPECFORGE_ZSTD_WORKERS=8`；服务端启动前设置
`SGLANG_SPEC_CAPTURE_WIRE_ENCODING=bf16_fields_zstd_v2`。
正式采集命令在原有参数上增加：

```sh
--wire-encoding bf16_fields_zstd_v2 --zstd --pipeline-depth 2 \
--client-buffer-bytes 8589934592 --server-buffer-bytes 8589934592
```

需要同步使用本worktree版本的serving plugin、client和训练reader，旧版reader不认识
NSACTZ02。`--pipeline-depth 1`用于关闭重叠。报告中的GPU forward没有因
流水线而加速：提升来自取消hash及减少客户端消费造成的GPU等待。
