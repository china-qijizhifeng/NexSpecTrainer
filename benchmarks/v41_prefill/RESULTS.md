# V4.1 Flash prefill 与完整特征采集优化结果

2026-11-03，BP node044，8×H100，TP8/EP8，64 CPU，512 GiB 作业内存。

完成了可复现的 4k 瓶颈定位与实现。开启 8 个无损编解码 worker 后，完整采集吞吐从 1,634.7–1,655.5 提升到 3,763.3–3,773.0 tokens/s，两轮对应提升 2.31×、2.27×。模型算术未变，GPU forward 没有加速；本次收益属于 hidden-state 采集链路。

## 对照与计时

每组使用相同的 16 个真实轨迹窗口，每窗 4,096 tokens、独立上下文，重复两遍；每个稳态阶段 32 次请求、131,072 tokens。层 37/38/39 auxiliary [1,4096,15360] 与 final [1,4096,5120] 全量 BF16，保留完整 token/mask。独立 cache namespace 强制完整 prefill。

| 实验 | codec workers | forward 均值 ms | HTTP 均值 ms | 请求到落盘 p50 / p95 秒 | 完整采集 tokens/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| baseline-drained | 1 | 218.21 | 1056.0 | 2.470 / 2.606 | 1634.7 |
| codec4-drained | 4 | 228.15 | 559.3 | 1.221 / 1.527 | 3255.1 |
| codec8-drained | 8 | 224.96 | 524.2 | 1.043 / 1.362 | 3773.0 |
| codec8-repeat | 8 | 未重复 profiling | 499.7 | 1.054 / 1.329 | 3763.3 |
| baseline-repeat | 1 | 未重复 profiling | 1032.6 | 2.444 / 2.645 | 1655.5 |

Forward 为单独 profiling 阶段、每组 4 次请求的八 rank CUDA event 最大值均值。HTTP 与完整采集使用未开启 CUDA 同步 profiling 的请求；完整采集包含取回、解码、校验/hash、编码/fsync、物理释放和逐请求日志开销。后续训练读取验证不计入吞吐。每个逻辑 token 只计一次，不乘 GPU 数量。4 次 forward 诊断不足以评估几个百分点的波动，不能宣称 forward 提升。

native 服务启动到 ready 约 811.5 秒，独立报告，不计入稳态收益；启动复用了同一份 kernel cache。未发布特征的 HTTP 均值约 225–227 ms，仍保留全局 auxiliary capture 的计算/分配，不是完全关闭 hidden states 的模型。

## 瓶颈证据

- 基线完整 forward 均值 218.21 ms；rank0 两个 Engram host gather 合计 0.890 ms、all-reduce 0.750 ms、Engram 完整路径 5.295 ms。attention 82.50 ms、MoE 68.25 ms；这些嵌套计时不能相加。采集 D2H 约 3.43 ms，排队约 0.3 ms。
- 基线服务端 materialize/encode 平均 761.60 ms，4 workers 降至 300.67 ms，8 workers 为 239.64 ms。RDMA put 为 3.59 / 3.59 / 3.56 ms。
- 客户端基线 decode 514.75 ms、校验/hash 207.94 ms、encode/write/fsync 668.99 ms；8 workers 对应 163.00 / 209.24 / 130.07 ms。复测 decode 182.86 ms、encode/write/fsync 129.59 ms。
- 旧 input throughput 的分母来自相邻 `metrics_reporter.report_prefill_stats` 调用之间的墙钟时间，包含客户端停顿，不能作为 GPU forward TPS。实际参数解析为 `speculative_algorithm=None`、`aux_hidden_state_capture=dspark`，未执行 draft inference。

## 改动与正确性

`specforge/data/feature_file.py` 增加可选 `SPECFORGE_ZSTD_WORKERS=1..32`，默认 1；API 也可显式传入 workers。按原独立 block 并行 BF16 字段转换与 Zstd 编解码，每线程独立 context，保持原 frame 顺序及 checksum，最多 2×workers 个 block 在途。文件格式与字节保持不变。服务和采集客户端都设置 `SPECFORGE_ZSTD_WORKERS=8` 才能取得完整收益。

同时修复 `ServerCaptureFileWriter` 成功 release/失败 abort 后遗漏物理 removal drain 的问题，collector 退出时先 drain 再 close。最初测试发现 Mooncake read lease 返回 -706，逻辑释放后仍占 hard-pinned 存储，继而触发 -200 容量不足；暂停测试并完成静态排查、失败回归和修复后，清理了专用 master 中本任务的对象。有效对照两边都启用 drain；最初 baseline 和失败 codec4 仅保留为诊断证据。

五组共 160 个文件均通过原生 feature reader、DSpark normalizer/collator 读取；同源样本 token、mask、BF16 tensor hash 和压缩文件 SHA256 全部一致。每组结束 resident_samples、active_leases、release_pending 均为 0。详见 `evidence/file-byte-parity.json` 和 `comparison.json`。

本轮本地验证：feature_file / hybrid_feature_file / capture_wire_compression 共 38 passed、62 subtests；server_capture / sglang_capture_plugin_sink 共 43 passed、5 skipped、11 subtests。black、ruff F/I 和 diff whitespace 检查通过。完整 ruff 中原有规则告警未扩大处理。

## 资源与适用范围

8 workers 复测 collector 最大 RSS 9.603 GiB，1 worker 9.427 GiB，约多 179 MiB；其中包含固定 8 GiB RDMA buffer。两次复测客户端 CPU 时间分别 195.51 / 196.73 秒，覆盖 warmup、三个测量阶段和读回验证，不能解读为仅落盘阶段 CPU。更多线程缩短墙钟时间，CPU 总工作量相近；高并发或训练多进程读取时应重新分配线程预算。

GPU peak allocated 约 70.54 GB（65.7 GiB），不是整卡总占用。cgroup memory.current 包含模型表、服务、客户端和缓存，不能归因于 codec。未采集可比的连续 GPU utilization 曲线；没有据此声称利用率提升。

本轮仅验证单请求、4k 完整采集，没有执行连续 16k/64k，也未验证跨节点或 GPUDirect。训练环境 Torch 2.10 + TE 2.20.2 未改；独立 serving 镜像为 Torch 2.14.1+cu130。

## Engram 与推理实践

独立子 agent 完成了 `inference-practices.md`：DeepInfer 的 all-layer fetch + per-layer wait、DeepEP CPU RDMA、Nextron PLE 双槽 ready/consumed 生命周期值得参考。现有 Engram 已 100% huge pages，而本负载 gather 低于 1 ms，因此本次没有引入 Engram 预取或改动模型通信顺序。

DeepInfer 官方固定版本主要面向 SM100/Ascend950；Hopper PR #92 尚为开放提案，H20-3e/YOCO/DSpark 数字不能作为本任务 H100 全量 capture 的基线。知乎原文访问返回 403，研究使用官方仓库 Q&A 与源码交叉核对，未冒充读过原文。

后续若继续优化，应先检查当前仍占约 200 ms 的校验/hash，以及请求与 CPU 消费能否有界重叠；需保留完整验证和输出契约。若目标限定为模型 forward，则重点重新 profile attention/MoE，不能把本次 2.3× 当成 kernel 加速。

## 版本、复跑与证据

- Trainer 基线：`debd442d8d751edb4a7bc6a8d7d7cd23e4633512`；改动位于独立 worktree `/Users/shom/Downloads/NexSpecTrainer-v41-prefill`，分支 `perf/v41-prefill-20261103`，未 commit/push/merge。
- Model：`deepseek-ai/DeepSeek-V4.1-Flash-1017`，revision `1bae0e1088993b9b82de08f59de5be5a34e42170`。
- SGLang：`38d9e4a3fe4a493b0ae468477a274e83defda81a`；镜像 `lmsysorg/sglang@sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa`。
- Nextron：`9763a325081d4ff9c2e0df880a339aae33886d88`；DeepInfer：`553741557a57df44a46fb3ab72b6aafeaf945337`。
- 输入 SHA256：`cb2ea99dc1b44739369f29eb61180edfa80c3a4e6886c2d3b0cf9906b9ece0bf`。真实 token 正文保留在 GPFS，不纳入代码提交。
- GPFS bundle：`/gpfs/users/linlizhi/nexspec-v41-prefill-20261103`；含 source、输入、job YAML、启动脚本、日志、压缩文件。`evidence/execution-files.json` 固定已执行文件 hashes。
- `benchmarks/v41_prefill/README.md` 给出环境与复跑命令；新作业中 setup 安装插件后，`restart-full.sh` 注入诊断 hook 并启动专用 Mooncake master 和 native SGLang。需使用新的结果目录，避免覆盖本轮证据。
- 本轮实际执行 harness 保存在 `executed-harness/`；旧服务 hook 使用 `engram_mode` 名称及 `engram_prefetch.py` 兼容导入，但该 shim 仅转调 codec worker 配置，未实施 Engram prefetch。最终代码统一命名为 `runtime_mode` / `runtime_control`；不要把旧 hook 与新 driver 混用。
- 本地汇总：`python3 /Users/shom/Downloads/NexSpecTrainer-v41-prefill/benchmarks/v41_prefill/summarize.py evidence baseline-drained codec4-drained codec8-drained codec8-repeat baseline-repeat --output comparison.json`。
- Northjob 项目 WeaverRSI、队列 proj-32。r2 `job-676293085163-20261103121054` 终态 Completed；已停止本任务进程组并删除精确 alias。r1/r2 查询均为空，见 `job-terminal.json`、`job-absent-r1.json`、`job-absent-r2.json`。
