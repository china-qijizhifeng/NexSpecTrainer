---
name: collect-hidden-state
description: 在 NexSpecTrainer 中采集完整轨迹 hidden states，复用 SGLang/Mooncake 有界流水线，配置原生压缩、断点续采与吞吐计量。用于离线特征采集和采集性能调优；不用于启动训练或判断 CP 数值等价。
---

# Collect hidden state

优先使用仓库已有的流水线，不把调试用串行验收入口直接扩成正式采集。

## 入口和数据契约

- 通用服务采集：[`scripts/prepare_server_hidden_states.py`](../../../scripts/prepare_server_hidden_states.py)，CLI 与格式说明见 [server capture](../../../docs/sections/examples/server-capture-to-disk.md)。
- 流水线本体：[`capture_pipeline.py`](../../../specforge/inference/capture_pipeline.py)。`depth=2` 是单 producer + 单 consumer：下一条请求与当前条 CPU 持久化重叠，最多两条在途；不等于两个压缩 writer，也不代表 GPU 能持续满载。
- Northgate 已 tokenize 完整轨迹：[`short_capture.py`](../../../ops/northgate_capture/short_capture.py)、[`pipeline_io.py`](../../../ops/northgate_capture/pipeline_io.py)。这是固定数据集的专用入口，不用于泛化到其他模型或重新分词。
- 保留原 input_ids/loss_mask、样本 ID、模型 revision、目标层语义；用户要求完整上下文时，先完整 prefill 后保存，不因训练切片长度而截断目标上下文。
- 分片按 token 负载均衡并保存精确样本集合；多副本使用平台提供的 rank，每个 worker 隔离安装目录、缓存、Mooncake、日志与输出。不能让多个副本写同一输出目录。

## 压缩与校验策略

先明确用户要求的质量/格式与回读策略，不擅自把完整解压校验加到每条热路径。区分：模型重复性验收、压缩 codec 测试、生产采集回读、文件身份记录。这些不是同一项工作。

用户取消生产回读时：不解压、不逐 tensor 比对、不因监控再把它加回来；保留请求长度、shape/dtype、输入/mask 身份与原子写入所需的检查。记录 `readback_performed=false` 和 `native_roundtrip_exact=null`，用 `CAPTURED.json` 表示写入完成，不能写成 `VERIFIED`。历史已通过的 gate 可保存为带来源的旧证据，不冒充当前 run 重新通过。

当前 native hybrid 参数与范围：

| 参数 | 当前值/行为 | 调优要点 |
|---|---|---|
| Brotli quality | 10（本次用户要求） | 未获授权不降质量；不把高质量等同于已调优 |
| Brotli chunk | 256 KiB 默认 | 对比 64/256/1024 KiB 的耗时、体积、峰值 RAM |
| compression workers | 通用 CLI 默认最多12；Northgate 优化配置32 | 对比8/16/32，按真实 CPU 配额而非宿主机核数设置 |
| Brotli window | `lgwin=22` | 每块独立；窗口大于块长不等于跨块共享字典 |
| Zstd mantissa | level1、单线程、1 MiB块 | hybrid 不读取 `SPECFORGE_ZSTD_WORKERS`；该变量只影响另一套Zstd codec |
| pipeline depth | 正式流水线2；通用CLI默认1 | 显式设置2，不能仅凭存在pipeline代码就声称已启用 |

当前 hybrid 的跨层 aux tensor 使用条件编码，符号/指数 Brotli、尾数 Zstd；没有跨层布局的 final hidden 与普通字段走标准 fields/raw 的 Zstd 路径。不要把所有 tensor 都描述为同一压缩分支。

参数值是起点，不是最优值。先用固定真实 tensor 测拆流/排序、压缩写盘、传输、file hash，再比较参数；只有实验中需要时做 codec 回读验证。采集单条的 `seconds` 含排队，不能直接累加它推算流水线墙钟吞吐。看每个 worker 的完成时间差、tokens/s 和最慢分片 ETA；样本由短到长时不能按早期条/分钟外推全量。

## 容量、切换与完成

- 原始字节按实际 tensor 宽度、dtype 与 passthrough 算；depth2 保守预留两条 source object，无压缩率假设。CPU 还需容纳 tensor、条件排序索引、压缩临时流和待写文件。
- context/prefill/pool 是调度容量，数据上限是另一字段。读取实际镜像的 admission/page 预留契约和启动回读，不只看启动 argv。Northgate 65,536输入上限配69,632池容量只是本模型镜像的已用配置。
- 仪电训练型采集按当前平台操作 skill：固定 clean source/image，prepare → submit dry-run → execute → detached reconcile。用户授权的 GPU 规模是边界，替换旧作业先确认其精确 JobId 和所有副本终态。
- 不修改运行中的冻结源码。新版本用新 run；[`resume_files.py`](../../../ops/northgate_capture/resume_files.py) 要求旧作业的终态回执、新 run 尚未提交、相同副本数/镜像，仅迁入有完整 receipt 的文件。它用 hardlink 保留已有产物，不解压或重新采集完成文件；半成品保留旧现场，在新 run 重采。
- 完成依据是精确记录/token覆盖、最终manifest和写入回执，以及Job与全部副本终态。Running不是完成；禁用回读时不要在报告中声称逐tensor解压验证过。
- 修改作业身份或回读策略时，同时更新已存在的监控任务和状态文档，避免监控旧作业或错误要求VERIFIED。

## Northgate 开关

`short_capture.sh` 自动隔离 `worker-N`。正式续采使用：

```text
NORTHGATE_PIPELINED_CAPTURE=1
NORTHGATE_RESUME_RUN=<先前已完成模型/codec gate的不可变run>
NORTHGATE_COMPRESSION_WORKERS=32
SPECFORGE_HYBRID_NATIVE_BUCKET=1
NORTHGATE_HYBRID_CHUNK_SIZE=262144
```

该模式复用旧gate并记录来源，调用现有depth2流水线，写入阶段不解码。`CAPTURED.json` 记录 request、transfer、queue、encode/write、file hash 秒数及生产/消费时间戳。模型或层契约变化时不能复用旧gate；是否做新的验收遵循用户的当前要求。

2026-11-06 的真实64H200续采（源码`22a0948`）已迁入97条完成记录并开始新采集。首批约19K–21K-token记录中，6个worker的第二条`produced_at`早于首条`written_at`，确认请求/传输与CPU写入实际重叠；这些worker完成间隔约35–40秒，请求约1.8–2秒、传输约0.4秒、编码写盘约34–39秒、file hash约0.4秒。另一worker首条编码约72秒。这只是首批实测，不是最优参数证明，也不是同样本受控加速比；CPU压缩仍决定吞吐，增加GPU或depth并不能自动解决。

## 计数分桶优化实测（2026-11-06）

使用同一生产镜像、64 CPU 配额，对固定 19,006-token 真实产物运行 `ops/northgate_capture/codec_bench.sh`（`CODEC_BENCH_NATIVE=1`）。Brotli quality10、256KiB 不变：

| 实现 | 线程 | 压缩写盘 | 文件字节 |
|---|---:|---:|---:|
| NumPy stable argsort | 16 | 33.70秒 | 489,838,893 |
| native stable counting bucket | 16 | 18.82秒 | 489,838,893 |
| native stable counting bucket | 32 | 13.36 / 13.44秒 | 489,838,893 |

分桶阶段从16.60秒降至1.23–1.31秒；32线程的Brotli阶段约8秒。所有256KiB输出文件SHA完全相同，单样本加速约2.5倍。64KiB为12.85秒、490,155,631字节，收益较小，当前保留256KiB。此为受控单样本结果，不代表全量或每个GPU节点统一加速。

`SPECFORGE_HYBRID_NATIVE_BUCKET=1` 显式启用 [`hybrid_bucket.py`](../../../specforge/data/hybrid_bucket.py) 的C稳定计数分桶，保持上下文内字节顺序；未启用继续用NumPy。启动时先运行 `python3 -m specforge.data.hybrid_bucket`，以便在模型加载前暴露编译器问题；编译产物按源码hash和架构缓存在节点临时目录，不写冻结源。需要C编译器。开发测试覆盖全部BF16位模式、空输入、随机上下文及非法context；生产不因此恢复每条回读。

显式指定压缩线程；实测容器 `os.cpu_count()` 为1，但affinity与cgroup配额均为64。保存CPU affinity、配额和codec开关到environment.json，每条新回执记录native_bucket。不要仅凭宿主机核数或慢worker耗时断言CPU争抢/NUMA原因。
