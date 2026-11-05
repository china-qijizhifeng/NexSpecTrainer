# Nextron Qwen3.8 PLE/Engram 优化对照

来源：[Nextron PR #2](https://github.com/china-qijizhifeng/Nextron/pull/2)，已合并，标题为 `feat(qwen38): 八层前后向对齐、三层 PLE 缓存与 gigatoken 热点统计`。PR head 为 `185ce49b6a01abc49d4513ca10986dabf7916ceb`，merge 为 `5bd386190567cf2b1dccdf6e1e3c1d42a08c8955`。本地核对 checkout `/Users/shom/Downloads/nextron-qwen38-20261027` 正好在该 head，工作区干净。

实现：[gpu_cache.cu](https://github.com/china-qijizhifeng/Nextron/blob/5bd386190567cf2b1dccdf6e1e3c1d42a08c8955/models/qwen38/ple/gpu_cache.cu)，配套 `models/qwen38/ple/README.md`、`experiments/qwen38/ple_flow/README.md`、`ple_relayout/README.md`、`ple_bandana/README.md`、`ple_e2e/README.md`。

## 实际实现

- 冻结 BF16 PLE 表 95.37 GiB；SSD 保存全表，RAM 用有界 4 KiB page LRU，HBM 放语料统计得到的固定热点行。
- 每批先查热点并去重冷行，仅传输冷行及 inverse map，再在 GPU 组装顺序完整的输出。
- 恰好两个 staging slots。`prepare` 排队 I/O，`acquire` 在非阻塞 copy stream 搬运与组装，compute stream 等待 ready event；`release` 在实际消费者 stream 记录 consumed event，复用 slot 必须等待消费完成。
- 分别约束 HBM、pinned host、page cache、队列和临时映射容量。OS page cache 是额外内存，不能当作已纳入预算。

## 证据边界

H100 上真实表的 65-token、八层前后向链路通过 16,768 个 HF 数值比较；这不等同于大 batch 训练吞吐证明。单独 pinned H2D 达到 48–55 GB/s；pageable→pinned staging 仍有明显开销。2 GiB 热点初始化用了 471 秒、约 28 GB logical reads；连续热点快照尚未实现。物理 frequency relayout 的读性能提升约 1.04–1.22 倍，追加 co-access relayout 没有稳定的进一步收益。

## 对当前 V4.1/H100 的取舍

当前是 CPU 内存常驻的 FP8 Engram、按 rank 分片、gather 后 all-reduce；不是 SSD 冷读。直接照搬 page LRU 或物理重排缺少收益依据。

旧 64K 完整 prefill 的八个 rank trace 汇总见 `evidence/old-trace-aggregate.json`。每个 rank 两次 Engram gather 合计约 5.05–6.78 ms；rank0 后续两次 all-reduce 合计约 9.37 ms，两个 wkv GEMM 合计约 47.56 ms，gate 约 6.08 ms。这里是 CUDA kernel 区间累计，不是端到端关键路径估计，且新配置需要重新核实。

这说明查表预取/缓存即使完全消掉现有 gather，也不足以解释从 3.46 秒降到 2 秒所需的差距。当前优先检查通信、prefill Graph、CP 和主机调度空隙。

值得后续适配的最小方案：哈希生成后，在独立 stream 预取第 1/14 层的本 rank `_owned_rows`；层内等待 ready，保持原 all-reduce 顺序；输出用 consumed event 或正确的 `record_stream` 生命周期保护。64K 两层完整 BF16 输出约 1.5 GiB/rank，不能忽略预算。初版应明确排除 CP、attention DP、Graph、speculative、scheduler staging、peer_read，避免多种数据所有权混用。

只有新 trace 显示查表/搬运成为显著瓶颈时，才继续加入训练语料热点缓存与冷行去重；需要额外计入去重、映射、缓存初始化和查找成本。暂未将此方案接入已测量的优化分支，也不声称 Nextron 缓存已带来 V4.1 性能提升。
