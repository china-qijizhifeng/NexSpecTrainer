# V4.1 Flash 连续 16k / 64k prefill 与完整采集

状态：全部完成。codec、bounded MoE、16k/64k fused indexer 及64k组合测试通过；作业终态 Completed，已删除并确认不存在。

## 固定条件与口径

8×H100，BP node044，Northjob WeaverRSI / proj-32。模型
`deepseek-ai/DeepSeek-V4.1-Flash-1017@1bae0e1088993b9b82de08f59de5be5a34e42170`；
SGLang `38d9e4a3fe4a493b0ae468477a274e83defda81a`，镜像
`lmsysorg/sglang@sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa`。
TP8/EP8、CPU Engram、FlashInfer MXFP4、BF16 full aux37/38/39 + final states。

输入来自两条真实约 80k-token OpenHands 轨迹：row3153 / row3352。窗口都是原轨迹
`[4096, 4096 + tokens)`，没有拼接短窗口。保留 token、loss mask 与监督位置。
`input-identity-receipt.json` 固定原始字节、token/mask hash、offset 和监督 token 数。
每阶段 2 个输入 × 2 次重复 = 4 请求；p95 仅是这 4 次的最大值，不能代表尾延迟。

全程一次完整模型 prefill，关闭 chunked prefill、bounded SWA replay 与 CUDA graph。
context=69632、max-total-tokens=131072、max-prefill-tokens=65536，
显式 SWA/full=0.8（实际 SWA capacity=104704）。使用
`PYTORCH_ALLOC_CONF=expandable_segments:True` 完成 64k。

采集吞吐含 HTTP、Mooncake 取回/解码、校验/hash、压缩写入/fsync、物理对象清理。
GPU forward 为独立同步诊断阶段，取每请求 8 ranks 的最大 CUDA event 时长。
启动、预热、文件回读不计入吞吐；最终文件全部经原生 DSpark reader/normalizer/collator 验证。
Attention、indexer、MoE、Engram 的嵌套计时不能相加当作总耗时。
训练 Torch2.10/TE2.20.2 未改；服务镜像独立使用 Torch2.14.1+cu130。

## Codec：完整采集吞吐

| 连续输入 | 单 worker | 8 workers | 提升 | 8 workers 落盘 p50 |
|---|---:|---:|---:|---:|
| 16k | 1,632 tok/s | 4,078 tok/s | 2.50× | 3.96 s |
| 64k | 1,537 tok/s | 3,302 tok/s | 2.15× | 19.67 s |
| 64k 重复 | 同上 | 3,223 tok/s | 2.10× | 20.12 s |

16k 切换 expandable allocator 后为 4,000 tok/s，输出逐字节相同；按此对照为 2.45×。
长配置下 4k calibration 为 3,779 tok/s。28 个 baseline/codec/repeat 文件均通过字节一致验证。
证据：`evidence/file-byte-parity.json`、`evidence/allocator-parity16k.json`。

原始 scorer 下 GPU forward：16k 934.6 ms，64k 7789.8 ms。
64k rank0 attention 6381.7 ms，MoE 558.6 ms，Engram 总计 72.9 ms；
其中 host gather 7.56 ms、all-reduce 7.89 ms。长输入优先优化 attention/indexer。

## Nextron bounded MoE：省显存，未提速

Nextron `9763a325081d4ff9c2e0df880a339aae33886d88` 的现成 BF16 kernel
针对 Qwen H2048/I512/E256/top8 或 GLM H4096/I2048/E288/top8，不能直接接受
V4.1 H5120/I2304/E384/top6、MXFP4 scales、clamp10 和 EP8。
本实验借鉴其有界 workspace 调度，仍调用原 FlashInfer CUTLASS kernel，
只将本地 MoE token batch 限为 8192；原 routing、权重、shared expert、collective 顺序不变。
这不是原生 Nextron MoE 算子的移植。

| 输入 | 原 GPU forward | bounded MoE | 原 allocated peak | bounded peak |
|---|---:|---:|---:|---:|
| 16k | 934.6 ms | 959.3 ms | 47.33 GiB | 46.07 GiB |
| 64k | 7789.8 ms | 7947.5 ms | 67.37 GiB | 55.10 GiB |

64k 降低 allocated peak 12.27 GiB，forward 慢约 2.0%；采集 3,237 tok/s，
与原版 3,223–3,302 tok/s 波动区间重叠。nvidia-smi 设备显存峰值从 78,145 MiB
降到 65,305 MiB。该口径包含 allocator、库与上下文，不等同于 allocated peak。
16k/64k 共 8 个完整输出文件，aux/final、token/mask 全部逐位一致。
证据：`evidence/moe-parity{16k,64k}.json`、`comparison.json`。

## Indexer：完整模型实测

`--fused-indexer` 使用独立 BF16 Triton kernel，融合 dot、ReLU、head 加权和归约，
不写出 `[rows,32,keys]` 大中间张量；跳过完全不可见的未来 key tile。
Candidate 消费层先按 published block IDs gather K，只评分候选 key。
原候选发布、top-k、索引排序、pool-slot 写回不变。保留参考 BF16 舍入点；
最终 `[rows,keys]` FP32 分数及原 row tiling 仍保留。没有修改模型 attention 范围。

同一服务进程、codec8、原版 MoE、同一输入的配对结果：

| 输入 | GPU forward 原版 → 新版 | Forward 加速 | Indexer 总计 原版 → 新版 | 完整采集吞吐 原版 → 新版 |
|---|---:|---:|---:|---:|
| 16k | 932.6 → 669.3 ms | 1.39× | 398.8 → 135.8 ms | 3,981 → 4,327 tok/s（+8.7%） |
| 64k | 7788.1 → 3207.2 ms | 2.43× | 5735.2 → 1160.1 ms | 3,285 → 4,194 tok/s（+27.7%） |

64k candidate 消费层约 3465.4 → 602.4 ms；full-topk scorer 路径约
1323.2 → 294.2 ms，candidate source 约 946.6 → 263.6 ms。
这些是包含投影、评分、选择等操作的 indexer 方法计时，不只是 kernel 微测。
落盘 p50：16k 4.024 → 3.734 s；64k 19.740 → 15.628 s。
64k HTTP mean：11.584 → 7.033 s。新 indexer 的 allocated peak 为 67.87 GiB，
配对原版为67.49 GiB；没有降低整模型峰值。消除了逐头 scratch 不等同于消除
其他模块的峰值，不能据此宣称整模型省显存。采集受 CPU codec/校验等开销限制，
不能将 2.43× GPU forward 加速写成 2.43× 落盘吞吐。

16k/64k 合计 8 份优化后完整文件与配对原版逐位一致，包括 aux37/38/39、final、
input_ids 和 loss_mask。所有重复请求也一致。
128 份真实层抽查（8 ranks × 8 indexer 层 × 2 长度，每份最多32行）top-k 全部一致。
抽样 logits 最大绝对差0.015625，最大 relative L2 8.49e-6；不是声称所有 logits
逐位相同。该浮点归约差异在本次完整输出中未传播成差异，但两条轨迹不代表所有数据。
完整输出验收：`evidence/indexer-parity16k.json`、`evidence/indexer-parity64k.json`；
逐层抽查：`evidence/indexer-audit-rank*.jsonl`。

H100 算子微测 8 个差分案例的评分和 top-k 完全一致，包括 signed weights、
零可见长度、部分 block、尾部 tile 和 -1 candidate。512×8192 的 scorer
0.802 → 0.111 ms；256×65536 为3.073 → 0.389 ms，candidate 路径0.248 ms。
选择 dense tile=2行×128keys，candidate tile=1行×128keys。

参考 DeepInfer Hopper PR #92 的融合与候选评分方向；本实现没有移植其 FP8
重编码、Gluon 或显式异步 WGMMA 管线。`indexer-plan.md` 固定参考来源与验收条件。

## Indexer + bounded MoE 组合

64k 同时启用两个实验开关：forward **3368.7 ms**，完整采集 **4053 tok/s**，
allocated peak **55.10 GiB**。相对仅 indexer 的3207.2 ms / 4194 tok/s / 67.87 GiB，
用约5% forward 耗时换取12.77 GiB allocated peak 下降。
4份完整输出仍全部逐位一致：`evidence/indexer-moe-parity64k.json`。

组合测试复用刚跑过原 MoE 的服务进程，allocator 保留原先显存块，因此这轮
nvidia-smi 仍显示77,985 MiB；不能将 allocated peak 下降写成设备驻留显存同步下降。
前面单独 bounded MoE 的独立启动才观察到较低设备占用。
建议性能优先时只启用 `--fused-indexer`；显存预算紧张时配合
`--moe-chunk-tokens 8192`，在干净进程启动下部署并验收。

## 最终检查与资源清理

最终代码的8个GPU差分案例再次通过，`indexer-final-validation.json` 中评分、
top-k 全部一致；本地源码 SHA256 与实际运行 `execution-files.json` 一致。
Black、Ruff F/I、compileall 和 git diff --check 通过。前序 codec/capture
测试在原4k报告中记录，本次未改这些模块。

本轮累计56份落盘输出经过原生训练reader回读，其中12份indexer/组合优化输出
与各自配对原版的完整tensors逐位一致。全部原始证据留在
`/gpfs/users/linlizhi/nexspec-v41-prefill-long-20261103`，小型证据镜像在本报告同级 `evidence/`。

作业 `job-0263536b7bda-20261103125905` / alias
`nexspec-v41-prefill-long-20261103-r1` 结束前已验证自有进程组并终止服务和Mooncake。
平台终态Completed；按精确alias删除后查询返回空。
清理回执：`job-terminal.json`、`job-delete.json`、`job-absent.json`。

## 排障记录

- 初始 prefix=0 的 4k 窗口没有监督 token，改用原已验证 offset4096，并断言 4k 输入完全一致。
- V4.1 有效 SWA ratio 默认0.1，预先发现容量不足并显式改0.8，未缩短64k请求。
- 64k 原 allocator 在 CUTLASS 15.03 GiB workspace 分配 OOM；静态核对后仅切换
  expandable segments，完成 full64k。16k allocator 前后文件一致。
- MoE 首次安装遇到旧 runtime module 缓存、随后 packed TopK 类型过窄；每次失败暂停，
  对 pinned producer/consumer 契约静态核对，补真实类型回归后固定源码重启，保留失败日志。

原始输入、权重、激活大文件仅留 GPFS，不纳入 Git。工作在独立
`/Users/shom/Downloads/NexSpecTrainer-v41-prefill`，未提交/推送/合并。
