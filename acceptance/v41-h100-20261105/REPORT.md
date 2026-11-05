# H100 V4.1 Flash：64K 冷 TTFT 向 2 秒优化

普通推理已达到 2 秒目标：完整 prefill 的64,326-token冷TTFT为 **1.54430s**（run008），复测 **1.50406s**（run009）。基本 Compact 已接入验证，CP8为 **1.20040s**，增加W4A8后 **1.03491s**。这些是推理结果，**不是全部 prompt hidden 采集耗时**。

用户确认训练需要每个 prompt token 的 hidden。本轮实现：Compact在full hidden模式自动关闭，prefill保持整块，MoE保留原GEMM行数并复用配置容量的workspace；通过SpecForge/Mooncake检查aux37/38/39及最终hidden、token IDs和loss mask。

## 固定口径

- 模型 DeepSeek-V4.1-Flash-1017，目标模型 TP8/EP8，8×H100 80GB，node067。
- 64K 指同一条 64,326-token 输入；每次冷测 flush cache，并断言 cached_tokens=0，报告三次中位数。
- 额外测 4K/16K、热缓存及 8 条输入×2 次、最多 128 输出 token 的实际生成。遵守 EOS；端到端 TPS 需另看输出长度一致的子集。
- 不运行任务质量分数。数值对拍用于避免 kernel 适配错误。
- run001–008显式保留完整prefill；run009新增用户授权的Compact推理对照，独立标注。Compact不与完整prefill数值等价；完整hidden采集必须回退完整计算。
- 所有输入有 SHA256；已提交的实验目录及清单不修改。独立 Northjob：`job-58a7093f827c-20261105131049`，项目 WeaverRSI、队列 proj-32。

## 已验证结果

| 实验 | 结果 | 证据 |
|---|---|---|
| 同节点已有优化基线 | 4K 0.28334s /16K 0.88848s /64K 3.97746s | runs/001/same-node-baseline/summary.json |
| I/O+mHC 整模型 | 4K 0.27689s /16K 0.85105s /64K 3.46133s | runs/005/io-mhc-full/summary.json |
| I/O+mHC+NVLS+16K Graph 整模型 | 4K0.28940s /16K0.66868s /64K2.69374s；64K样本2.69063/2.69374/2.69576 | runs/005/io-mhc-nvls-graph16k/summary.json |
| I/O+mHC+NVLS 整模型 | 4K 0.27652s /16K 0.87517s /64K 3.41752s；64K样本3.26093/3.70670/3.41752 | runs/005/io-mhc-nvls-full/measurements.json |
| 64K 新 causal score/mask/select/write 链路 | 58.2153→27.3654ms，缩短53.0% | runs/003/prefill-io-bench.json |
| 16K 同链路 | 4.6287→4.3723ms | 同上 |
| CP8 形状的局部链路，8041 query rows | 7.4569→3.8800ms；不是整模型CP结果 | 同上 |
| 64K mHC post/combine/RMSNorm | 3.0146→2.2102ms，缩短26.7% | runs/003/mhc-prefill-bench.json |
| namespace/dispatch/运行时索引检查 | 154 tests、2 subtests 通过 | runs/003/worker.log |

微测不能替代整模型首 token 延迟。mHC 数值检查要求 BF16 post 完全一致，归一化结果 rtol/atol=1/128，另测改输入后的 CUDA Graph replay。

I/O+mHC 的实际运行时 GPU 测试在 run005 通过 2 项（4096/64326 rows），包含改变输入后的 Graph replay 和 CP fallback。基线与 I/O+mHC 两边都输出128 tokens的12组配对请求，aggregate TPS27.83945→29.89598，median TPOT7.7524→7.7491ms。全16次aggregate TPS26.32879→26.92999；3组输出长度变化，因此不把全量比例直接当作固定工作量加速。

## 新改动

1. `SGLANG_OPT_DSV41_SM90_PREFILL_IO`：按 score tile 的最大可见长度裁掉未来 K，保留 top-k 最小宽度及 64 列对齐；Triton 原地 mask 只写未来位置；一个 kernel 同时写回原始索引和可选 page slots。保留候选块与无效索引契约。
2. `SGLANG_OPT_DSV41_SM90_PREFILL_MHC`：Hopper opt-in 使用已有 CUDA post/combine/RMSNorm prefill kernel；仅4096–65536行、hidden5120、4 streams、符合norm契约、无CP/attentionDP的边界。
3. 新算子注册为 attention 组的 KDA/SM90 backend，runtime 经 ops facade 调用。所有新开关默认关闭。

## 实验限制与排除项

- run001 的 CP8 配置在模型初始化被视觉分支门控拒绝，不能记作 CP 性能差。后续候选明确使用 language-model-only，只代表纯文本服务。
- V4.1 暂不支持直接叠加 interleave CP 与 prefill Graph；通用注册列表虽列有interleave+DSA，模型适配器实际只放行GLM。
- run002 的 mHC 微测缺少 standalone 并行上下文，未运行到 kernel。静态检查后在run003补齐初始化并通过。
- run004 的新集成测试调用了不存在的 ParallelContext.scoped；在run005改为实际override API。run004未启动整模型；生产源码不因该测试错误改变。
- 原同机基线生成 aggregate TPS 26.3288；带 prompt logprobs、遵守EOS，不能直接把输出长度变化计为性能改善。

Graph16K实际捕获耗时约4.7秒，占7.96GB，剩余约26GB；首块trace出现50次cudaGraphLaunch。该profile只录到首个16K块，不能把其累计kernel时长当完整64K。完整64K首token测量无此限制。Graph16K相对基线，12组配对满128token请求aggregate TPS27.83945→36.42897（+30.9%），TPOT约7.77ms。

- run005 CP8 在权重加载后因 decode Graph 桶 `[1]` 不满足 CP8 对齐而失败；run007 的独立配置回归已通过2项，改用桶8、请求池8，实际仍逐个发送单请求。
- run006 关闭 expandable segments：4K冷中位数0.22681s，16K0.61855s；64K warmup因MoE申请14.75GiB连续workspace时碎片OOM，无64K结果，第二个NVLS变体未运行。剩余约7.3GiB，另有16.75GiB reserved/unallocated，不能据短输入收益推荐该全长配置。已静态核对MoE transport、FlashInfer workspace和CP激活切分：CP下MoE仍gather全序列，不能把workspace估为1/8；仅attention/残差激活分片。

## Nextron 与分配器线索

已找到并核对 Nextron PR #2 的三层 PLE 缓存，分析见 [NEXTRON-ENGRAM.md](NEXTRON-ENGRAM.md)。当前 Engram gather 和其通信仅约十余毫秒；双缓冲预取值得借鉴，但 SSD page LRU 不是本工作负载的大头。

旧64K trace 的 GPU 时间线出现大约0.9秒的主要空隙，对应 `aten::empty` 等分配操作。其中单个MoE workspace分配对应约0.53秒空隙。静态检查确认基准每次 `/flush_cache` 同时调用 `torch.cuda.empty_cache()`，实验启动脚本设了 `PYTORCH_ALLOC_CONF=expandable_segments:True`。run006 将保持全部冷测操作和完整prefill，只对照关闭 expandable segments；不靠省掉清缓存改变测试口径。

32K Graph 对照：64K冷TTFT2.82828s，样本2.82828/3.05720/2.63296；占15.74GB Graph显存，剩余约18.3GB；全长profile两块共100次GraphLaunch，累计kernel约2.372s。没有优于16K，暂不推荐。

已生成 `launch-best-verified.sh`，须配合本目录固定镜像和源码overlay；脚本不是生产部署动作。

CP8桶8原生分配器：4K0.24306s、16K0.41178s；64K整块warmup仍因14.77GiB MoE workspace碎片OOM，free12.59GiB/allocated48.34GiB/reserved-unallocated14.78GiB。此项不能作为64K性能结果。静态确认CP分块metadata含extend长度与prefix，run008改为16K分块，将单次MoE workspace限制在已通过的规模，保持完整prefill。

## 完整 prefill 与 Compact 推理结果

| 配置 | 64K冷TTFT中位数 | 说明 |
|---|---:|---|
| CP8，原生分配器，16K prefill分块 | 1.544297s | run008；三次1.644984/1.505558/1.544297，完整计算，无Compact |
| 同配置复测 | 1.504059s | run009；三次1.503558/1.514516/1.504059 |
| Compact128，CP8，16K分块 | 1.200400s | run009；推理专用 |
| Compact128，16K Graph，无CP | 1.354588s | run009；解码更快，端到端TPS更高 |
| Compact128，CP8，复用workspace，W4A16 | 1.216623s | run010；未提升TTFT，但降低采样HBM峰值 |
| Compact128，CP8，复用workspace，W4A8 | 1.034914s | run010；三次1.034082/1.040660/1.034914 |

run008相对run001，双方均输出128token的12组配对请求：27.83945→43.29095 TPS；median TPOT7.752→10.324ms。CP必须使用decode bucket8，单请求decode变慢，但长prefill整体收益更大。

run009将生成请求的prompt logprobs统一关闭，以免触发Compact的完整prefill回退；不得将这些TPS直接与run001/008相比。在run009相同12组128token配对输入上：完整CP46.3742TPS，CompactCP51.6560TPS，CompactGraph56.4647TPS；TPOT分别10.185/10.181/7.619ms。遵守EOS，没有跑任务质量分数。输出hash和长度变化记录在comparison.json，输出差异不等同于质量结论。

NVML每100ms采样，覆盖就绪后至profile：完整CP最高约57.53GiB/卡，CompactCP57.57GiB，CompactGraph57.02GiB。它是设备已用HBM采样，不是PyTorch精确分配峰值，可能漏掉瞬时峰值。

run008全长profile GPU span1.530s、busy1.406s，各rank NCCL累计383–436ms，indexer scorer36.5–36.9ms。证据cp8-chunk16k-trace-aggregate.json；CP通信仍是后续可优化项。

## DeepInfer PR #92 与后续适配

固定参考head `e205d42e4b28185e50ce30f6b7204cd01e59974c`，分析见DEEPINFER-PR92.md。其8×H20的64K1.058s采用Compact keep140与DSpark，不能拿来当完整prefill对照。本实现复用SGLang已有的Compact128，不声称精确移植keep140。

SM90没有SM100 mega_mhc可直接搬；PR的DeepEP V2/EPBuffer接口也与当前镜像不同。当前镜像DeepEP只有Buffer/EPHandle，实际EP dispatch/combine尚未验证；未通过删除兼容性门控强行启用。先适配FlashInfer0.7.2现有workspace接口和W4A8路径。

run010通过FlashInfer Humming W4A8路径，workspace复用量为2.82GiB/卡，W4A16为3.76GiB/卡。两项NVML设备占用采样最高55.37/55.33GiB，均低于run009未复用workspace的约57.57GiB。相同输出128token配对请求W4A16/W4A8为50.218/55.779TPS，TPOT10.190/9.909ms。W4A8改变activation精度，当前只作为可选推理配置；完整训练采集验收先固定W4A16。

## 全 prompt hidden 采集适配（数值验收未通过）

- SpecForge插件要求chunked-prefill-size=-1，因此上述16K prefill分块性能不能直接作为训练采集性能。
- SGLang full hidden模式在构建KV池和模型前关闭decoder SWA bounded replay，避免DSpark capture convention落到在线尾窗口路径。
- 新MoE开关`SGLANG_OPT_DSV41_SM90_MOE_WORKSPACE`默认关闭；保留原GEMM行域，以配置的prefill容量一次分配workspace，上限65536行/16GiB，每设备/运行时只持有一块。不兼容的形状/stream、TBO/SBO、Graph capture使用vendor原分配路径。16K推理仍用原有较小容量。
- SpecForge writer限制为attention TP0且CP0，避免CP gather后8次重复发布。
- run012的MoE内部16K分块方案已拒绝并从当前生产源码撤掉：4096-token hidden逐位一致，但16385-token aux cosine仅0.9820109411、relative L2为0.1888424660、max abs为52215.0625。阈值未放宽，64K未执行。run013合成EP测试通过并不能解释或否定整模型误差，根因尚未确认。
- run012在数值失败前通过2项GPU、7项配置、34项插件测试及26个subtests。run014验证恢复原GEMM行域后的完整容量workspace，16项生产GPU测试和6项实际几何测试全部通过；run015使用expandable_segments=True进行真实采集。

实验已停止；worker正常退出，GPU任务清理回执见cleanup目录。

后续候选：CP auxiliary capture 已不使用 pre_hc_head，但 eager runner 当前仍先 gather 再丢弃。64,326×20,480×2 字节约2.45GiB/卡，准备在 gather 前去掉该未使用张量；当前尚未安装或宣称收益。

run015同样在16385-token对拍失败：aux cosine0.9803202676、relativeL2 0.1974426872，64K仍未执行。更关键的是，run012与run015的workspace关闭对照在16385上hidden指纹不同、4096上相同；不能继续将失败单独归因于workspace或分块。静态发现DeepSelect BF16 ties使用atomic选取，sorted_index仅排序已选索引。run016将重复选择、关闭DeepSelect后的整模型重复采集，以及同一forward真实MoE输入workspace开关对拍一起用于诊断。

## 收尾结论

用户要求停止扩展优化，合并NexSpecTrainer main，包含固定版本SGLang补丁。

- 生产SGLang改动固定为`ed529db58e00e44132f3f4698688319c09a67cd8`，基于`38d9e4a3fe4a493b0ae468477a274e83defda81a`；42文件patch经过apply、逐文件hash和reverse验证。原始本地SGLang main未修改。
- run016对真实模型每层同一MoE输入分别执行workspace关闭/开启，8 ranks × 240次，共1920次逐位一致。这项排除了所测输入上workspace输出差异，不能替代完整模型采集验收。
- 同一进程关闭DeepSelect后，4096-token重复hidden一致；16384/16385-token的aux relativeL2仍为0.2174066/0.2014754，最终hidden relativeL2为0.2732664/0.2722243。run016的退出0仅表示诊断收集完成，**不表示数值验收通过**。
- run017复现DeepSelect cluster路径ties选择不稳定：1/8/64行、k512/1024、width16512。它是已证实的选择重复性限制，但关闭DeepSelect仍存在长输入漂移，不能断言它是唯一根因。
- 64K完整hidden采集尚未执行，不能发布其耗时或声称达到2秒；完整训练采集不要启用本轮可选推理配置，默认优化开关保持关闭。
- 新MoE内部拆分及省略pre_hc_head gather的候选都未纳入补丁。本轮不再扩展实验，保留未决数值问题及复现实验清单。
