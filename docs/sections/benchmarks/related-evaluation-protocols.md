# 投机解码评测协议调研与平台适配参考

> 归档说明：本记录于 2026-11-09 收录到 NexSpecTrainer，保留 2026-11-02 的调研结论与版本边界。它不是当前实现状态或已复现的 benchmark 结果；执行实验前应重新核对对应版本与入口。
> 来源会话：`01a19ad7-5f41-7bb0-ba02-0d5bd8ec04ea`（梳理投机采样评测方式）。

核对日期：2026-11-02。文档类型：研究参考与设计解释。适用对象：NAR、NexEval、NorthGate、SGLang 的投机解码测试设计。

本文核对论文正文、附录和部分官方实现，未运行 GPU 实验，未复现论文数值。正文中的“论文披露”“代码核对”“建议”分别表示来源事实、固定源码快照的静态结果和待实现方案。“未披露”表示已核对的资料没有给出该值，不能用库默认值补齐。

配套设计：[NAR + NexEval + NorthGate + SGLang 投机解码评测设计](./nar-nexeval-northgate-sglang-design.md)。

## 1. 研究结论

建议将平台评测分为三种协议，并分别保存结果：

1. **固定前缀生成**：不同方案读取相同的真实请求历史，调用真实模型生成一次后续输出。用于比较草稿接受情况、推理耗时及上下文泛化。
2. **服务负载评测**：以明确的并发或到达过程发送真实内容，比较端到端延迟、总吞吐量、失败率和满足延迟目标的有效吞吐。
3. **真实 Agent 执行**：从相同任务初始状态分别运行各方案，由原生 verifier 评分；同时记录模型、工具、环境和排队时间。

SPEED-Bench 对第 2 类最有参考价值；TraceSpec 提供第 1、3 类的对照方法，但其公开实现的编辑、测试反馈存在模拟，且不评分任务正确性；DFlash 提供新草稿结构在研究后端、生产引擎和长上下文中的不同实验设置。

需要避免四种混用：

- 会话并发、HTTP 在途请求数与 SGLang 实际调度 batch size 混用。
- 草稿 token 接受数、每轮最终输出 token 数与 SSE chunk 大小混用。
- 返回录制模型响应的运行时回归，与向真实引擎发送固定前缀的推理评测混用。
- 论文原始实验设置与今天官方仓库的默认参数混用。

## 2. 资料与版本

| 研究 | 固定论文版本 | 主要用途 | 官方实现核对 |
| --- | --- | --- | --- |
| SPEED-Bench | [2604.09557v2](https://arxiv.org/html/2604.09557v2)，页面版本日期 2026-05-28 | 语义多样性、长度桶、生产吞吐 | NVIDIA Model-Optimizer `d48a5450d70e7ec4d9ea22bc6ce4d403fef37a12` |
| DFlash | [2602.06036v2](https://arxiv.org/html/2602.06036v2)，本次查询的最新论文版本 | 并行草稿、thinking、服务引擎、长上下文 | z-lab/dflash `07ebd93db9f472af339b644bb70221ad8428328a`；已包含 DFlash 2 |
| TraceSpec | [2610.41177v1](https://arxiv.org/html/2610.41177v1)，2026-10-30 | Agent 动态执行、固定前缀、分层统计 | Robot2050/TraceSpec `1acb4be52e028b2a1aed550b187c227dc8b87cbe` |
| Scaling Up, Speeding Up | [2509.04474v1](https://arxiv.org/html/2509.04474v1) | Best-of-N、多轮推理 | 本次主要核对论文，未完成独立实现核对 |
| Speculative Acceptance Does Not Decay with Position | [2610.30644v1](https://arxiv.org/html/2610.30644v1)，2026-10-22 | 完整 CoT、配对分析、长度偏差 | 本次主要核对论文，未完成独立实现核对 |
| Only Accepted Tokens Count | [2610.40287v1](https://arxiv.org/html/2610.40287v1)，2026-10-29 | 能耗、验证成本、固定 batch | 本次主要核对论文，未完成独立实现核对 |

原始论文 HTML、文本提取、仓库版本信息和所查代码保存在调研工作区 `/Users/shom/Downloads/speculative-decoding-evaluation/research-sources/`，未随本文收录到仓库；可追溯的公开来源与固定版本链接保留在下文。代码链接均固定到上述 SHA。固定模型名称不能替代模型 revision；本文引用的多数论文没有给出全部模型文件的不可变摘要。

## 3. SPEED-Bench

### 3.1 数据与切分

来源：[论文 §4–6、附录 B](https://arxiv.org/html/2604.09557v2)、[官方数据卡](https://huggingface.co/datasets/nvidia/SPEED-Bench)。

| 字段 | 论文披露 |
| --- | --- |
| Qualitative split | 18 个公开来源，11 类，每类 80 个样本，共 880 个 |
| 类别 | Coding、Humanities、Math、Multilingual、QA、RAG、Reasoning、Roleplay、STEM、Summarization、Writing |
| 选样原则 | 以语义多样性为目标选择紧凑样本；保留 source、src_id、subcategory、difficulty、multiturn 元数据 |
| 多轮 | 约 20% 样本含 2–5 轮交互 |
| Throughput split | 1K、2K、8K、16K、32K 五个输入长度桶 |
| 每桶规模 | Low Entropy、Mixed Entropy、High Entropy 三类，每类 512 个，共 1,536 个；五桶合计 7,680 个 |
| 长度构造 | 使用 `o200k_base` 计算长度，按来源截断或追加 `please answer now` 中性后缀 |
| 输出长度描述 | 数据构造时验证平均可生成长度；Qualitative 约 650 token，Throughput 在 GPT-4、16K 输入条件下约 2.4K token。这些值不等于每个评测模型的实际输出长度或上限 |

Qualitative 来源包括 HLE 文本题、LiveCodeBench Lite、Code Contests、HumanEvalPack、RoleBench、CoSER、WritingBench、Creative Writing V3、MT-Bench 101、MMLU-Pro、MMATH、OPUS-100、MCIF、ChatRAG-Bench、CNN/DailyMail 和 SpecBench 等。Throughput 使用 BAMBOO、Gutenberg、WritingBench、AdaLEval、HLE、Long Code Arena 和 RepoBench 等来源。详细构造按论文附录 B 的原表执行，不能直接取同名原始数据集代替。

**代码核对：**数据卡有明确占位字符串 `FULL BENCHMARK DATA SHOULD BE FETCHED FROM THE SOURCE USING SPECDEC_BENCH`。只下载 Hugging Face parquet 不一定得到完整 prompt。官方入口是 [prepare_data.py 所在目录](https://github.com/NVIDIA/Model-Optimizer/tree/d48a5450d70e7ec4d9ea22bc6ce4d403fef37a12/examples/specdec_bench)，README 给出的数据准备命令为 `python3 prepare_data.py --dataset speed --config all`。平台导入后应校验占位文本、样本数和正文摘要，并冻结补全后的数据。

### 3.2 模型、硬件与推理引擎

来源：论文 §8、附录 I，表 11–13。

| Target | TP / EP | Draft 或方法 |
| --- | --- | --- |
| `openai/gpt-oss-120b` | 1 / 1 | EAGLE3：`nvidia/gpt-oss-120b-Eagle3-long-context`；附录 N 另测 `lmsys/EAGLE3-gpt-oss-120b-bf16` |
| `meta-llama/Llama-3.3-70B-Instruct` | 1 / 1 | Vanilla：`meta-llama/Llama-3.2-1B-Instruct`；EAGLE3 路径见下方差异说明 |
| `Qwen/Qwen3-Next-80B-A3B-Instruct` | 8 / 8 | Native MTP；EAGLE3：`lmsys/SGLang-EAGLE3-Qwen3-Next-80B-A3B-Instruct-FP8-SpecForge-Meituan` |
| `Qwen/Qwen3-235B-A22B` | 8 / 8 | Vanilla：`Qwen/Qwen3-0.6B`；EAGLE3：`nvidia/Qwen3-235B-A22B-Eagle3` |
| `deepseek-ai/DeepSeek-R1` | 8 / 4 | Native MTP |

论文主要比较 N-gram、Vanilla SD、EAGLE3、Native MTP，主实验采用链式草稿。树形验证单列于附录 F。通常使用单张 B200；DeepSeek、Qwen 推理与 GPT-OSS 草稿训练使用 8 张 B200。

论文引擎版本为 TensorRT-LLM `1.2.0rc1`、SGLang `0.5.7`、vLLM `0.13.0`。表 1 中非 Vanilla SD 的 `T=1` 使用 TensorRT-LLM `1.2.0rc7`，因为 rc1 不支持该组合。

**版本差异：**论文表 12 的 Llama EAGLE3 项链接为 `yuhuili/EAGLE-LLaMA3-Instruct-70B`；当前官方 README 示例为 `yuhuili/EAGLE3-LLaMA3.3-Instruct-70B`，且示例 TP=8。应向作者的实验配置核对，不能自动把 README 示例替换为论文原始条件。

### 3.3 生成与负载

| 字段 | 论文披露 | 当前代码核对 |
| --- | --- | --- |
| 温度 | 默认 0；表 1 包含 1 | `--temperature` 覆盖 runtime YAML；两者未设时默认 greedy |
| reasoning | GPT-OSS 默认 medium；§8.2 词表裁剪实验用 low | 支持 `chat_template_args`，应显式保存 |
| 主表负载 | BS=32，draft length=3 | `run.py` 用 `asyncio.Semaphore(concurrency)` 包住整段多轮会话；实际 batch 由引擎决定 |
| 扫描 | 图 6 的 BS 1–128；图 7 的 BS 2–256，并比较 DL=1、3 | 客户端并发和 `max_running_requests` 等参数共同限制负载 |
| 输出上限 | 主文未统一给出所有表的精确上限 | `--output_length` 默认 4096；README 示例也使用 4096。该默认值不能证明所有论文实验如此 |
| EOS | 未统一披露全部实验的策略 | `--ignore_eos` 可选；默认使用 tokenizer EOS |
| 到达模型 | 未披露 Poisson、固定 QPS 或真实时间戳回放 | 当前代码预建全部会话任务，以 semaphore 有界执行；属于有限数据集的闭环并发 |
| 预热 | 论文未给出完整策略 | 已查 `run_simple` 中引擎初始化后直接进入运行循环，未见显式 benchmark warmup；引擎初始化可能含 CUDA graph capture，不能等同工作负载预热 |
| cache | 论文未给出统一的 prefix cache 隔离策略 | adapter 沿用引擎配置；应补充冷 cache、暖 cache 的独立协议 |

源码：[run.py](https://github.com/NVIDIA/Model-Optimizer/blob/d48a5450d70e7ec4d9ea22bc6ce4d403fef37a12/examples/specdec_bench/run.py)、[SGLang adapter](https://github.com/NVIDIA/Model-Optimizer/blob/d48a5450d70e7ec4d9ea22bc6ce4d403fef37a12/examples/specdec_bench/specdec_bench/models/sglang.py)。该 adapter 直接调用 `sgl.Engine.async_generate(input_ids=..., stream=True)`，并非 HTTP Chat Completions 客户端。

### 3.4 指标与计时

论文定义条件接受率 `AR_i` 为此前缀已接受条件下，第 i 个草稿 token 的接受概率。接受长度 AL 包含每轮额外输出的 token：

`AL = 1 + Σ(i=1..γ) Π(j=1..i) AR_j`。

平台只应在相同的链式提议和计数约定下使用这个公式。树形、多候选和可变长度草稿需保存自身原始计数。

论文 §7 使用流式返回对象中的新增 token 数推断步长，并记录对象到达时间。当前 SGLang adapter 使用 `meta_info.completion_tokens` 的累计差值构造每个 chunk 的 token 段。**这一推断依赖引擎流式输出与验证步的对应关系；经过 HTTP、NorthGate 或 NAR 的缓冲、拆分、合并后，该关系不能假定成立。**

当前 [SpecBench 聚合器](https://github.com/NVIDIA/Model-Optimizer/blob/d48a5450d70e7ec4d9ea22bc6ce4d403fef37a12/examples/specdec_bench/specdec_bench/metrics/specbench.py) 先对一个样本的全部轮次和输出步求平均 AL，再对样本平均，最终是样本等权。它与 TraceSpec 的总计数相除不同。

当前 [Timing 聚合器](https://github.com/NVIDIA/Model-Optimizer/blob/d48a5450d70e7ec4d9ea22bc6ce4d403fef37a12/examples/specdec_bench/specdec_bench/metrics/timing.py) 使用：

- 总 Output TPS：所有输出 token / 最早请求开始至最晚结束的时间。
- TTFT：首次输出对象到达时间减请求开始时间。
- User generation TPS：`(tokens - 1) / (最后到达时间 - 首次到达时间)`，依赖首个输出对象的计数约定。
- Generation step time：输出事件间隔；当前实现的间隔列表还包含首个输出前的间隔，不能直接当作纯 TPOT。
- 默认统计分位数为 P25/P50/P75，不能声称默认已报告 P95/P99。

论文未披露完整性能表的独立重复运行次数或统一置信区间。附录 C 的 10 次随机选样与 10 次优化选样属于数据选择稳定性分析，不能作为每个性能点重复 10 次的证据。

### 3.5 可借鉴与待补齐

可复用数据类别、真实输入、长度分层、统一 tokenization 和吞吐—用户速度曲线。NexEval 应额外保存实际 target tokenizer 后的 token 长度，避免将 `o200k_base` 的 32K 名义桶等同于模型的 32K 输入。

尚需补齐论文实验的逐点配置、权重 revision、采样 top-p/top-k、预热、缓存、重复次数与完整统计口径。不要把新代码提供的 DFlash、DSpark 支持写成论文原始基线集合。

## 4. DFlash

### 4.1 论文的三组实验

来源：[论文 v2 §5、附录 A](https://arxiv.org/html/2602.06036v2)。

| 实验 | 模型与数据 | 硬件、后端 | 关键设置 |
| --- | --- | --- | --- |
| 非 thinking 主表 | Qwen3-4B、8B；GSM8K、MATH-500、AIME25、HumanEval、MBPP、LiveCodeBench、MT-Bench | H200，Transformers | `T=0/1`；最大新 token 2048；thinking disabled |
| thinking | Qwen3-4B、8B；GPQA、MATH-500、AIME25 | H200，Transformers | `T=0/1`；使用含 reasoning trace 的 target 输出训练 draft；该表的生成上限未独立明确 |
| 服务吞吐 | Qwen3-4B、8B、Qwen3-Coder-30B-A3B-Instruct；数学和代码任务 | 单 B200，SGLang、FlashAttention-4 | 并发 1、4、8、16、32；开启 Spec-v2 scheduling overlap |
| Llama 训练数据消融 | LLaMA-3.1-8B-Instruct；GSM8K、HumanEval、MT-Bench | 单 B200，SGLang、FlashInfer | 比较 DFlash 与 EAGLE3；与 FA4 主服务表后端不同 |
| 长上下文适配 | Qwen3.5-27B；LongBench 的 hotpotqa、qasper、gov_report | 论文未逐点明确全部执行配置 | 输入 1K–32K；base 与 long-adapted draft 比较 AL |

附录 A.4 增加更多 Qwen3.5、Qwen3-Coder、GPT-OSS 模型的 SGLang 结果，固定并发 8、单 B200；另有 Qwen3.5-9B 的 vLLM 并发结果。不能把不同硬件或后端的表合并为一个统一加速排名。

### 4.2 Draft 与基线控制

论文一般使用 5 层 DFlash，Qwen3-Coder 为 8 层；block size=16，LLaMA 3.1 为 10；从 target 的 5 层提取特征。Qwen 的 EAGLE3 基线使用 AngelSlim checkpoint；Llama 基线使用 EAGLE3 团队发布权重。

非 thinking 主表同时给出 EAGLE3 tree size=16 和 60，draft steps=7、top-k=10。这里的 top-k 是树展开宽度，不能误写成采样 top-k。DFlash block size 与 EAGLE3 tree size 对齐的是候选预算的一种比较方式，不能据此推定二者每轮 FLOPs、显存或验证成本相同。

主训练数据约 80 万条，来自 Nemotron Post-Training Dataset V2 与 CodeAlpaca，并由 target 重新生成答案。附录 A.1 为 AdamW、6 epochs、学习率 `6e-4`、训练 warmup ratio=0.04，训练序列长度 3072，Coder 为 4096。多数消融用随机抽取的 10 万条。长上下文适配使用 LongAlign-10K 中 1.6K 样本、3 epochs。

论文未完整披露所有 benchmark 的实际采样题数、LiveCodeBench 时间切分、每题重复次数、全部 checkpoint revision、全量训练测试去重结果。训练 warmup ratio 不能当成推理计时的预热策略。

### 4.3 当前官方实现与论文的差异

截至核对日，仓库 README 已包含 DFlash 2、更新模型和三个后端。以下是固定 SHA 的当前行为，并不证明论文 v2 原始实验采用了同样实现。

来源：[benchmark.py](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/benchmark.py)、[model.py](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/model.py)、[cli.py](https://github.com/z-lab/dflash/blob/07ebd93db9f472af339b644bb70221ad8428328a/dflash/cli.py)。

| 字段 | Transformers benchmark | OpenAI-compatible benchmark |
| --- | --- | --- |
| 样本顺序 | `random.Random(42)` 打乱；按 `max_samples` 截取 | 同一选样器；需要时重复数据，达到 `num_prompts + concurrency` |
| 默认上限与采样 | 2048 new tokens，T=0、top-p=1、top-k=0 | 相同 CLI 默认值；num-prompts=1024、concurrency=1 |
| 精度与随机种子 | BF16、SDPA；`torch.manual_seed(0)` | 实际引擎由外部服务决定，客户端没有给服务设置 seed |
| 预热 | 第一个样本，AR 与 DFlash 各生成至多 64 token | 额外 concurrency 条 prompt，各生成至多 64 token |
| 多轮 | 逐轮执行，后续历史使用投机输出，AR 在同一当前前缀上比较 | 每个样本只使用第一个 user turn |
| 计时 | CUDA synchronize 后计时；prefill 与 decode 分开 | 非流式 `requests.post`；从线程池测量开始至所有请求结束 |
| 吞吐 | 先求各请求 TPOT 均值，再取倒数；加速比为两组均值比 | completion token 总数 / 整个测量窗口时间，含请求及服务等待 |
| 接受长度 | 每请求的轮次平均，再对请求平均 | 读取 `meta_info.spec_accept_length` 后对请求平均 |

Transformers `model.py` 的 decode 计时从首 token 产生后开始，但分母使用整个输出 token 数，包含首 token。输出极短时需要明确这个差异。`acceptance_lengths` 记录的是每轮 `produced`，包含额外推进的 token，并在 EOS 或长度截断处调整。它与不含 bonus 的草稿接受数不同。

OpenAI-compatible benchmark 使用 `return_meta_info: true` 并尝试读取 `meta_info.spec_accept_length`、`meta_info.spec_verify_ct`。这只表明客户端期待这种返回形状；实际 SGLang/NorthGate 版本是否支持并保留字段，需要兼容性验证。父任务核对的 SGLang SHA 使用 `return_spec_tokens_details`，返回位置为 `sglext.spec_tokens_details`，流式路径还可能在 `choices=[]` 的末尾扩展 chunk 中返回。这些是不同版本或接口的契约，不能统一改写或假定互相兼容。

当前官方脚本没有为每个性能点建立多次运行置信区间。默认种子、样本打乱和预热已明确，但部署的 prefix cache 状态、引擎 commit、硬件频率及后台负载仍需要实验 manifest 补齐。

### 4.4 对平台的用途

直接复用“研究后端、生产后端、thinking、长上下文”分开报告的设计。初期可在同一 SGLang SHA 上比较 AR 与一个已支持的 draft 方案，并固定模型与计算资源。DFlash block size、EAGLE3 steps/top-k/tree tokens、MTP steps 应保留为各自原生配置，不强制压成一个语义不明的 `draft_length`。

## 5. TraceSpec

### 5.1 动态执行与固定前缀

来源：[论文 §3、§4.1、附录 A–C](https://arxiv.org/html/2610.41177v1)。

| 项目 | 论文披露 |
| --- | --- |
| 动态任务 | BrowseComp-Plus 100 题，SWE-bench Verified 100 题；任务抽样 seed=980406 |
| 动态动作 | 最多 50 步，每 5 步反思；上下文上限 32,768 token |
| 回放 | TraceSpec-Traj 的 2,500 test states；保留相同历史和工具观察，各方案生成一次 assistant continuation |
| 回放长度 | 上下文上限 36,864 token |
| 每次输出 | 最多 4,096 new tokens |
| Target | Qwen3-4B、8B、14B；对应 tokenizer 和 non-thinking chat template |
| 温度 | 0、1；thinking disabled |
| 方法 | DFlash、EAGLE3、DSpark、LongSpark、DLite、TraceSpec，共 108 个 workload×method×target×temperature 配置 |
| 草稿 | DFlash、DSpark、TraceSpec 使用 7-token block；EAGLE3 对应已发布配置；DSpark 关闭 confidence early stopping |
| 吞吐比较 | Native PyTorch 的 DFlash、EAGLE3、DSpark、DLite、TraceSpec；LongSpark 用 SGLang，仅参与接受情况比较 |
| episode 时间比较 | 论文原有 DFlash、EAGLE3、DSpark 运行组内比较 |

浏览工具检索和阅读 benchmark 文档；软件工程工具读真实仓库，但 edit/test 返回模拟反馈。论文不能提供真实 patch 成功率的证据。

### 5.2 数据规模、来源与隔离

10,000 个训练状态来自 5,795 个源任务；2,500 个测试状态来自 1,380 个源任务。训练和测试各一半为浏览、一半为软件工程。

| 来源 | 训练状态 | 测试状态 |
| --- | ---: | ---: |
| DeepResearch-traj | 5,000 | 1,250 |
| Nebius SWE-agent | 3,806 | 972 |
| R2E-Gym SFT | 1,031 | 243 |
| R2E-Gym testing agent | 163 | 35 |

名义上下文层为 2K、4K、8K、16K、32K，分别占 18%、20%、23%、21%、18%；统计使用实际分词长度。覆盖 2 个 workload × 5 个长度层 × 4 个 phase，共 40 个组合。phase 为 plan、act、reflect、answer。

论文先按源任务 hash 做 80%/20% 划分，再展开 assistant turns，源任务交集为零。软件任务用移除通用 scaffold 后的问题文本摘要识别；浏览任务用带来源限定的 query ID。其报告还包括 token 3-gram 重叠检查、排除全部 500 个 SWE-bench Verified 实例、与 BrowseComp-Plus 问题文本对照。development 仅从训练部分预留，用于调参。

### 5.3 指标与统计

对调用 i，A_i 为包含 correction/bonus 的最终接受输出数，V_i 为验证轮数，P_i 为提议草稿数，N_i 为输出 token 数，t_i 为模型调用时间：

- `accepted_length = ΣA_i / ΣV_i`。
- `verification_yield = ΣA_i / Σ(P_i + V_i)`。
- `generation_throughput = ΣN_i / Σt_i`，包含 prefill 与 decode。
- `episode_latency` 为完整交互时间，包含工具执行。

第三个公式是累计模型调用服务时间口径；存在并发时，它不等于系统总 tokens / 墙钟窗口。平台应分别命名。

点对点 95% 区间采用 5,000 次 bootstrap：动态执行以原始 task 重采样；回放以 source-task cluster 重采样；保留方法配对并重算分子分母。附录给出 bootstrap seed=20261021。不能把同一源任务的多个决策状态当作完全独立样本，也不能将这些区间解释为经过多重比较校正的同时置信区间。

### 5.4 公开实现的范围与缺口

固定源码：[README](https://github.com/Robot2050/TraceSpec/blob/1acb4be52e028b2a1aed550b187c227dc8b87cbe/README.md)、[eval_traj.sh](https://github.com/Robot2050/TraceSpec/blob/1acb4be52e028b2a1aed550b187c227dc8b87cbe/eval_traj.sh)、[config.py](https://github.com/Robot2050/TraceSpec/blob/1acb4be52e028b2a1aed550b187c227dc8b87cbe/agentspec/config.py)。

已核对的公开 README 仍描述 5 个 speculative baseline、90 个配置，最优方法描述为 DSpark；launcher 也只枚举这 5 个基线。与论文新增 TraceSpec 自适应训练、6 方法/108 配置存在范围差异。非 vendor 代码树中未发现对应的独立自适应训练入口，因此本次不能确认论文训练收益已完整发布并可复现。

README 明确说明：

- 不评分 SWE-bench patch 或 BrowseComp 答案。
- native evaluator 一次处理一个请求，每次完整 prefill 历史。
- 没有持久跨调用 KV 复用，也没有 continuous batching。
- native job 每个使用一张 GPU；LongSpark replay 每 worker 使用 target、draft 各一张 GPU。

代码默认 `tree_draft_temperature=0.7`，与 target `temperature=0/1` 是两个字段；`confidence_threshold=0.0`；工具观察最长 6000 字符；命令超时 60 秒；搜索 top-k=5。参数选择需要进入平台 manifest。

论文披露训练时用两张 A800 80GB，其中一张运行 vLLM 0.18.0 提供 target hidden states。**这不是全部评测硬件与推理版本的声明。**本次未核实推理 GPU 型号、驱动、全部依赖版本、测量预热、重复运行数量和所有 sampling top-p/top-k。应将这些字段记为未确认。

### 5.5 对 NAR 的用途

NAR 的真实执行与工具环境可补足 TraceSpec 的模拟反馈限制。固定前缀应取真实模型调用时刻的有效请求，保持 tools、tool_choice、reasoning、chat template、context compaction 结果。后续实验按同一 source task 分层和配对。

如果回放只生成下一段文本，后续轮次继续使用录制历史，可以公平比较指定状态；如果使用当前输出调用工具并推进历史，就进入动态执行协议。两种协议的结果不能合并。

## 6. 补充研究

### 6.1 Test-time scaling benchmark

来源：[Scaling Up, Speeding Up，§3.1–3.3、§4.1](https://arxiv.org/html/2509.04474v1)。

- 120 题：AIME24 和 AIME25 各全量 30 题；MATH-500 取前 30 题；GPQA 取前 30 题。GPQA 为该文所述完整来源中的子集，不能写成 GPQA Diamond 全量。
- Target 为 DeepSeek-R1-Distill-Llama-8B 与 Qwen3-8B；FP16；batch size=1。
- multi-round thinking 为 2 轮；Best-of-N 为 4 条候选轨迹。
- 多轮只把上一轮最终答案作为后续提示的一部分，丢弃之前完整 reasoning chain；不是普通完整上下文续聊。
- 使用单张 RTX A6000 48GB，CPU 为 16 核 Xeon Silver 4309Y。
- 评测 AR、SpS、EAGLE3、PLD、REST、Lookahead、PIA、SAM、Recycling 等及混合方案，按各实现默认参数运行；多轮表含 T=0、0.6。
- 主要指标为 Mean Accepted Tokens 与 walltime speedup。部分方法不支持相同采样设置，表中缺项不能按零处理。

本次未核对到所有实验的输出上限、BoN 排序器耗时是否计入、各候选的并发安排、跨候选 N-gram 状态生命周期、源码 SHA、统一预热和重复数。平台借鉴该工作时，应显式保存候选数、候选间状态共享与最终选择器成本。建议同时报告固定时间预算下的任务得分；这是平台建议，不是对论文已完成实验的陈述。

### 6.2 完整 CoT 接受率诊断

来源：[长 CoT 诊断论文 §3–5、附录 A–B](https://arxiv.org/html/2610.30644v1)。

- MATH-500 全部 500 题，seed=42，每题最多 16,384 new tokens。
- Target：DeepSeek-R1-Distill-Qwen-7B；draft：同系列 1.5B，以及跨系列 Qwen2.5-1.5B-Instruct。draft 未用该 target 输出训练。
- 草稿长度 γ=5；greedy 与随机采样分别测。随机采样 T=0.6、top-p=0.95，target/draft 使用同样截断规则。
- 四组总验证轮数依次为 394,572、382,556、499,428、502,976，约 178 万轮。
- 每轮记录生成位置、5 个草稿 token、6 个 target 预测、接受数量；随机采样另记相应概率。
- 区分 argmax agreement 与采样接受概率，二者不能互换。
- 同一道题按轮次位置排序，分成前后两半做配对；双侧符号检验、按题 bootstrap 10,000 次得到 95% CI。
- 将输出 ≥16,000 token 的样本按长度规则标记为 loop，在主配对分析中排除，并做阈值敏感性分析。这个标签没有使用正确率，但“长度触顶”也不应在平台上自动等同真正重复循环。
- 论文硬件池包含 RTX 4090 与 RTX 3090 Ti；所有吞吐比较限定同一物理 GPU，曾混用设备的吞吐增益明确撤回。

其“未观察到位置衰减”结论仅覆盖这些独立草稿、MATH-500 及有足够样本支持的区间：greedy 0–8K、sampling 0–12K。不能推及所有经过训练的 draft 或超长上下文。

平台建议保留全体样本和排除规则后的诊断两份报告；任务成功率和服务成本的主报告仍必须包括触顶、超时和失败样本，避免选择性剔除。

### 6.3 能耗与延迟目标

来源：[Only Accepted Tokens Count，§3、§6、§10](https://arxiv.org/html/2610.40287v1)。

论文比较 11 个模型、4 类 attention 结构，草稿涵盖 N-gram、MTP、EAGLE3、DFlash、DSpark。固定 batch 实验中，B 条请求同时提交，每条生成 512 token，无后续到达。每任务从 96 个不同 prompt 的池中取样，batch 扫描最高 64。

主要在单 H100 SXM5 94GB、功率上限 700W 上测；部分大模型改用 H200 141GB。BF16 权重和 KV cache、greedy；每个实验点至少 3 个 seed；每个模型固定一个 SGLang build，版本为 0.5.12 或更新，CUDA 13.2。

能量使用 `nvmlDeviceGetTotalEnergyConsumption` 累计计数器，10Hz 读取，以整次 generation 的差值计算。只覆盖 GPU 板卡，不包括 CPU、主机内存、网络和制冷。不能用 draft/verify 时间占比直接分摊能耗，论文使用移除 draft 的对照估计。

基线必须同模型、同 batch、同 prompt 池、同输出长度、同 GPU、同引擎配置。CUDA graph 应对 AR 与 SD 公平开启；eager baseline 会改变结论。各 drafter 的原生 block 大小不同，比较包含验证工作量差异。

对平台而言，能耗可以后续增加；首期优先保证同设备、稳定功率条件、GPU 秒和有效输出 token 的精确记录。

## 7. 统一指标契约建议

本节是平台建议。先保存原始计数，再计算指标，避免用一个名为 `acceptance_rate` 的浮点数承载不同含义。

| 建议字段 | 定义或用途 |
| --- | --- |
| `draft_tokens_proposed` | 实际提出的候选 token；标注 chain/tree 及计数范围 |
| `draft_tokens_accepted` | 通过验证的草稿 token，不含 correction/bonus |
| `verification_rounds` | 实际 target verification 轮数 |
| `committed_tokens_from_verification` | 验证循环最终提交的 token，含 correction/bonus；EOS、截断时按实际提交计 |
| `output_tokens_total` | 整次请求输出，包括首 token；另存 reasoning/tool/text 分类是否可得 |
| `accept_length_histogram` | 每轮最终推进长度的直方图，注明 bonus 和终止轮约定 |
| `position_acceptance_counts` | 每个 draft 位置的可接受前缀数与接受数，用于条件接受率 |
| `input_tokens_effective` | 真正送入 target 的 token 数；另存请求体及输入 token 摘要 |
| `engine_queue_ms`、`prefill_ms`、`decode_ms` | 引擎内部计时；缺失应为 null，不能填零 |
| `client_ttft_ms`、`client_e2e_ms` | 客户端观测；首 token 应排除 role-only、空 delta、心跳等事件 |
| `physical_attempt_id`、`logical_call_id` | 区分重试、路由与逻辑调用；幂等 replay 另行标记 |

建议同时提供两个 AL：`AL_micro = Σcommitted / Σrounds`，以及每请求 AL 的宏平均 `AL_macro`。报告必须标明主指标。Tree yield、TraceSpec verification yield、普通 draft acceptance fraction 分列。

吞吐至少区分：

- 系统吞吐：完成请求的有效 output tokens / 完整测量墙钟窗口。
- 模型服务时间吞吐：output tokens / 累计 model-call duration。
- 满足延迟目标的有效吞吐：满足预先声明的 TTFT/TPOT 或请求时限的成功请求所产 token / 墙钟窗口；同时报请求数版本和违约率。

TTFT、TPOT 与 SSE 间隔分开记录。投机解码会成块提交 token；客户端 chunk 可能再次合并。HTTP chunk 间隔只能描述交付节奏，不能证明单 token 的引擎生成时刻。

统计以 source task 或完整会话作为 cluster；配对 bootstrap 保留全部相关调用。在正式实验前确定样本数、重复次数和失败规则。建议性能点至少 3 次独立运行作为初始方案，若波动大则增加；该数字为平台建议，不代表前三篇论文均采用此设置。

## 8. NAR、NexEval、NorthGate、SGLang 的映射

以下按本次父任务提供的仓库核对结果整理，均为设计建议，未部署验证。

| 模块 | 可复用基础 | 建议新增或核对 |
| --- | --- | --- |
| NexEval | benchmark revision、evaluation plan、任务结果与 provenance | 性能实验 manifest、固定前缀数据集、负载 runner、分层和配对汇总；准确率与性能分开保存 |
| NAR | 真实 Agent 执行、模型请求 capture、逻辑调用与运行身份、工具环境 | 导出有效请求前缀；保存物理重试与引擎记录的关联；模型时间与工具时间并列 |
| NorthGate | 路由、request/session ID、数据采集 | 固定实验方案路由；记录实际 backend 与 retry；核对 SGLang 扩展字段和流式行为是否保留 |
| SGLang | SD 推理、真实调度、原始 spec counters | 固定 image/commit；输出 token/verification 计数；采集引擎实际 batch、queue、cache 和资源状态 |

父任务核对的 SGLang SHA 为 `3d33971dea1762fe2cc6dc8ec8172eecf3d6a3a4`。该版本的 `spec_decoding_metrics.py`、`tokenizer_manager.py` 和 OpenAI protocol/serving 路径已有 `spec_verify_ct`、`spec_num_correct_drafts` 等统计，及 `return_spec_tokens_details`/`SglExt` 返回机制。引擎已有能力与跨 NorthGate/NAR 是否保留要分别验收，避免重复发明一套与引擎不一致的统计。

NAR 当前文档明确内部身份头不会完整上送 NorthGate，主要提供 `X-Session-ID`；完整跨网关 provenance envelope 尚未实现。MVP 可以先建立逻辑调用到每次物理 attempt 的映射，再使用已存在的合法 `X-Request-ID` 或 `X-NorthGate-Request-ID` 与引擎记录关联。需要实测链路传播，不能只按近似时间或相同 session 猜测调用归属。

NexEval 现有 `integration-tests/replay` 返回录制的 assistant 响应，用于运行时回归。投机解码所需的固定前缀协议必须调用真实 target/draft；应另建清晰命名的数据与运行接口。

### 8.1 最小实验矩阵

以下是建议的首轮矩阵，尚未执行：

| 维度 | 初始选择 |
| --- | --- |
| 模型 | 一个有稳定 SGLang SD 支持、显存足够的 target；固定权重 revision、tokenizer 与 template |
| 方案 | AR 与一个 SD 方法；先保持相同 GPU 数和同一引擎 SHA |
| 固定前缀 | 实际 NAR capture 按 task 隔离；覆盖短/长上下文、工具调用、工具结果后的继续生成、最终答案 |
| 服务负载 | 先并发 1、4、16、32；只有容量足够时增加；分别记录客户端并发和实际引擎 batch |
| cache | 首轮固定无跨请求复用或已清空状态；再独立测同会话真实 prefix reuse |
| 解码 | greedy 正确性检查；再用目标模型适用的实际采样参数，明确 thinking、top-p、top-k、penalty、stop |
| 真实任务 | 从已有 NexEval benchmark 中选择可由原生 verifier 评分的任务；每方案独立环境和一致初始状态 |
| 统计 | 保留全部失败、超时、截断、重试；按 task 配对和 cluster bootstrap |

后续再增加固定 QPS 或真实到达时间戳回放，测突发负载与 SLO。当前控制并发的闭环 runner 不能替代开放到达过程：服务变慢时闭环会自动减少实际发起速率，隐藏排队风险。

### 8.2 必须冻结的实验信息

每个 experiment revision 至少冻结：数据集及源任务摘要、样本选择 seed、有效请求正文或 token 摘要、target/draft/tokenizer/template revision、SGLang image digest 与完整启动参数、GPU 型号与数量、TP/EP、精度、CUDA graph、attention backend、spec 配置、cache 策略、预热规则、生成参数、负载模型、失败与重试规则、计时范围和统计版本。

路径有三个层级：直连 SGLang、经 NorthGate、经 NAR Model Gateway。逐级运行相同固定前缀，才能辨认网关开销与推理收益。真实 Agent 结果单列，不能从 decode 加速倍数推算任务总耗时。

## 9. 复现入口与未完成核对

| 项目 | 已确认入口 | 仍需取得或验证 |
| --- | --- | --- |
| SPEED-Bench | 官方 HF 数据、Model-Optimizer `examples/specdec_bench/prepare_data.py` 和 `run.py` | 全量源数据补全、论文逐点 config、权重 revision、缓存及预热、实际模型 token 桶 |
| DFlash | 固定仓库的 `dflash benchmark` 三后端 | 论文 v2 对应源码快照；主表全部样本与 sampling、SGLang commit、原始日志 |
| TraceSpec 基线 | `run_agent_eval.py`、`eval.sh`、`eval_traj.sh`；公开 2,500 test prefixes | 真实推理 GPU 与版本、全部 checkpoint revision、训练方法发布差异 |
| TraceSpec 自适应训练 | 论文给出训练设置 | 当前固定公开仓库未确认完整训练入口与产物 |
| Test-time scaling | 论文协议与表格 | 完整 runner、选择器计时、候选调度、cache 状态和生成上限 |
| CoT 诊断 | 论文逐轮记录与统计定义 | 原始数据文件、可执行采集脚本及完整 runtime 配置 |
| 能耗 | 论文 NVML 累计能量协议 | 模型逐点 build/配置、功率预热和采样对齐；CPU 能量另计 |

本文没有运行上述脚本，所有命令仅作为已确认入口。正式实验应在隔离部署中先核对输入、字段保留与计数不变量，再运行性能矩阵。
