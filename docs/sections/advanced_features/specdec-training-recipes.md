# DFlash、DSpark 与 MoE 训练配方参考

检索日期：2026-11-06。本文汇总公开训练配置、论文、模型卡和实验记录，重点覆盖 DeepSeek V4 Flash。文档类型为参考文档；外部实验结果均按来源报告，NexSpecTrainer 未在本次文档整理中复现这些训练。

## 1. 范围与证据口径

目标模型为 MoE 与草稿模型为 MoE 是两个独立属性。为 MoE 目标模型训练的 DFlash、DSpark 草稿可以使用 dense FFN。本文分别记录这两个属性。

参数采用以下证据标记。表格未逐项标记时，由表格前的说明指定来源。

| 标记 | 含义 |
| --- | --- |
| 运行记录 | 作者保存的最终命令、训练记录或评测结果 |
| 配置 | 固定 commit 中提交的 YAML、JSON 或 Python 配置 |
| 默认 | 该版本代码的默认值，运行命令未显式覆盖 |
| 推导 | 根据配置与实现计算，尚无独立运行验证 |
| 未披露 | 在本次检查的来源中未找到；不以其他项目默认值补齐 |

PR 状态均为检索当日状态。未合并 PR 的配置不代表主分支已支持；已发布权重也不代表完整训练配方已公开。`online` 在这些训练框架中通常表示训练时采集目标隐藏状态，与在当前草稿生成前缀上做 on-policy distillation 的含义不同。

| 来源 | Target | 草稿结构 | 公开内容与状态 |
| --- | --- | --- | --- |
| SpecForge 主分支 [S1] | V4-Flash-0731 | 5 层 dense GQA DSpark | 配置、运行文档、B200 与 MI355X 训练记录 |
| SpecForge PR #940 [S2] | V4-Flash-0731 | 5 层 GQA，64 专家 MoE DSpark | 配置与 serving 对照结果；未合并 |
| SpecForge PR #1057 [S3] | V4-Flash-0731 | 3 层 V4 attention、mHC、256 专家 DSpark | 配置与实现；未合并，WIP |
| Sawyer117/speculators [S4] | V4-Flash Preview | 原生 3 层 MoE DSpark | 完整训练链路、5 epochs、同栈评测；完整链路位于 fork |
| RedHatAI/Speculators [S5] | V4-Flash | 5 层 dense、全 SWA DFlash | 已发布权重；部分训练参数公开 |
| DeepSpec [S6] | Qwen3-4B/8B/14B、Gemma4-12B | dense DFlash、DSpark | 论文对照配置与对应权重 |
| TorchSpec [S7] | Qwen3.6-35B-A3B、Kimi-K3 等 MoE | dense DFlash、MLA DSpark | 训练配置；部分路径为占位符或缺失 |
| AngelSpec [S8] | Qwen3-8B 等 | DFlash、DSpark，另有 MoE MTP | 示例配置；本次未找到 V4 原生 DSpark 配方 |

## 2. DeepSeek V4 Flash 原生 MoE DSpark 复现

### 2.1 来源与适用版本

[S4] 的流水线、训练脚本和技术报告保存了完整复现。上游 RFC #952 明确指定 `deepseek-ai/DeepSeek-V4-Flash` Preview。`DeepSeek-V4-Flash-0731` 属于另一目标版本，本文不将两者的训练结果合并。

上游 PR #1041 只包含模型等部分实现，完整 EP 训练链路仍在 fork 中。技术报告及部分标题使用“从零训练”，最终命令实际开启目标层 warm start；下表按有效命令记录初始化方式。

### 2.2 数据与优化参数

表中除明确标记外，均来自最终运行命令与收官报告。

| 参数 | 值与说明 |
| --- | --- |
| Prompt 数据 | Open-PerfectBlend |
| 回答生成 | 目标模型自蒸馏，`temperature=0`，greedy |
| 样本量 | 775,965 条去重记录；文档中存在其他历史数据目录与数量，复现入口指定 `arrow_0730_77w_dedup` |
| 最大生成长度 | 3072 tokens |
| 多轮比例 | 作者报告 99.96% 为单轮 |
| `total_seq_len` | 3072，collator 将文档 packing 到该 token budget |
| Epochs | 5 |
| Optimizer steps | 124,480，技术报告值 |
| Optimizer | AdamW |
| Learning rate | `2e-4` |
| Weight decay | `0.01`，CLI 默认 |
| Adam β、ε | 未覆盖 PyTorch 默认：`(0.9, 0.999)`、`1e-8` |
| Scheduler | Cosine，最终 LR 为 0 |
| Warmup ratio | `0.04` |
| Gradient clipping | 全局 norm `1.0`，训练循环固定值 |
| 训练设备 | 8×Atlas A2，EP8 与 FSDP2 |
| 专家分布 | 每层 256 个 routed experts，每 rank 持有 32 个完整专家 |
| 每步 token budget | 推导值约 `8 × 3072 = 24576` packed input tokens；各 rank 使用不同输入，每个 batch 更新一次，无额外梯度累积。原始对话条数随 packing 变化 |
| Expert dtype | BF16；64GB A2 命令显式设置 `BF16_EXPERTS=1` |
| Activation recompute | `RECOMPUTE=1` |
| Compile | `COMPILE=0` |
| Seed | `42`，CLI 默认 |
| Checkpoint frequency | 每 `0.5` epoch |
| 训练内 validation | `--no-validation`；训练后单独执行 serving 评测 |
| Dataloader | 最终共享文件系统命令使用 workers=12、prefetch factor=4 |
| 特征供给 | 在线 target forward，按数据行号产生隐藏状态文件，训练端按相同行号读取 |

`24576` 是输入 token budget，不能当作参与 loss 的 token 数。Anchors、loss mask、packing 和有效 block 数共同决定监督位置数。

### 2.3 草稿架构与训练目标

架构字段来自 fork 配置，方法参数以最终命令覆盖值为准。

| 参数 | 值 |
| --- | --- |
| Draft layers | 3 |
| Hidden size | 4096 |
| Vocabulary | 129280，完整词表 |
| Attention | V4 低秩投影，per-head attention sink |
| Attention heads / head dim | 64 / 512 |
| RoPE head dim | 64 |
| Q LoRA rank / O LoRA rank | 1024 / 1024 |
| O groups | 8 |
| Target context window | 128 |
| Block 内注意力 | Non-causal |
| Target feature layer IDs | `[40, 41, 42]`，层编号语义需与采集实现一同固定 |
| Block size | 5，全部 5 个位置参与预测，`sample_from_anchor=true` |
| Max anchors | 512，最终命令覆盖 launcher 的 64 默认值 |
| Markov head | `vanilla`，rank=256 |
| Confidence head | 开启，使用 Markov 信息，loss 权重 1.0 |
| Routed / shared experts | 每层 256 / 1 |
| Top-k / expert intermediate | 6 / 2048 |
| Router score | `sqrtsoftplus` |
| Routed scaling factor | 1.5 |
| SwiGLU limit | 10.0 |
| Load balancing | `noaux_tc` selection bias，update rate=`1e-3` |
| mHC streams | 4 |
| Sinkhorn iterations / epsilon | 20 / `1e-6` |
| Mask token ID | 128799 |
| Position decay | γ=4，`w[k] = exp(-k / 4)` |
| 主损失 | `0.1 CE + 1.8 TV` |
| Confidence loss | 权重 1.0，单独加入 |
| KD temperature | 1.0 |
| Hidden-state augmentation | Uniform noise `[-0.05, +0.05]` |
| Teacher double norm | 关闭 |

损失归一化需要随配方记录。该 Speculators 实现的 `TV = 0.5 × Σ|p-q|`，DeepSpec 的 `L1 = Σ|p-q|`，因此 `1.8 TV = 0.9 L1`。系数相同并不保证两个框架使用同一损失强度。

最终初始化命令为 `INIT_LAYER=1 INIT_MOE_NO_ROUTER=1`：attention、MoE experts、mHC 和 norm 从目标 `[40,41,42]` 层初始化，router 重新随机初始化。该实验没有使用官方已训练草稿作为整体起点，也没有将全部参数随机初始化。

作者记录 warm-start 实验使用 `6e-4` 时在 LR 接近峰值处出现 NaN，最终采用 `2e-4`。该观察限于其数据、实现和初始化方式，不构成所有 MoE 草稿的 LR 上限。

### 2.4 有效启动参数

以下是 [S4] 最终启动命令的参数摘录。路径为占位符；完整安装、采集、转换和评测流程见来源。该脚本属于外部 fork，不能作为 NexSpecTrainer CLI 直接运行。

```bash
DSPARK_EP=1 BF16_EXPERTS=1 RECOMPUTE=1 COMPILE=0 \
DSPARK_MOE_BALANCE=1 DSPARK_MOE_BALANCE_RATE=1e-3 \
DSPARK_LOG_EXPERT_LOAD=1 \
INIT_LAYER=1 INIT_MOE_NO_ROUTER=1 \
LR=2e-4 EPOCHS=5 MAX_ANCHORS=512 CKPT_FREQ=0.5 \
DATA=/path/to/arrow_0730_77w_dedup \
  bash examples/ascend_npu_dflash/train_dsv4_dspark.sh faithful
```

复现时还需要覆盖该脚本的 `VERIFIER`、`HS_DIR`、`ENDPOINT`、`RUN` 和 `CANN_ENV`。脚本包含清理旧隐藏状态文件的操作，应为本次运行设置独立目录。

该 fork 的旧配置注释仍出现 `block_size=6`。最终 launcher 明确传入 `5` 并训练 slot 0；文档中的历史命令应按记录日期和修正状态读取。

### 2.5 评测结果与限制

以下统一取 [S4] 2026-08-11 收官报告。作者在同一 serving 栈上评估自训草稿与官方草稿，使用 greedy、`num_speculative_tokens=5`。后续 PR 更新的数字略有变化，本文不混合不同评测运行。

| 数据集 | 自训 5 epochs 平均接受长度 | 官方草稿平均接受长度 |
| --- | ---: | ---: |
| GSM8K | 4.849 | 4.658 |
| MATH500 | 4.565 | 4.661 |
| HumanEval | 4.933 | 4.942 |
| MBPP | 4.555 | 4.535 |
| MT-Bench | 3.079 | 3.294 |
| 五项均值 | 4.396 | 4.418 |

作者还报告并发 48 时相对自回归的端到端加速比为 1.39×。这些结果支持该训练链路能够产出接近所测官方草稿接受长度的模型。报告未建立与自回归逐 token 完全一致的证明；不同 serving 栈之间也不能直接比较接受长度或吞吐。

MT-Bench 的剩余差距与数据单轮比例同时存在。作者将差距归因于数据覆盖，本文未找到控制该变量的对照实验，原因未查明。

## 3. SpecForge V4-Flash-0731 dense DSpark

### 3.1 训练配置

[S1] 主分支 recipe 从随机初始化训练五层 dense GQA 草稿。目标模型为 MoE。参数来自 YAML、JSON；默认项另外标出。

| 参数 | 值 |
| --- | --- |
| Target | `deepseek-ai/DeepSeek-V4-Flash-0731` |
| 数据 / max length | ShareGPT / 8192；该 recipe 未要求先重生成回答 |
| Epochs | 2 |
| Microbatch / trainer DP / accumulation | 1 / 4 / 32 |
| Global batch | 推导值 128 sequences/update |
| Optimizer | AdamW，FP32 master weights |
| Weight decay | 0，优化器实现默认 |
| Adam β / ε | PyTorch 默认 `(0.9, 0.999)` / `1e-8` |
| LR / scheduler / warmup | `6e-4` / constant / 0 |
| Clip norm / dtype | 1.0 / BF16 |
| FSDP | `SHARD_GRAD_OP`，schema 默认 |
| Block / anchors / decay γ | 7 / 512 / 4 |
| Loss | `0.1 CE + 0.9 L1 + 1.0 confidence` |
| Objective chunk | 128 anchor blocks |
| Objective checkpointing | 开启，schema 默认 |
| Attention backend | `flex_attention` |
| Seed / prompt seed | 42 / 1 |
| Save / log interval | 128 / 10 optimizer steps |

| 架构参数 | 值 |
| --- | --- |
| Layers / hidden / FFN intermediate | 5 / 4096 / 12288 |
| Attention | GQA，五层 full attention |
| Q heads / KV heads / head dim | 32 / 8 / 128 |
| Target taps | `[1,11,21,31,41]` |
| Markov head | vanilla，rank=256 |
| Confidence head | 开启，使用 Markov 信息 |
| Vocabulary / mask ID | 129280 / 128799 |
| RoPE | YaRN，θ=10000，factor=16，original length=65536 |
| 最大位置配置 / 实际训练长度 | 1048576 / 8192 |
| Attention dropout / RMSNorm ε | 0 / `1e-6` |

### 3.2 采集与运行参数

| 参数 | 值 |
| --- | --- |
| B200 拓扑 | 单机 8 卡；两个 TP2 target server 占 4 卡，trainer 占 4 卡 |
| SGLang | v0.5.18 加 `spec-capture` patch |
| Capture method | `dspark`，使用 V4 专用 hook |
| Context / max requests | 8704 / 每 server 8 |
| Prefill | `chunked-prefill-size=-1`，`max-prefill-tokens=8704` |
| MoE backend | `flashinfer_mxfp4` |
| Static memory fraction | 手工 server 命令 0.88；YAML 字段为 0.85。external server 以实际启动命令为准 |
| Mooncake | TCP，global segment 200 GiB，local buffer 1 GiB |
| Lease / transfer timeout | 300 秒 / 300 秒 |
| Producer concurrency | 8 |
| In-flight high / low watermark | 256 / 128 samples |
| Resident high / low / cap | 128 / 96 / 160 GiB，YAML 值 |
| 数据预处理 / dataloader workers | 64 / 8 |

V4 的 inter-layer residual 为 mHC 多流张量。该 recipe 使用模型自己的 `set_dspark_layers_to_capture` hook；普通 DFlash capture 的输出不能自动视为等价输入。

[S1] 报告 B200 约 4.6 秒/update。MI355X 两轮训练共 1885 optimizer steps，平均 6.32 秒/update，总计约 3 小时 18 分。MI355X 命令改用 `attention-backend=dsv4`、page size=256、SWA/full token ratio=0.15、关闭 radix cache，并设置 `AITER_BF16_FP8_MOE_BOUND=0`。以上时间只对应 dense recipe。

## 4. SpecForge MoE 草稿配置

### 4.1 三种结构

[S2] 的 64 专家方案保留 GQA 与常规 residual，仅替换 FFN。[S3] 使用 V4 原生 attention、mHC 与 256 专家。Qwen 方案取自 [S3] 依赖栈内的配置；PR 后续演进可能改变字段。

| 参数 | V4 GQA + MoE | V4 原生 MoE | Qwen3-8B MoE DSpark |
| --- | --- | --- | --- |
| 状态 | PR #940 未合并 | PR #1057 未合并，WIP | 未合并配置 |
| Draft layers / hidden | 5 / 4096 | 3 / 4096 | 5 / 4096 |
| Attention | GQA 32Q、8KV、dim128 | V4 64 heads、dim512、sink | GQA 32Q、8KV、dim128 |
| Residual | 常规 | mHC，4 streams | 常规 |
| Routed / shared experts | 64 / 1 | 256 / 1 | 8 / 0 |
| Top-k / expert intermediate | 6 / 2048 | 6 / 2048 | 1 / 1536 |
| Block | 7 | 5 | 16 |
| Target taps | `[1,11,21,31,41]` | `[40,41,42]` | `[1,9,17,25,33]` |
| Markov rank | 256 | 256 | 256 |
| Confidence loss weight | 1.0 | 1.0 | 0，head 关闭 |
| Router bias update rate | `1e-3` | `1e-3` | `1e-3` |
| Router auxiliary loss | aux-loss-free 路径 | 0 | 0.005，batch-wise |
| Router preset | `deepseek_v4` | `deepseek_v4` | `qwen3_5_moe` |

V4 preset 使用 `sqrtsoftplus`、归一化路由权重、scale=1.5、SwiGLU limit=10。原生方案另外设置 Q/O LoRA rank=1024、O groups=8、context window=128、Sinkhorn iterations=20、epsilon=`1e-6`、confidence input=`pre_norm` 且不使用 confidence bias。

### 4.2 训练参数与结果

| 参数 | V4 GQA + MoE | V4 原生 MoE | Qwen3-8B MoE DSpark |
| --- | --- | --- | --- |
| 数据 | ShareGPT | ShareGPT | PerfectBlend，target 重生成，T=0、non-thinking |
| Max length | 8192 | 8192 | 8192 |
| Epochs | 2 | 2 | 6 |
| Microbatch / DP / accumulation | 1 / 4 / 32 | 1 / 4 / 32 | 2 / 6 / 11 |
| Global batch，推导值 | 128 | 128 | 132 |
| LR | `6e-4` | `6e-4` | `6e-4` |
| Scheduler / warmup | constant / 0 | constant / 0 | cosine / 0.04 |
| Anchors / decay γ | 512 / 4 | 512 / 4 | 512 / 7 |
| CE / L1 weight | 0.1 / 0.9 | 0.1 / 0.9 | 0.1 / 0.9 |
| Clip norm | 1.0 | 1.0 | 1.0 |
| FSDP | 默认 SHARD_GRAD_OP | 默认 SHARD_GRAD_OP | NO_SHARD |

V4 原生 PR 的 YAML 沿用 dense recipe 的四 rank 布局。本文未找到该配置在相同硬件上的完整训练与质量验收记录，不能沿用第 3 节的内存和耗时结论。

[S2] 的 serving patch 为 MoE 导出增加专用加载类；普通 dense loader 曾跳过专家参数，导致接受长度约 1.006。修复后的 step 1024 对照结果如下，均为作者报告：

| 指标 | 64 专家 MoE | Dense |
| --- | ---: | ---: |
| GSM8K，单并发接受长度 | 1.30 | 1.38 |
| MT-Bench，单并发接受长度 | 1.34 | 1.40 |
| 单并发 tok/s | 94 | 117 |

该实验建立了训练、导出和 serving 的对照路径，结果尚未显示 MoE 优于 dense。旧 PR #812 已关闭且未合并，后续工作拆分至新的 PR 栈。

## 5. V4 Flash DFlash 已发布模型

[S5] 为 `RedHatAI/DeepSeek-V4-Flash-speculator.dflash` 模型卡与配置。除未披露项外，下表均为发布记录或模型 JSON 值。

| 参数 | 值 |
| --- | --- |
| Target / 框架 | `deepseek-ai/DeepSeek-V4-Flash` / Speculators |
| 数据 | Magpie-Llama-3.1-Pro-300K-Filtered 子集与 UltraChat-200k `train_sft` |
| 数据量 | 模型卡称约 500K samples |
| 特征 | 在线 vLLM，target 使用 DP 与 EP |
| Draft | 5 层 llama-style dense transformer |
| Hidden / FFN intermediate | 4096 / 2048 |
| Q heads / KV heads / head dim | 64 / 1 / 256 |
| Attention | 全部层 SWA，window=2048 |
| `sliding_window_non_causal` | false |
| Target hidden width | `hc_mult=4`，每个 tap 为 `4 × 4096 = 16384` |
| Target taps | `[3,13,23,32,42]` |
| Draft / target vocabulary | 32000 / 129280 |
| Block / speculative tokens | 8 / 7 |
| Max anchors / sequence length | 3072 / 8192 |
| Optimizer | Muon |
| `muon_lr` / `lr` | 0.02 / `6e-4` |
| Precision / mask ID | BF16 / 1 |
| RoPE θ / RMSNorm ε | 10000 / `1e-6` |
| Attention dropout | 0 |
| Epochs、batch、accumulation | 未披露 |
| Scheduler、warmup、weight decay | 未披露 |
| Muon momentum、NS steps、完整 loss weighting | 未披露 |
| 数据重生成采样设置 | 未披露 |

模型卡报告平均接受长度约 3.9 tokens。其验证口径与第 2 节不同，不能直接进行模型优劣排序。另有 [50K all-SWA/Muon checkpoint](https://huggingface.co/inference-optimization/dflash-DeepSeek-V4-Flash-all-swa-muon-speculators-50k)，应与该约 500K 模型分别记录。

## 6. 其他 MoE 目标模型

### 6.1 SpecForge dense 草稿

以下来自 [S1] 同一主分支快照中的对应 YAML 与 JSON。Global batch 为配置推导值；标记“默认”的 scheduler 和 loss 来自该版本 schema。

| 参数 | Qwen3.5-35B-A3B DFlash | LongCat-Flash DFlash | Kimi-K3 DSpark | GLM-5.2 DSpark |
| --- | --- | --- | --- | --- |
| 数据 | `opc_train_regen_first_turn` | ShareGPT | 本地 K3 数据 | 本地 `glm52_dspark_train` |
| Max length | 4096 | 3072 | 65536 | 4096 |
| Epochs / max steps | 10 / 10000 | 6 / 10000 | 10 / 未设 | 10 / 未设 |
| Microbatch / DP / accumulation | 2 / 4 / 1 | 2 / 1 / 1 | 1 / 4 / 32 | 1 / 1 / 512 |
| Global batch | 8 | 2 | 128 | 512 |
| LR | `6e-4` | `6e-4` | `6e-4` | `6e-4` |
| Scheduler | cosine，默认 | cosine，默认 | constant | cosine，默认 |
| Warmup | 0.04 | 0.04 | 0 | 0.04 |
| Anchors / decay γ | 512 / 7 | 512 / 7 | 512 / 4 | 512 / 4 |
| Block / draft layers | 16 / 8 | 16 / 5 | 7 / 5 | 7 / 5 |
| Hidden / FFN | 2048 / 6144 | 6144 / 12288 | 7168 / 14336 | 6144 / 12288 |
| Q heads / KV heads / dim | 32 / 4 / 128 | 32 / 8 / 128 | 64 / 16 / 64 | 64 / 16 / 64 |
| Target taps | `[1,10,19,28,37]` | `[1,7,13,19,25]` | `[7,23,51,67,83]` | `[1,19,38,57,76]` |
| Markov | 无 | 无 | rank256 | rank256 |
| 主损失 | DFlash CE | DFlash CE | 0.1 CE + 0.9 L1 + confidence | 同左，默认 |

这些表中的 FFN 均为 dense。Kimi runbook 描述 weights-only warm start 和重生成 agentic 数据；可移植 YAML 仍使用本地数据路径。采用该配方前需要另行提供与运行记录对应的初始化权重及数据。

来源：[四个训练 YAML 所在目录][SF-YAML]、[架构 JSON 所在目录][SF-JSON]。

### 6.2 TorchSpec

[S7] 中的两份配置提供了不同的优化和并行设置。

| 参数 | Qwen3.6-35B-A3B DFlash | Kimi-K3 MLA DSpark stage1 |
| --- | --- | --- |
| 数据 | 注释称重生成 Magic-Prompt；实际路径为 sample conversations | 用户指定路径，占位符 |
| Max length / epochs | 2048 / 3 | 32768 / 5 |
| Microbatch / DP / accumulation | 1 / 6 / 2 | 1 / 8 / 16 |
| Global batch，推导值 | 12 | 128 |
| LR / min LR | `6e-4` / `6e-5` | `5e-5` / 0 |
| Scheduler | YAML 未显式设置 | cosine |
| Warmup / weight decay | 0.04 / 0.01 | 0.04 / 0 |
| Clip norm | 1.0 | 1.0 |
| Block / anchors / decay γ | 16 / 512 / 7 | 7 / 512 / 4 |
| Loss | DFlash | 0.1 CE + 0.9 L1 + 1.0 confidence |
| FSDP | FULL_SHARD | FULL_SHARD |
| 训练 GPU 数 | 6 | 8，两个节点各 4 卡 |
| 采集 GPU 数 | 2，两个 TP1 engine | 32，两个 TP8×PP2 engine |
| Gradient checkpointing | 未显式设置 | true |
| 保存间隔 | 1000 steps，另按 epoch 保存 | 1000 steps，另按 epoch 保存 |

Kimi draft 的 MLA 参数为 hidden=7168、FFN=14336、layers=5、heads=64、q-rank=1536、kv-rank=512、qk-nope dim=128、qk-rope dim=64、v-head dim=128。Target taps 为 `[7,31,47,63,87]`；采集配置说明这些是 layer output 编号，engine 再转换为对应 capture ID。

Qwen3.6 YAML 引用 `torchspec/config/dflash_draft_config_qwen36_35b.json`，本次固定主分支快照中未找到该文件。Kimi 配置包含 `???` 路径。因此两者均需要补充材料才能直接运行。

[Z-Lab/Modal 的 Qwen3.6-35B-A3B DFlash 模型][S9]使用另一套已发布配置：40K 训练长度，6 层，前 5 层 SWA4096、最后 1 层 full attention，block16，hidden=2048，FFN=6144，32Q/8KV，head dim=128，8 个 target taps `[1,6,11,16,22,27,32,37]`。其模型卡未给出完整优化 schedule；不能将上表的 2K 示例参数归给该权重。

## 7. DeepSpec 的 DFlash 与 DSpark 对照

[S6] 配置对应论文对照实验和发布权重。以下参数由配置和优化器实现共同确定。

| 参数 | DFlash 对照 | DSpark |
| --- | --- | --- |
| Draft layers / block | 5 / 7 | 5 / 7 |
| Anchors / decay γ | 512 / 4 | 512 / 4 |
| CE / L1 weight | 1.0 / 0 | 0.1 / 0.9 |
| Markov rank | 0 | 256，vanilla |
| Confidence weight | 0 | 1.0 |
| Optimizer | AdamW，FP32 masters | 同左 |
| LR / weight decay | `6e-4` / 0 | 同左 |
| Adam β / ε | PyTorch 默认 `(0.9,0.999)` / `1e-8` | 同左 |
| Scheduler / warmup | cosine / 0.04 | 同左 |
| Clip norm / precision | 1 / BF16 | 同左 |
| Local / global batch | 1 / 512 | 同左 |
| 默认 8 卡时 accumulation | 推导值 64 | 同左 |
| Epochs / max length | 10 / 4096 | 同左 |
| Sharding | `no_shard` | 同左 |
| Seed / workers | 42 / 4 | 同左 |
| Log / checkpoint interval | 10 / 3000 steps | 同左 |

| Target | Feature layers | Mask ID | `torch_compile` |
| --- | --- | ---: | --- |
| Qwen3-4B | `[1,9,17,25,33]` | 151669 | true |
| Qwen3-8B | `[1,9,17,25,33]` | 151669 | true |
| Qwen3-14B | `[1,10,19,28,37]` | 151669 | true |
| Gemma4-12B | `[5,17,29,41,46]` | 4 | false |

论文使用约 130 万条原始 Open-PerfectBlend 样本，组成比例为 chat 17.6%、math 39.4%、code 38.9%、instruction following 4.1%。脚本切出 5% test，再由各目标模型重生成回答，使用 non-thinking 模式。Qwen3-4B 示例参数为 `temperature=0.7, top_p=0.8, top_k=20, min_p=0, max_tokens=4096`。其他目标应使用各自的生成设置。

该离线流程预计算 target cache。官方文档估计默认 Qwen3-4B 数据配置约需 38 TB，实际占用取决于数据 token 数、采集层数、hidden width 和保存精度。

## 8. DFlash 原论文与 Speculators 参数实验

### 8.1 原论文

[S10] 正文与附录 A.1 披露：

| 参数 | 值 |
| --- | --- |
| 数据 | 约 800K，Nemotron Post-Training Dataset V2 与 CodeAlpaca |
| 回答 | 目标模型重新生成 |
| Draft layers | 5；Qwen3-Coder 为 8 |
| Block size | 16；LLaMA3.1 为 10 |
| Features | 5 层，在第 2 层至倒数第 3 层之间均匀选取 |
| Epochs | 6 |
| Optimizer / LR | AdamW / `6e-4` |
| Scheduler / warmup | cosine / 0.04 |
| Clip norm | 1.0 |
| Max length | 3072；Qwen3-Coder 为 4096 |
| Anchors | 512 |
| Position decay | block16→γ7；block10→γ5；block8→γ4 |
| Global batch、Adam β/ε、weight decay | 本次检查的论文训练段落未披露 |

DeepSpec 的 block7 对照实验与原论文的 block16 实验应分别记录。不同实现对 anchor slot 是否计入预测数的定义也可能不同。

### 8.2 Speculators DFlash best-practices RFC

[S11] 在 Qwen3-8B 上报告一组参数实验。该配方属于独立实验，不能代替第 5 节 V4 权重未公开的字段。

| 参数 | RFC 最佳配置 |
| --- | --- |
| Draft | 5 层、block16、全部层 causal SWA2048 |
| Vocabulary | 完整 151936 |
| Target taps | `[2,10,18,26,34]` |
| Loss | CE 与 D-PACE，α=0.5 |
| Muon LR / AdamW LR | `3e-3` / `6e-4` |
| Muon momentum / weight decay | 0.95 / 0.1 |
| Muon NS steps / LR adjustment | 5 / `match_rms_adamw` |
| Max anchors | 1536 |
| Packed token budget | 65536 tokens/rank |
| 硬件 / 每步输入 token budget | 4×H100 80GB / 约 262144 tokens |
| Epochs / scheduler | 20 / cosine，连续完整 schedule |
| Checkpoint frequency | 每 0.1 epoch |
| 数据生成 teacher | Qwen3-235B-A22B-Instruct-2507；target 为 Qwen3-8B |

RFC 报告将专门补充的 code、多语言数据从 epoch 0 混入。其内部 9 类 benchmark 的最终平均接受长度为 3.737。这个数值与 V4 实验的目标、数据和 serving 设置不同。

作者还报告：增加 anchors 在单变量短实验中改善接受长度，但 2048 anchors 的完整训练未超过 1536 anchors 配方，原因未查明。Muon LR 应与 AdamW LR 分别记录；该 RFC 的 `3e-3` 与 RedHat V4 模型卡的 `0.02` 属于不同实验。

## 9. AngelSpec 示例配置

[S8] 的 Qwen3-8B 示例用于检查另一框架的参数范围。两者训练数据都指向 sample conversations，缺少完整生产数据配比。

| 参数 | DFlash | DSpark |
| --- | --- | --- |
| Microbatch / DP / accumulation | 1 / 4 / 2 | 1 / 4 / 2 |
| Global batch，推导值 | 8 | 8 |
| Max length / epochs | 2048 / 3 | 2048 / 3 |
| LR / min LR | `6e-4` / `6e-5` | 同左 |
| Weight decay | 0.01 | 0 |
| Warmup / clip | 0.04 / 1.0 | 同左 |
| Block / anchors / γ | 16 / 512 / 7 | 7 / 512 / 4 |
| Target feature count | 5 | 5 |
| DSpark loss | 不适用 | 0.1 CE + 0.9 L1 + 1.0 confidence |
| FSDP / reduce dtype | FULL_SHARD / BF16 | 同左 |
| 训练 GPU / 采集 GPU | 4 / 4 | 同左 |
| Capture | SGLang TP1，batch=8，static memory fraction=0.7 | 同左 |
| Feature storage | Mooncake TCP，global segment=16GB，local buffer=4GB，沿用配置单位 | 同左 |
| Save / max checkpoints | 1000 steps，加 epoch 保存 / 2 | 同左 |

## 10. 与 NexSpecTrainer V4.1 配置的关系

本仓库的 [V4.1 原生训练说明][N1]、[Muon 实现说明][N2]和[验证记录][N3]分别定义实现、优化器契约和本地验收范围。外部 V4 配方没有验证本仓库的 V4.1 参数。

| 项目 | 本文外部 V4 原生复现 | NexSpecTrainer V4.1 原生说明 |
| --- | --- | --- |
| Hidden width | 4096 | 5120 |
| Routed experts / top-k | 256 / 6 | 128 / 3 |
| Draft layers / mHC streams | 3 / 4 | 3 / 4 |
| 特征层 | `[40,41,42]`，需固定外部 hook 语义 | layer inputs `[37,38,39]`，四流均值后拼接 |
| 拼接特征宽度 | 由外部 capture 契约定义 | 15360 |
| 最终 target hidden | 需核对 norm 契约 | 宽度 5120，post-final-norm |
| 训练并行 | 外部 fork EP8 + FSDP2 | 本仓库原生说明的 FSDP adapter 不包含 EP/TP |
| 优化器参数来源 | 外部 AdamW 训练命令 | 本地 AdamW 或 Muon profile；部分超参为项目选择 |

[N2] 的 `v41_flash_muon` 固定 V4.1 主干论文的算法常量，并明确标出 DSpark 参数映射和训练 LR 的项目选择。不能将主干预训练优化器参数视为官方 DSpark 训练配方。

迁移配方时需要独立记录以下契约：

- Target 模型名称、revision、量化与计算精度；V4 Preview、0731 和 V4.1 分别建档。
- 草稿的架构、初始化来源、冻结参数、router 初始化与 balancing 设置。
- Capture 层编号、layer input/output、mHC 聚合方式、最终 norm 位置。
- Block width、预测位置数、anchor slot 标签位移和注意力可见范围。
- Loss 的 L1/TV 定义、有效 token 分母、position weighting、confidence 权重。
- Microbatch、DP、EP、梯度累积、packed input tokens 与有效监督位置数。
- Optimizer 分组、两组 LR、默认 β/ε、schedule、warmup 和总 updates。
- 同 target revision、同 serving 栈、同数据和并发下的接受长度与端到端吞吐。

## 11. 来源与固定版本

以下 GitHub 配置链接固定到本次读取的 commit。PR 和 RFC 链接用于说明状态与实验记录，其正文可能在检索日期之后更新。论文链接采用 arXiv 文章标识；再次复现时应额外记录所用论文版本。

| 来源 | 版本 |
| --- | --- |
| NexSpecTrainer 本地契约基准 | `764eb921e22e96ec64436c6f0807a97181df1a0d` |
| SpecForge 主分支 | `b520d2b4c08b94413db56392682092d938819c61` |
| SpecForge PR #940 | `9061bc866236bf714e11b392517eaf49e3cb6a43` |
| SpecForge PR #1057 | `9e12b96941c77134d58b5c814761b9aad79ec2f2` |
| Sawyer117/speculators，`feat/dsv4-dspark` | `13ae97a030c5bb80d8f778e2ea998dd7194ee0ee` |
| DeepSpec | `005e03b81cec38b7da6399833d609ee89a2587f2` |
| TorchSpec | `c4fc3e0f6d9254b642e9def8edc71c58eb4a64f9` |
| AngelSpec | `d3412bed231025ea85d35dbf5e0af44f1ac62a5b` |
| Speculators 主分支 | `65167c0465a78f70b703ed6d84802087708623fd` |
| RedHatAI V4 DFlash 模型 | `e44fc94ceb1e7ed45550d15e782aeadd08050483` |
| DeepSeek 官方 V4 Flash DSpark 模型 | `62af8fffb2f7030cac4de2f0169f5b8d1101b646` |
| Z-Lab Qwen3.6-35B-A3B DFlash 模型 | `f181eece646affea2c38b2765f1aaa01a9734ccd` |

- **S1**：SpecForge [V4 运行文档][S1]、[训练 YAML][SF-V4-YAML]、[草稿 JSON][SF-V4-JSON]、[schema][SF-SCHEMA]、[optimizer][SF-OPT]、[optimizer assembly][SF-ASSEMBLY]。
- **S2**：SpecForge [PR #940][S2]、[64 专家 JSON][SF-MOE64]、[训练 YAML][SF-MOE64-YAML]。旧 [PR #812](https://github.com/sgl-project/SpecForge/pull/812) 已关闭且未合并。
- **S3**：SpecForge [PR #1057][S3]、[原生 MoE JSON][SF-NATIVE]、[训练 YAML][SF-NATIVE-YAML]、[Qwen MoE JSON][SF-QWEN-MOE]、[Qwen 训练 YAML][SF-QWEN-MOE-YAML]。
- **S4**：Sawyer117 [流水线][S4]、[最终 launcher][SW-LAUNCH]、[运行命令记录][SW-LOG]、[技术报告][SW-REPORT]、[架构配置][SW-CONFIG]、[CLI 默认值][SW-CLI]、[dataloader][SW-DATA]、[训练循环][SW-TRAIN]；上游 [RFC #952](https://github.com/vllm-project/speculators/issues/952)、[PR #1041](https://github.com/vllm-project/speculators/pull/1041)。
- **S5**：RedHatAI [模型卡][S5]与[config][RH-CONFIG]。DeepSeek 官方 [DSpark config][DS-CONFIG]、[DSpark 论文 §5.1](https://arxiv.org/html/2607.05147#S5.SS1)描述原生三层 MoE、mHC、window128 与 block5，未公开完整 V4 DSpark 训练 schedule。
- **S6**：DeepSpec [配置目录][S6]、[数据流程][DEEP-DATA]、[优化器][DEEP-OPT]；[DSpark 论文](https://arxiv.org/abs/2607.05147)。
- **S7**：TorchSpec [Qwen3.6 YAML][S7]、[Kimi YAML][TORCH-KIMI]、[Kimi MLA JSON][TORCH-KIMI-JSON]。
- **S8**：AngelSpec [DFlash YAML][S8]与[DSpark YAML][ANGEL-DSPARK]。
- **S9**：Z-Lab/Modal [模型卡][S9]与[config][ZLAB-CONFIG]。
- **S10**：[DFlash 论文][S10]正文与附录 A.1。
- **S11**：Speculators [DFlash best-practices RFC #979][S11]。主分支另有 [5K 示例脚本][SP-EXAMPLE]，其数据量、LR、词表与 RFC 完整实验不同。

[S1]: https://github.com/sgl-project/SpecForge/blob/b520d2b4c08b94413db56392682092d938819c61/docs/recipes/deepseek-v4-flash-dspark-disaggregated.md
[SF-V4-YAML]: https://github.com/sgl-project/SpecForge/blob/b520d2b4c08b94413db56392682092d938819c61/examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark.yaml
[SF-V4-JSON]: https://github.com/sgl-project/SpecForge/blob/b520d2b4c08b94413db56392682092d938819c61/configs/deepseek-v4-flash-dspark.json
[SF-YAML]: https://github.com/sgl-project/SpecForge/tree/b520d2b4c08b94413db56392682092d938819c61/examples/configs/online/disaggregated/external
[SF-JSON]: https://github.com/sgl-project/SpecForge/tree/b520d2b4c08b94413db56392682092d938819c61/configs
[SF-SCHEMA]: https://github.com/sgl-project/SpecForge/blob/b520d2b4c08b94413db56392682092d938819c61/specforge/config/schema.py
[SF-OPT]: https://github.com/sgl-project/SpecForge/blob/b520d2b4c08b94413db56392682092d938819c61/specforge/optimizer.py
[SF-ASSEMBLY]: https://github.com/sgl-project/SpecForge/blob/b520d2b4c08b94413db56392682092d938819c61/specforge/training/assembly.py
[S2]: https://github.com/sgl-project/SpecForge/pull/940
[SF-MOE64]: https://github.com/sgl-project/SpecForge/blob/9061bc866236bf714e11b392517eaf49e3cb6a43/configs/deepseek-v4-flash-dspark-moe.json
[SF-MOE64-YAML]: https://github.com/sgl-project/SpecForge/blob/9061bc866236bf714e11b392517eaf49e3cb6a43/examples/configs/online/disaggregated/external/deepseek-v4-flash-dspark-moe-disaggregated.yaml
[S3]: https://github.com/sgl-project/SpecForge/pull/1057
[SF-NATIVE]: https://github.com/sgl-project/SpecForge/blob/9e12b96941c77134d58b5c814761b9aad79ec2f2/configs/deepseek-v4-flash-moe-dspark.json
[SF-NATIVE-YAML]: https://github.com/sgl-project/SpecForge/blob/9e12b96941c77134d58b5c814761b9aad79ec2f2/examples/configs/online/disaggregated/external/deepseek-v4-flash-moe-dspark.yaml
[SF-QWEN-MOE]: https://github.com/sgl-project/SpecForge/blob/9e12b96941c77134d58b5c814761b9aad79ec2f2/configs/qwen3-8b-moe-dspark.json
[SF-QWEN-MOE-YAML]: https://github.com/sgl-project/SpecForge/blob/9e12b96941c77134d58b5c814761b9aad79ec2f2/examples/configs/online/disaggregated/managed-local/qwen3-8b-moe-dspark.yaml
[S4]: https://github.com/Sawyer117/speculators/blob/13ae97a030c5bb80d8f778e2ea998dd7194ee0ee/docs/deployment/ascend-npu-dsv4-dspark-pipeline.md
[SW-LAUNCH]: https://github.com/Sawyer117/speculators/blob/13ae97a030c5bb80d8f778e2ea998dd7194ee0ee/examples/ascend_npu_dflash/train_dsv4_dspark.sh
[SW-LOG]: https://github.com/Sawyer117/speculators/blob/13ae97a030c5bb80d8f778e2ea998dd7194ee0ee/docs/deployment/ascend-npu-dsv4-worklog.md
[SW-REPORT]: https://github.com/Sawyer117/speculators/blob/13ae97a030c5bb80d8f778e2ea998dd7194ee0ee/docs/deployment/ascend-npu-dsv4-dspark-technical-report.md
[SW-CONFIG]: https://github.com/Sawyer117/speculators/blob/13ae97a030c5bb80d8f778e2ea998dd7194ee0ee/src/speculators/models/dsv4_dspark/config.py
[SW-CLI]: https://github.com/Sawyer117/speculators/blob/13ae97a030c5bb80d8f778e2ea998dd7194ee0ee/scripts/train.py
[SW-DATA]: https://github.com/Sawyer117/speculators/blob/13ae97a030c5bb80d8f778e2ea998dd7194ee0ee/src/speculators/train/dataloader.py
[SW-TRAIN]: https://github.com/Sawyer117/speculators/blob/13ae97a030c5bb80d8f778e2ea998dd7194ee0ee/src/speculators/train/trainer.py
[S5]: https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-speculator.dflash/blob/e44fc94ceb1e7ed45550d15e782aeadd08050483/README.md
[RH-CONFIG]: https://huggingface.co/RedHatAI/DeepSeek-V4-Flash-speculator.dflash/blob/e44fc94ceb1e7ed45550d15e782aeadd08050483/config.json
[DS-CONFIG]: https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-DSpark/blob/62af8fffb2f7030cac4de2f0169f5b8d1101b646/config.json
[S6]: https://github.com/deepseek-ai/DeepSpec/tree/005e03b81cec38b7da6399833d609ee89a2587f2/config
[DEEP-DATA]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/scripts/data/README.md
[DEEP-OPT]: https://github.com/deepseek-ai/DeepSpec/blob/005e03b81cec38b7da6399833d609ee89a2587f2/deepspec/utils/optim.py
[S7]: https://github.com/lightseekorg/TorchSpec/blob/c4fc3e0f6d9254b642e9def8edc71c58eb4a64f9/configs/sglang_qwen36_35b_dflash.yaml
[TORCH-KIMI]: https://github.com/lightseekorg/TorchSpec/blob/c4fc3e0f6d9254b642e9def8edc71c58eb4a64f9/configs/vllm_kimi_k3_dspark_stage1.yaml
[TORCH-KIMI-JSON]: https://github.com/lightseekorg/TorchSpec/blob/c4fc3e0f6d9254b642e9def8edc71c58eb4a64f9/configs/draft_models/kimi_k3_dspark_mla.json
[S8]: https://github.com/Tencent/AngelSpec/blob/d3412bed231025ea85d35dbf5e0af44f1ac62a5b/configs/sglang_qwen3_8b_dflash.yaml
[ANGEL-DSPARK]: https://github.com/Tencent/AngelSpec/blob/d3412bed231025ea85d35dbf5e0af44f1ac62a5b/configs/sglang_qwen3_8b_dspark.yaml
[S9]: https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/blob/f181eece646affea2c38b2765f1aaa01a9734ccd/README.md
[ZLAB-CONFIG]: https://huggingface.co/z-lab/Qwen3.6-35B-A3B-DFlash/blob/f181eece646affea2c38b2765f1aaa01a9734ccd/config.json
[S10]: https://arxiv.org/html/2602.06036
[S11]: https://github.com/vllm-project/speculators/issues/979
[SP-EXAMPLE]: https://github.com/vllm-project/speculators/blob/65167c0465a78f70b703ed6d84802087708623fd/examples/train/dflash_qwen3_8b_ultrachat_online_5k_bestpractices.sh
[N1]: https://github.com/china-qijizhifeng/NexSpecTrainer/blob/764eb921e22e96ec64436c6f0807a97181df1a0d/docs/deepseek-v41-native.md
[N2]: https://github.com/china-qijizhifeng/NexSpecTrainer/blob/764eb921e22e96ec64436c6f0807a97181df1a0d/docs/muon.md
[N3]: https://github.com/china-qijizhifeng/NexSpecTrainer/blob/764eb921e22e96ec64436c6f0807a97181df1a0d/docs/validation/deepseek-v41-20261103.md
