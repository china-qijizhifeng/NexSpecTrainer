# dSpark 数值边界定位计划

状态：r2 已完成并清理，当前无活跃诊断 GPU job。本文是下一次独立诊断的准备材料，**没有提交新任务**。`bounded_trace.py` 是摘要采集基础组件；SGLang 接入、实际镜像预检和 GPU 覆盖尚未完成，不能据此启动。

原配置与关闭 FlashInfer allreduce fusion 均存在相同 validation 输入的输出分歧。有效开关已核实，单独关闭不足以恢复一致性。正式 r7 A/A 仍失败，held-out test 0；10 对完整 A/A、冻结 61-token 参考及 988 请求审计门槛保持不变。详见相邻 `numeric-probe-r2/RESULT.zh.md`。

## 按优先级验证的假设

1. **稀疏索引或缓存/压缩状态首先变化。** 若同一请求前缀、同层的索引输入摘要相同，但逻辑 top-k 有序索引先变化，之后注意力输出才变化，则优先缩小 indexer/缓存状态路径。仅末行摘要一致不能证明整个 KV/cache 相同。
2. **注意力、MoE 或 mHC 边界内计算首先变化。** 若相同采样输入下，某子层输入摘要一致、输出或下一子层输入先变化，则缩小到该边界。allreduce 干预结果不足以排除其贡献。延迟合并的输出不等于完整层残差，不得为了采集而提前 finalize 或 materialize。
3. **LM head、采样或返回分数路径变化。** 若最后 hidden 摘要一致而完整 logits 行先变化，先检查 LM head/gather；若完整采样前 logits 一致而 argmax/返回值变化，再查采样与响应管线。采样器会原地预处理 logits，必须在原调用前复制有界行并在返回后核对。
4. **异步依赖或缓冲复用造成差异。** 摘要的 GPU→CPU 拷贝会同步并扰动执行。若采集开启后差异消失，只能记录“本次未复现”，不能证明修复；不从带采集的速度推导吞吐结论。

## 固定源码边界

上游 SGLang commit `38d9e4a3fe4a493b0ae468477a274e83defda81a`。本机源码位于 `/Users/shom/Downloads/dspark-evaluation-research-20261104/sglang-pinned-full/python/sglang`。`audit_source_boundaries.py` 对实际文件记 SHA，并校验关键 AST 契约；worker 必须重新核对实际安装源码。

- `models/deepseek_v4.py::DeepseekV4ForCausalLM.forward` 直接调用 `self.model.forward`，仅给 model 注册 nn forward hook 会漏采。应包装实际方法，并核每 rank、每 request、每 forward 的覆盖。
- CUDA 的 `DeepseekV4CudaDecoderLayer.forward` 按 `attn_boundary.prepare → attention → attn_boundary.finish → ffn_boundary.prepare → mlp → output.finish` 执行；mHC 残差和归约可跨边界延后，不能把返回 tensor 当成完整 residual。
- `MQALayer.forward` 输入是已准备的 `x`；输出可能是 `mhc.AttnOutput`。`DeepseekV2MoE.forward` 输出可能是 `MoEOutput` 或 deferred finalize handle。只读取已有 tensor 字段，禁止调用 `get_merged()`、`materialize()` 或额外归约以便采集。
- `attention/dsv4/indexer.py::C4IndexerBackendMixin.forward_c4_indexer` 的 `raw_indices` 是逻辑索引；物理 page indices 随内存分配变化，不能直接当作逻辑不一致。现成 `IndexerTopkCapturer.__init__` 要求 attention TP=1，不能在 TP8 直接启用。若接入 scoped observer，只在单 token decode 创建有界临时索引，不在 50K prefill 开启全量 raw-index 缓冲；observer 路径改变必须写入解释限制。
- `layers/logits_processor.py::LogitsProcessor.forward` 输出 `next_token_logits`，`layers/sampler.py::Sampler.forward` 在 greedy 分支使用 `torch.argmax`，返回 logprob 另经 `log_softmax`。同时记录完整单行字节 SHA、两候选分数和实际选取验证，不能只比较返回的 logprob。
- plugin 加载、执行以及 HookRegistry 应用错误会被记录后继续。因此“plugin 已发现”不足以验收，必须检查每个 target 的实际 patch 身份及 8 rank 的覆盖标记。任一缺失应停止诊断并留证。
- 现有 `TensorDumper` 会把完整中间 tensor 递归复制到 CPU；本任务不能使用它进行全模型/全 prefill dump。

## 有界执行契约

下一次计划仍只读固定 validation 输入：先重放 r2 已冻结的 6 个历史请求（最多 1024 tokens、关闭边界采集），保留已复现的执行历史；随后 4 个相同 50,527-token 输入请求、每个最多 24 个真实生成 tokens 并开启采集，共最多 10 个请求。用 r2 原配置，不混入新 flag，不发 held-out。每个请求单并发，真实 EOS1、temperature0；一次错误即停止，不重试。资源为 8H100/64CPU/512Gi、不固定节点，WeaverRSI/proj-32。新 attempt 必须有独立目录、冻结 manifest、启动防重、worker/controller 时限和精确清理回执。

只采 decode 的有界行：位置、请求序号、已消费 token 前缀 SHA、层号、边界、shape/dtype、所选行字节 SHA 和 finite；token/logits 值留 private，不进公开证据。初次 prefill 可记录已存在的最后一行摘要，但不声称覆盖全部 50K token 或全部缓存。每次比较必须先验证请求 SHA、位置及完整已消费前缀一致，输出分歧后停止跨请求的数值因果比较。

每 rank 最多 16,000 条、8 MiB JSONL；每条最多 4 KiB；每次被复制的 tensor 行最多 2 MiB，切行在 `.cpu()` 和 contiguous 之前。超限、非 finite、缺失必需边界、重复 invocation、额外请求、源 SHA 不符均停止，不截断后继续。所有原始 trace/响应 0600 exclusive create，公开覆盖和摘要 0644。保留 init 077；不 chmod 旧私有证据。

## 动态验证前仍需完成

1. 实现与上述实际方法匹配的 SGLang adapter；覆盖多 rank、prefill/decode/idle、deferred-output 类型和 sampler 独立调用上下文，验证 wrapper 原样调用一次并原样返回。
2. 实际 CPU tensor 回归覆盖 BF16、非连续行、空行、超宽行、非 finite、资源上限、进程重放、权限与不修改输入；Linux/实际镜像再验证插件加载和控制 UID 可读性。已有基础组件 CPU 回归只证明自身边界，不能代替上述 adapter/GPU 验收。
3. 检查每个来源文件 SHA、8 rank patch/覆盖、输入和 EOS/SSE 审计、超时与信号清理，冻结完整计划，再单次提交。
4. 终态先留证，再按精确 identity 清理并确认 alias 空；只报告实际覆盖的第一个分歧边界，不把有限一致性写成确定性或修复。

不重启任何旧任务，不清 checkpoint，不碰健康 baseline A/B。所有新脚本和非敏感证据检查后推 main，并确认 GitHub API SHA 与已发布祖先。
