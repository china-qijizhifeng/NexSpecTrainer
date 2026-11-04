# V4.1 Flash + dSpark 服务部署核对

核对时间：2026-11-04，北京时间。此记录是部署依据与候选配置，尚不是 GPU 性能验收或 NorthGate 上架回执。

## 结论

在当前 BP Dev 的 H100 环境，先采用两个独立的单机 TP8/EP8 worker，共 16 张 H100，并通过支持前缀复用的路由入口接入 NorthGate。这是面向多用户容量的部署选择，不是已测得的最优拓扑，也不表示单个请求会比一个 TP8 worker 更快。官方通用面板只提供单节点配置，没有证据支持直接把跨节点 TP16 当作最佳实践。

baseline 使用官方 Flash-1017 checkpoint 中随模型附带的 dSpark，不加载训练 step25/118。V4.1 在本 SGLang 版本中只接受 DSPARK 这一推测解码算法，不能改成单独的 EAGLE/MTP 参数路径。历史文件中的 official-mtp 组名不改变这一事实。

## 来源与版本

- [SGLang V4.1 cookbook](https://github.com/sgl-project/sglang/blob/4c7c775cf4adcbd2c7419d13b51ec838a17e559e/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx)。同时检查了实际生成命令的 deployment config，而非只读页面正文。
- [通用部署配置](https://github.com/sgl-project/sglang/blob/4c7c775cf4adcbd2c7419d13b51ec838a17e559e/docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx)。该文件和 cookbook 与现有镜像对应的 38d9e4a3f 源码逐字相同。
- [最新 Agentic 配置及来源披露](https://github.com/sgl-project/sglang/blob/4c7c775cf4adcbd2c7419d13b51ec838a17e559e/docs/src/snippets/agentx/deepseek-ai/deepseek-v4_1.jsx)。它比镜像对应的文档更新，不能据此认定旧镜像支持全部新示例。
- [H100 InferenceX 原始脚本](https://github.com/SemiAnalysisAI/InferenceX/blob/510568c51a127f46c6fcc54d66e925395a512ed1/benchmarks/single_node/agentic/dsv41flash_fp4_h100_sglang_mtp.sh)。该脚本在非 accuracy 模式下显式设置模拟接受长度 3.51。
- [Flash-1017 官方 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md)。其生产编码建议指向 deepseek-recipe。
- [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)：协议转换、编码、输出解析库；本身不提供模型推理或生产 HTTP 部署。

抓取地址、完整 SHA256 与字节数保存在 recipe-review/verified-sources.json。实际执行镜像仍应固定 digest，不能用最新文档暗中替换已有 test attempt 的代码或协议。

## 硬件与 recipe 的边界

| 硬件 | 文档中的配置 | 使用边界 |
| --- | --- | --- |
| H100 | Agentic TP8/EP8，Engram 放 CPU，chunk 4096，memory fraction 0.7，decode graph cap 64 | 历史镜像使用 Marlin；性能点披露 synthetic acceptance，不能当作真实 dSpark 接受率或加速成绩 |
| H200 | 通用 TP8/EP8，memory fraction 0.8，decode graph cap 64 | 低延迟面板漏了 DSPARK flags，与正文不一致；不能直接复制 |
| B200/B300/GB300 | 通用低延迟 TP4/EP4，DSPARK block 5，memory fraction 0.8，decode graph cap 64 | 后端由平台自动选择；不能套到 SM90；两张 GB300 加 CPU Engram 是另一个 CI 配置 |

Flash-1017 在通用面板的各项配置仍标记 verified=false。H100 历史 Agentic 示例使用原版 Flash 和更旧镜像。因此新的 Flash-1017/H100 服务必须实测，不能把邻近配置的 verified 状态继承过来。

当前 WeaverRSI/proj-32 API 返回通用 GPU deserved=64，B300 deserved=0。容量监控显示有空闲卡，但不保证存在两个满足 CPU、内存、节点限制与镜像条件的完整 H100 节点；排队不等于失败。

## 本次应采用的原则

1. 使用 V4.1-capable preview 镜像并锁定 digest。当前缓存镜像是 c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa，对应源码 38d9e4a3fe4a493b0ae468477a274e83defda81a。不能换用一个普通 release tag 并假设支持。
2. 低延迟启用 `--speculative-algorithm DSPARK --speculative-dspark-block-size 5`。高吞吐 NVIDIA recipe 会关闭 speculation；扩容带来的总吞吐和 dSpark 的单请求收益需分别测量，不能预报加速倍数。
3. 不强行指定旧 Marlin、Blackwell MegaMoE 或 MXFP8 dense 后端。当前源码会在 H100 自动选择 dsv4、flashinfer_mxfp4；SM90 MoE 是 CUTLASS MXFP4×BF16。Blackwell dense 的 flashinfer_cutedsl 选择不适用于 H100。记录真正解析的 backend。
4. H100 使用 `--engram-config '{"placement":"cpu"}'`。新版参数替代旧 host-table 环境变量。Engram 两张表约 189 GiB，每个 worker 都需要足够 host RAM；同时为 KV、图捕获和并发留下 HBM。
5. 服务保留 tokenizer，并开启 `--reasoning-parser auto --tool-call-parser auto`。SGLang 的 dsv41 adapter 已使用 deepseek-recipe；无需自建通用 Jinja 模板。V4.1 的 spaced DSML 需要 deepseekv41 parser。确认 reasoning、工具调用在 stream/nonstream 都能正确回传。
6. 真实按 EOS 停止，最大输出预算与 thinking 开关由服务边界和请求决定。不照搬 test 的固定 1024 预算、temp0、单并发或 skip-tokenizer-init。Flash-1017 model card 推荐 temperature=1、top_p=1；cookbook 中旧 reasoning 评测的 top_p=0.95 不是新版模型的统一线上默认值。
7. Agentic 服务保留 Radix prefix cache，并为同一会话提供稳定路由；先用有限并发和 chunked prefill，之后再依据实际 TTFT/TPOT/显存扩容。不要照搬 test 的禁 radix、全 eager 和不分块 prefill。HiCache/Mooncake 另行预算与验收，不把示例的可选借用配置当作已测最优组合。
8. 官方 decoder SWA bounded replay 通常默认开启，但文档明确它不与 full prefill 数值等价。上线应记录其解析值，验收多轮/工具行为；严格 full-prefill 对照另列。全局 deterministic-inference 当前拒绝 dsv4，不能盲开。
9. 禁用模拟接受长度，保留正常 TP 同步。原始 H100 性能脚本的 synthetic AL=3.51 不能复制进真实服务，SGLANG_OPT_DSV41_DSPARK_SIM_DRAW_AHEAD 等模拟优化也不应启用。
10. 暂不以跨节点 TP16、PP、DeepEP、PD 分离作为首个 baseline。它们有额外硬件与版本限制；PD+CP 的某项 CI 在本源码还明确 disabled，不能用存在测试文件证明已验收。

## 候选启动配置与上线验收

候选配置在 candidate-profile.json，尚未作 GPU 验证。每 worker TP8/EP8、CPU Engram、memory fraction 0.7、chunk 4096、max-running-requests 8、decode graph cap 64、DSPARK block 5。先给出明确 128K 上下文边界；model card 的 1M 能力不等于此部署已经通过 1M 验收。

启用 graphs、prefix cache、真实 tokenizer 和多并发后，这与 held-out test 的固定 eager 配置不同，不改写 test 的结果或把 test 吞吐当作线上服务吞吐。尚未定位的 FP4 AR/verify greedy 差异也不因此消失；不能称为无损加速。

上架前要求：实际节点/8H100/镜像源码/官方模型和 bundled draft 身份通过；CPU 编码解析契约通过；chat、reasoning、工具调用与工具结果回填在 stream/nonstream 通过；EOS/finish reason 与真实 spec 计数通过；有限并发和长前缀请求通过；两个 worker 均健康后才注册入口，最后通过 NorthGate 实际请求和模型可见性验收。当前账号的 `/api/onboarding/check` 已通过 dry-run，目标分组 token-bp-self；该 dry-run 不检查 GPU readiness 或上游可达性。

现有 test-r4 及其 controller、job、冻结文件保持独立。本记录不表示已创建 serving job 或 NorthGate 渠道。
