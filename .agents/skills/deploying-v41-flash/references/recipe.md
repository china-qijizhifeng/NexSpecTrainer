# V4.1 Flash recipe：版本、参数与证据边界

## 固定参考版本

以下是已完成源码/文档核对的参考组合，不是对未来镜像或当前运行状态的保证。使用其他版本时重新核对其实际命令、参数解析与验收证据。

| 项目 | 参考值 |
| --- | --- |
| 官方模型 | `deepseek-ai/DeepSeek-V4.1-Flash-1017` |
| 模型 revision | `1bae0e1088993b9b82de08f59de5be5a34e42170` |
| SGLang 镜像 | `lmsysorg/sglang@sha256:c6da252c6837bb062184b01c7aa2df7a2aadb1ed4944f236d7027174c73c45fa` |
| serving commit | `38d9e4a3fe4a493b0ae468477a274e83defda81a` |
| 该源码依赖的 deepseek-recipe | `0.2.0`；部署时还需核对实际安装版本 |
| 文档核对 commit | `4c7c775cf4adcbd2c7419d13b51ec838a17e559e` |

该文档 commit 的 cookbook、通用 deployment config 和 DeepSeek model override 与 serving commit 对应文件字节相同；Agentic config 更晚更新。因此不能把最新 Agentic 参数直接视为该镜像已支持。

## 硬件与目标

| 硬件 | 来源中的配置 | 采用时的限制 |
| --- | --- | --- |
| H100 | Agentic 单节点 TP8/EP8、CPU Engram、memory fraction0.7、chunk4096、decode graph cap64 | 历史原版 Flash/旧镜像使用 Marlin，性能点含模拟接受长度；新的 Flash-1017 组合要实测 |
| H200 | 通用单节点 TP8/EP8、memory fraction0.8、graph cap64 | 低延迟生成面板遗漏 DSPARK flags，与正文不一致；先审查实际命令 |
| B200/B300/GB300 | 通用低延迟单节点 TP4/EP4、memory fraction0.8、DSPARK5、graph cap64 | 后端和硬件能力不同；另有两张 GB300+CPU Engram 的 CI 配置，不能混用验收范围 |

通用面板的 Flash-1017 cells 标记 `verified=false`。通用 recipe 只提供单节点配置，不能据此宣称跨节点 TP16 最优。NVIDIA 高吞吐配置会关闭 speculation；需要区分 DSPARK 的单请求收益与副本扩容的总吞吐收益。PD/CP、PP、DeepEP 等应按具体版本和拓扑另行核对；参考源码中的一项 PD+CP 测试仍为 disabled。

## H100 参考起点

每 worker8×H100 80GB、TP8/EP8。以下参数是 baseline 起点，尚不构成所有并发、长度或性能目标的验收：

```text
--trust-remote-code
--tp-size 8
--ep-size 8
--engram-config '{"placement":"cpu"}'
--mem-fraction-static 0.7
--speculative-algorithm DSPARK
--speculative-dspark-block-size 5
--cuda-graph-max-bs-decode 64
--chunked-prefill-size 4096
--max-running-requests 8
--context-length 131072
--prefill-decode-interval 16
--reasoning-parser auto
--tool-call-parser auto
--enable-metrics
```

这只是模型参数表；监听地址、认证、路由、资源限制与租期另行配置。完整机器可读样例在 [candidate-profile.json](../../../../acceptance/dspark-stage1-test-20261104/serving-baseline/candidate-profile.json)。

在参考源码中，H100 自动解析为 attention `dsv4`、MoE `flashinfer_mxfp4`，SM90 为 CUTLASS MXFP4 weights×BF16 activations。记录实际解析值，不因旧示例强制 Marlin，也不套用 Blackwell MegaMoE/MXFP8 dense `flashinfer_cutedsl` 参数。

CPU Engram 两张表约189GiB/worker，同时需要模型加载、进程、图捕获等主存预算，以及 KV/并发所需 HBM。新版 `--engram-config` 代替旧 host-table 环境变量。自动 decoder SWA bounded replay 不与 full prefill 数值等价；严格 full-prefill 对照单独记录。参考版本的 `--enable-deterministic-inference` 拒绝 dsv4，不以开启该 flag 代替数值诊断。

128K 是此起点的部署边界；模型卡的1M能力不是本配置已验收1M的证据。并发8、graph cap64和两个副本均是有界起点，之后依真实 TTFT、TPOT、吞吐与内存测量调整。

## Chat、reasoning 与工具

服务保留 tokenizer。SGLang 的 dsv41 adapter 已接入 deepseek-recipe，`auto` parser 对应该版本的 `deepseek-v41` reasoning 与 `deepseekv41` tool parser；V4.1 spaced DSML 不能用旧 parser 随意替代。deepseek-recipe 负责协议转换、编码和输出解析，本身不执行模型推理、提供生产 HTTP 服务或执行工具。该模型没有通用 HF Jinja chat template。

参考 adapter 使用 `chat_template_kwargs.thinking` 控制 thinking，支持对应版本的 effort 语义。按实际 API 契约验证 reasoning 字段、tool-call JSON、工具结果回填与连续多轮，stream/nonstream 都要保留完整语义。

真实按 EOS 停止，采样与最大输出由请求和服务边界共同决定。Flash-1017 model card 推荐 temperature1、top_p1、完整模型 context1M、max_tokens 至少256K；这些是模型级建议，必须受部署的上下文和输出预算约束。旧 cookbook reasoning 评测的 top_p0.95、held-out test 的 temp0/1024/单并发均不是线上统一默认值。

Agentic 服务保留 Radix cache，并以会话或前缀亲和避免多副本打散复用。优先验证实际安装的 SGLang router；HiCache/Mooncake 和跨节点并行需要另行预算及验收，不能因示例出现就默认启用。

## 真实计数与性能

H100 InferenceX 脚本在非 accuracy 模式设置 `SGLANG_SIMULATE_ACC_LEN=3.51`、`SGLANG_SIMULATE_ACC_METHOD=match-expected` 和 `SGLANG_SIMULATE_ACC_TOKEN_MODE=real-draft-token`。真实服务不设置这些变量，也不复制模拟 draw-ahead 或 benchmark TP 同步削减；3.51不是已测得的真实接受长度。

报告原生 raw proposed/correct/verify 与真实返回 completion。`tau=sum(completion)/sum(verify)`，接受率 `sum(raw_correct)/sum(raw_proposed)`；gamma5时核对 `raw_proposed=verify*5`。最终 verify block 可能计算 EOS 后草稿，不能强制 `completion=raw_correct+verify`。prefill 直接 EOS 的单 token 请求允许 verify0，指标保留 null 而非除零。

保留 SSE chunk 到达时间。TTFT 使用首个有 token 的事件，TPOT 是首包后实际新增 token 的平均总间隔并包含网络；单有 token chunk 时可为 null。不要制造逐 token 时间，也不要用请求均值冒充 verify 加权 tau。若要比较 AR 与 DSPARK 输出等价，需另有相同输入、协议和重复稳定性证据；有差异时只能报告观察性能。

## 固定来源

- [SGLang cookbook](https://github.com/sgl-project/sglang/blob/4c7c775cf4adcbd2c7419d13b51ec838a17e559e/docs/cookbook/autoregressive/DeepSeek/DeepSeek-V4_1.mdx)
- [通用生成配置](https://github.com/sgl-project/sglang/blob/4c7c775cf4adcbd2c7419d13b51ec838a17e559e/docs/src/snippets/configs/deepseek-ai/deepseek-v4_1.jsx)
- [Agentic 配置与来源披露](https://github.com/sgl-project/sglang/blob/4c7c775cf4adcbd2c7419d13b51ec838a17e559e/docs/src/snippets/agentx/deepseek-ai/deepseek-v4_1.jsx)
- [H100 InferenceX 原始脚本](https://github.com/SemiAnalysisAI/InferenceX/blob/510568c51a127f46c6fcc54d66e925395a512ed1/benchmarks/single_node/agentic/dsv41flash_fp4_h100_sglang_mtp.sh)
- [Flash-1017 model card](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/blob/1bae0e1088993b9b82de08f59de5be5a34e42170/README.md)
- [deepseek-recipe](https://github.com/deepseek-ai/deepseek-recipe/tree/935817c66f5d6e64838f419692545f06787e485f)
- [来源 URL/SHA256/bytes 归档](../../../../acceptance/dspark-stage1-test-20261104/serving-baseline/recipe-review/verified-sources.json)
