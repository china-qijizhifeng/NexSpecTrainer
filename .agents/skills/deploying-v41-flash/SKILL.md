---
name: deploying-v41-flash
description: 部署 DeepSeek V4.1 Flash/Flash-1017 与官方 DSPARK，核对 SGLang recipe、硬件后端、API 验收及 NorthGate 接入。Use when preparing, deploying or publishing a V4.1 Flash serving baseline；训练使用 training-run。
---

# Deploying V4.1 Flash

## 选择部署配置

1. 先读 [版本、硬件与 recipe](references/recipe.md)，核对实际生成命令及其原始性能脚本，而不只读 cookbook 正文。固定模型 snapshot、镜像 digest、serving commit 和实际解析配置；升级其中一项时重新核对兼容性。
2. 官方 baseline 使用 checkpoint 随附的 DSPARK draft。在本参考版本中使用 `DSPARK` 算法和 block size5；历史 `official-mtp` 名称不代表另一个 EAGLE/MTP 启动路径。只有用户选择训练 draft 时才使用相应导出与验收回执。
3. 区分单请求延迟与多用户吞吐。H100 的参考起点是每 worker8卡、单节点 TP8/EP8、CPU Engram；多个独立 worker 扩容需要支持前缀复用的路由。副本数按用户需求和当前资源决定，两个副本不是已测得的最优值。
4. 保留 tokenizer、Radix cache、CUDA Graph 和分块 prefill，开启 reasoning/tool parser，真实按 EOS 停止。不要把固定 eager、禁 cache、单并发的 held-out test 参数直接当服务配置；服务和测评分别取证。

## 准备与启动

BP Dev 部署或 NorthGate 发布时再读 [平台与接入](references/bp-dev-northgate.md)。只记录或审查 recipe 时不创建资源。

- 使用独立运行目录、缓存、端口、alias 和一次性提交意图；查已有身份与收据后再提交。提交结果不确定时先查精确 API，不自动重提或恢复旧 controller。
- 将可审查脚本和配置纳入仓库，格式化后冻结执行 SHA。后续归档或 GitHub 发布不得改写运行中的文件和历史回执；需要修复运行代码时建立新的 attempt。
- 在实际 worker 核对卡型/数量、驱动、Torch、protobuf backend、SGLang、deepseek-recipe、serving 源码与官方模型/draft 身份。通用 `nvidia.com/gpu` 配额不能证明卡型，模型下载的历史完整 SHA 回执与本轮重新 hash 必须分开描述。
- H100 同时预算 HBM、CPU Engram、模型加载峰值、KV、图捕获和并发。Engram 两张表约189GiB/worker；这不是完整 host RAM 预算。
- 优先使用已核验的固定 digest 缓存。确需离线导入时按平台引用中的 archive/CRI 契约操作；导入成功、server ready、API 验收和发布完成分别记录。
- 网络入口启用 worker/router 认证。凭据从私密文件加载，不进入 YAML、OS argv、Git 或公共日志；解析后的 server args/server-info 也可能含密钥，需要脱敏。

## 验收与发布

先验收一个 worker，再扩展同配置副本；首个 worker 暴露的新错误不能在其他副本上重复试出。

1. 确认实际后端、parser、Engram placement、cache、graphs 和 decoder SWA bounded replay 的解析值。依赖 CPU 检查或 `Running` 不能代替模型执行。
2. 验证未认证请求被拒绝，以及模型列表、chat、reasoning、工具调用与工具结果回填；stream/nonstream 均需覆盖。验证真实 EOS/finish reason、SSE 完成与原生 spec 计数，再做有界并发和长前缀验收。
3. 扩容后核对每个 worker 身份与健康，验证实际安装版本的路由器、worker 认证、健康检查和会话/前缀亲和。保留真实 draft 接受计数，不使用模拟接受长度。
4. 用户要求 NorthGate 上架时，先有可达且通过验收的上游，再创建或更新本用户目标模型；最后核对模型可见性、真实端到端请求、stream、reasoning 和工具往返。`onboarding/check` 只验证接入前提。
5. 保存配置、版本、身份、资源限制/租期、验收范围和脱敏回执。失败或到期先留证再按精确 identity 清理；健康服务按用户要求保留，不能当 benchmark 结束就删除。

## 故障与性能边界

- 首次运行、协议或计数错误后暂停同类动态验证；按仓库 AGENTS 完整检查配置、契约、调用链和同类实现，批量修复并补必要回归后才新 attempt 验证。
- H100 历史 Agentic 性能点含 synthetic acceptance；通用 Flash-1017 面板的 `verified=false` 不能继承成已验收。先测真实工作负载，再报告延迟与吞吐。
- 有输出差异就保持 `lossless_greedy_verified=false`，仅报告该配置下观察性能。EOS 后差异、bounded replay 或参数变更都不能证明 FP4 AR/verify 数值根因已修复；本参考版本也不能直接启用拒绝 dsv4 的全局 deterministic flag。

历史来源和运行样例见 [recipe 核对记录](../../../acceptance/dspark-stage1-test-20261104/serving-baseline/RECIPE-REVIEW.md)。样例中的目录、节点、身份和验收状态不能原样当作新部署事实。
