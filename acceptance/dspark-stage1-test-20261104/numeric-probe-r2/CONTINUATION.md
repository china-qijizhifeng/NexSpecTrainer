# dSpark 数值定位接续

## 已完成，禁止重启

探针 r2 的 24 个 validation 请求于 2026-11-04 13:08:05 UTC 完成；退出码 0，13:08:27 UTC 清理完成，精确 API 已确认 alias 为空。job `job-5c838f4eb38e-20261104200236`，alias `denovo-dspark-aa-numerics-20261104-r2`。controller 已退出，`controller_verified=false` 是终态预期。所有 launch/started/submit intents 仍在，禁止重跑任一入口及旧 job/controller/训练。当前没有本次数值定位的活跃 GPU 作业。

本地 `/Users/shom/Downloads/dspark-aa-numerics-20261104-r2`；远端 `/gpfs/users/linlizhi/dspark-aa-numerics-20261104-r2`；SSH `linlizhi-bp-dev`。先读 `RESULT.zh.md`、`final-observation.json`、`completion-audit.json`、`plan.json`、`source-provenance.json`。`monitor.py` 和新增 `collect_completion.py` 仅只读远端并保存本地证据；不启动或清理资源。冻结 33 项 manifest SHA `e0c85a2eed5adf12002067a88a2dad47e95d4ad86e396aee55c8e94d40569d79` 已复核，不能改写冻结文件。

原配置 4 次长输入 decode 有 3 种输出，关闭 fusion 后仍有 2 种；有效开关已核实。原配置第 20 token 历史候选分数差变号；对照组该处打平，但仍于第 22 token 分歧。有限诊断不证明确定性或数值修复。正式 r7 A/A 失败、held-out test 0，门槛不变。原始 48 份响应/事件 uid0/0600 不可读是预期，不改权限，不声称独立回放。

## 下一步

仓库 `/Users/shom/Downloads/NexSpecTrainer-dspark-r5-nfc-fix-20261104`，边界定位计划见 `acceptance/dspark-stage1-test-20261104/numeric-localization/PLAN.zh.md`。先固定实际源码和模型的 mHC、稀疏索引、注意力、MoE、logits/argmax 边界，完成有界 instrumentation、覆盖标记及 CPU 回归，再准备独立单次 GPU 诊断；不盲扫 flag，不触碰 held-out 输入。插件加载失败会被上游捕获，因此必须显式核对所有 rank 覆盖，不得静默退化为未采集。

下一次资源仍单组 8H100/64CPU/512Gi、不固定节点，WeaverRSI/proj-32。新错误先留证停止，完整静态排查与同类批量回归后才恢复动态验证。公开控制 0644、原始输出 0600，保留 init 私密权限。终态先 receipt 再精确清理并确认 alias 为空。正式评测仍须 10 对完整 A/A 通过，随后四组各 247、共 988 请求审计；真实 EOS1/temp0/max1024，冻结输入与指标契约不变。

所有新脚本和必要非敏感证据检查后推 GitHub main，核 API SHA 与祖先；不提交凭据、输入输出、权重或 ckpt。不再清理 checkpoint，不碰他人及健康 baseline A/B。Harbor 已完成；NorthGate 剩余工作见服务目录 CONTINUATION。正常健康或无变化安静；只通知实质结果、失败、完成、资源问题或需要决定。保持 automation。
