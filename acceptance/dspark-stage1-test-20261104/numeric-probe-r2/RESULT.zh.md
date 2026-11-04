# dSpark 数值探针 r2：完成记录

24 个 validation 请求已完成并通过运行审计；关闭 FlashInfer allreduce fusion 后仍有输出差异，数值问题未解决。正式 r7 的 A/A 仍失败，held-out test 已执行 **0** 个请求。

作业 `job-5c838f4eb38e-20261104200236` 于 2026-11-04 13:08:05 UTC 完成，退出码 0；controller 于 13:08:27 UTC 完成精确删除，随后 API 确认 alias 为空。实际 node054、8×H100 80GB、64CPU/512Gi，调度未绑定节点。已复核 r2 的 33 项和 r7 的 860 项冻结文件 SHA。

| 观测 | 原配置 | 关闭 allreduce fusion |
| --- | --- | --- |
| 4 次相同长输入、最多 32 tokens 的不同输出数 | 3 | 2 |
| 后 3 次相对首次的首次分歧位置（从 0 开始） | 21、19、相同 | 21、相同、相同 |
| 第 20 token 的两历史候选 logprob 差 | 0.25、0、约 −0.25、0 | 0、0、0、0 |
| 相同前缀内，候选 logprob 相对首次的最大绝对差 | 约 0.8124、0.9531、0.625 | 0.90625、1.125、约 2.002 |
| 普通 validation 历史请求的输出长度 | 59、60 | 56、61 |
| 与 r7 冻结 61-token 参考完全一致 | 否、否 | 否、是 |

两模式的两个旧 prompt 均与 r7 记录一致（831、1024 tokens）。两次“原输入加共同 19-token 前缀、重新 prefill、只生成 1 token”的输出和两候选分数在模式内及跨模式均一致，但它们与增量 decode 使用不同路径，不能据此证明确定性。

有效配置确认对照组 `enforce_disable_flashinfer_allreduce_fusion=true`、`flashinfer_allreduce_fusion_backend=null`；启动参数仅增加这一开关。此次干预不足以恢复重复一致性；不能由此排除 allreduce 的贡献，也不能定位到具体算子。

worker 在请求到达时核验原始 SSE、EOS、计数及 logprob finite，并生成公开摘要。本次 SSH 复核了摘要互相一致、有效配置、清理回执及文件 stat。48 个原始响应/事件文件仍为 uid0/0600，SSH 用户不可读，未改权限、未进行独立事后回放。采集 logprob 本身也可能改变执行。

下一步先沿 mHC 残差、稀疏索引、注意力、MoE 和采样边界设计有界摘要。任何新 GPU 定位都须单独固定计划、覆盖标记、资源和终止条件；保持原 A/A 与正式 test 门槛。
