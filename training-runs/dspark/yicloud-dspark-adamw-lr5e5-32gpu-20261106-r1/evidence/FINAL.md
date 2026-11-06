# 仪电 AdamW peak LR 5e-5 完成汇总

2026-11-06 20:07:13 北京时间平台成功结束；20:14 完成只读终态核验。

- JobId：j-1793942702-213881；run UUID：c363a7e3-a7cf-46a6-8cc2-6897b0dfda40。
- 32 ranks 各完成 118 optimizer steps、472 训练微批和20验证微批，完成回执 passed / frozen_unchanged 均为 true。
- 4节点 exit=0，平台作业与4副本均 Succeeded。
- 全部已记录 step/micro 数值有限，合法 anchor 覆盖检查通过、零重复；峰值 allocated 33.288 GiB，reserved 74.184 GiB。
- 实际 LR 峰值 5.0009329708996855e-5，保留既有 step4 微小 overshoot；seed42 fresh 初始化，无 resume。
- 最终 checkpoint118：training_state step118；common.pt SHA256 与清单一致：dc2f00ce6581e1789f7884e554a463e7a8187a06e4baf09d3086cfb3029fab5b；NFC回执 step118，.metadata 非空，128个非空 distcp 分片覆盖32 ranks。

| 验证步数 | loss | acc |
|---|---:|---:|
| 25 | 0.915946605 | 78.024215% |
| 50 | 0.910117742 | 78.154265% |
| 75 | 0.907236240 | 78.212160% |
| 100 | 0.905706752 | 78.243927% |

没有第118步验证。仪电2e-4第100步 loss 0.912526434、acc 78.254463%；5e-5 loss更低但acc低0.010536个百分点，不能给出单一指标全面胜出的结论。BP H100特征独立采集，跨平台不是仅LR差异。

未做恢复验证，不声称checkpoint可恢复；这些指标不证明实际推理质量提升。2e-4仍在训练，继续原heartbeat。

证据：terminal-verification.json、heartbeat-20261106-2012.json、curves.json。对照图位于 ../adamw-lr-summary-20261106/。

核验脚本首次因远端Python不提供hashlib.file_digest失败；静态确认API版本差异后改为标准sha256分块读取，核验通过。此为只读核验工具兼容问题，训练无报错。
