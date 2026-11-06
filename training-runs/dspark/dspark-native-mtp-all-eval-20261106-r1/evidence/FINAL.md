# 未微调官方 MTP：同验证集 baseline

2026-11-06 北京时间，官方 DeepSeek-V4.1-Flash-1017 原生 MTP，无微调checkpoint、无优化器更新。8×H100，完整17条验证trace/136窗口/323930合法anchor，全部anchor恰好一次；输入、mask、覆盖计数与之前重评测逐窗口相同。

## 对齐既有32卡训练日志

将前24个窗口重复计权，匹配原32rank DistributedSampler的160窗口；accuracy由原始正确数/目标数汇总。loss按32rank分组分母重建，8rank重建与实际loss误差1.81e-08。前向实际用8卡，未声称32卡逐位复现。

| 配置 | 验证step | Accuracy | Loss | 相对官方MTP |
|---|---:|---:|---:|---:|
| 官方MTP未微调 | 0 | 77.4319% | 0.944420（重建） | — |
| adamw_1e5 / all anchors / batch128 | 100 | 78.0624% | 0.913552 | +0.6305个百分点 |
| adamw_1e4 / all anchors / batch128 | 100 | 78.2798% | 0.906610 | +0.8479个百分点 |

最好已完成配方AdamW1e-4相对官方MTP准确率增加0.8479个百分点，loss下降4.00%。这是离线teacher-forced验证，不代表真实serving接受率或SWE成绩。

## 去重136窗口的原始baseline

- Accuracy：77.6484%
- 原生8rank evaluator loss：0.929781
- Teacher top1一致率：80.6880%
- 五个draft位置准确率：86.8567%, 82.5796%, 77.8010%, 72.9178%, 68.0250%
- 此口径不能直接与上表32卡日志的accuracy/loss相减。

## 验收及后处理修复

8rank均完整完成17窗口，全部参数前后SHA不变；0次optimizer更新，0次checkpoint写入。原始GPU任务job-0828a6b23265-20261106185151在GPU评测通过后，因CPU汇总把带不同run名称前缀的sample_id直接比较而退出1，平台终态Failed；原始日志与回执保留。静态核对launch.py和offline_reader.py确认ID由run_id和窗口index组成，实际136个输入SHA、mask SHA、anchor计数完全一致。新summarize_verified.py校验各自预期命名空间及相同index/content，补充真实数据和index/input/mask/count破坏检测回归，通过后仅重跑CPU汇总。aligned-result.json及本最终验收passed；不把原平台Failed改写为Completed。后台monitor已按精确job身份删除，回读空，8卡释放。

证据：evidence/step0-result-rank*.json、evidence/native-initialization-rank*.json、aligned-result.json、identity-regression.json、final-verification.json。
