"""Summarize the completed validation probe without reading private responses."""

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def write(name, value):
    (ROOT / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def main():
    result = json.loads((ROOT / "result.json").read_text())
    audit = json.loads((ROOT / "completion-audit.json").read_text())
    assert audit["passed"] and audit["all_aliases_absent"]
    assert result["request_audit"] == {"passed": True, "requests": 24}
    observations = {}
    for mode, summary in result["modes"].items():
        rows = summary["requests"]
        assert [r["index"] for r in rows] == list(range(12))
        observations[mode] = {
            "decode_unique_output_hashes": len(
                {r["output_sha256"] for r in rows[6:10]}
            ),
            "decode_first_divergence_zero_based": [
                r["first_difference_zero_based"]
                for r in summary["original_decode_comparisons"]
            ],
            "decode_index19_historical_candidate_margins": [
                r["numeric_steps"][19]["historical_candidate_margin"]
                for r in rows[6:10]
            ],
            "matching_prefix_candidate_logprob_max_abs_differences": [
                r["max_abs_candidate_logprob_difference"]
                for r in summary["original_decode_comparisons"]
            ],
            "history_validation_output_lengths": [
                r["output_tokens"] for r in rows[4:6]
            ],
            "history_validation_matches_frozen61": [
                r["same_as_r7_recording"] for r in rows[4:6]
            ],
            "old_prompt_matches_r7": all(r["same_as_r7_recording"] for r in rows[:4]),
            "fresh_prefix_prefill_comparison": summary[
                "common_prefix_prefill_comparison"
            ],
        }
    write(
        "final-observation.json",
        {
            "probe_complete": True,
            "requests": 24,
            "held_out_requests": 0,
            "formal_r7_aa_remains_failed": True,
            "numerical_root_cause_confirmed": False,
            "numerical_fix_validated": False,
            "all_aliases_absent": True,
            "intervention_effective_config_verified": True,
            "observations": observations,
            "cross_mode_original_decode": result["cross_mode_original_decode"],
            "cross_mode_common_prefix_prefill": result[
                "cross_mode_common_prefix_prefill"
            ],
            "conclusion": "Disabling FlashInfer allreduce fusion is insufficient to restore repeatability in these fixed validation requests.",
            "limits": [
                "Native returned scores vary on matching preceding token prefixes; the responsible operator is not localized.",
                "This intervention does not exclude allreduce as one contributing factor.",
                "Finite agreements do not establish determinism; logprob instrumentation can perturb execution.",
                "Fresh prefix-prefill and incremental decode are distinct execution paths.",
                "Worker checked raw SSE, EOS, finite logprobs and counts during capture; SSH audit only checked public summaries and private file stat.",
                "All48 raw response/event files remain uid0 mode0600, unreadable by SSH UID; no independent posthoc private replay was performed.",
            ],
        },
    )
    (ROOT / "RESULT.zh.md").write_text(
        """# dSpark 数值探针 r2：完成记录

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
"""
    )
    (ROOT / "CONTINUATION.md").write_text(
        """# dSpark 数值定位接续

## 已完成，禁止重启

探针 r2 的 24 个 validation 请求于 2026-11-04 13:08:05 UTC 完成；退出码 0，13:08:27 UTC 清理完成，精确 API 已确认 alias 为空。job `job-5c838f4eb38e-20261104200236`，alias `denovo-dspark-aa-numerics-20261104-r2`。controller 已退出，`controller_verified=false` 是终态预期。所有 launch/started/submit intents 仍在，禁止重跑任一入口及旧 job/controller/训练。当前没有本次数值定位的活跃 GPU 作业。

本地 `/Users/shom/Downloads/dspark-aa-numerics-20261104-r2`；远端 `/gpfs/users/linlizhi/dspark-aa-numerics-20261104-r2`；SSH `linlizhi-bp-dev`。先读 `RESULT.zh.md`、`final-observation.json`、`completion-audit.json`、`plan.json`、`source-provenance.json`。`monitor.py` 和新增 `collect_completion.py` 仅只读远端并保存本地证据；不启动或清理资源。冻结 33 项 manifest SHA `e0c85a2eed5adf12002067a88a2dad47e95d4ad86e396aee55c8e94d40569d79` 已复核，不能改写冻结文件。

原配置 4 次长输入 decode 有 3 种输出，关闭 fusion 后仍有 2 种；有效开关已核实。原配置第 20 token 历史候选分数差变号；对照组该处打平，但仍于第 22 token 分歧。有限诊断不证明确定性或数值修复。正式 r7 A/A 失败、held-out test 0，门槛不变。原始 48 份响应/事件 uid0/0600 不可读是预期，不改权限，不声称独立回放。

## 下一步

仓库 `/Users/shom/Downloads/NexSpecTrainer-dspark-r5-nfc-fix-20261104`，边界定位计划见 `acceptance/dspark-stage1-test-20261104/numeric-localization/PLAN.zh.md`。先固定实际源码和模型的 mHC、稀疏索引、注意力、MoE、logits/argmax 边界，完成有界 instrumentation、覆盖标记及 CPU 回归，再准备独立单次 GPU 诊断；不盲扫 flag，不触碰 held-out 输入。插件加载失败会被上游捕获，因此必须显式核对所有 rank 覆盖，不得静默退化为未采集。

下一次资源仍单组 8H100/64CPU/512Gi、不固定节点，WeaverRSI/proj-32。新错误先留证停止，完整静态排查与同类批量回归后才恢复动态验证。公开控制 0644、原始输出 0600，保留 init 私密权限。终态先 receipt 再精确清理并确认 alias 为空。正式评测仍须 10 对完整 A/A 通过，随后四组各 247、共 988 请求审计；真实 EOS1/temp0/max1024，冻结输入与指标契约不变。

所有新脚本和必要非敏感证据检查后推 GitHub main，核 API SHA 与祖先；不提交凭据、输入输出、权重或 ckpt。不再清理 checkpoint，不碰他人及健康 baseline A/B。Harbor 已完成；NorthGate 剩余工作见服务目录 CONTINUATION。正常健康或无变化安静；只通知实质结果、失败、完成、资源问题或需要决定。保持 automation。
"""
    )


if __name__ == "__main__":
    main()
