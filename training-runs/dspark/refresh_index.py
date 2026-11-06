"""Render the run index from archived metadata; no network or training changes."""

import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def value(v):
    return "—" if v is None else str(v).replace("|", "\\|")


def render():
    inventory = json.loads((ROOT / "inventory.json").read_text())
    rows = [
        json.loads((ROOT / name / "run.json").read_text())
        for name in inventory["records"]
    ]
    for r in rows:
        p = r["parameters"]
        lines = [
            f"# {r['id']}",
            "",
            f"状态：**{r['status']}**。",
            f"实时检查时间：{value(r.get('checked_at'))}；归档时间：{r['archived_at']}。",
            "",
            "本页由 `../refresh_index.py` 从 `run.json` 生成。历史文档里的 Running/Pending 仅表示当时状态；以本页状态和对应证据为准。",
            "",
            "| 字段 | 记录 |",
            "|---|---|",
        ]
        for k, v in [
            ("类型", r["category"]),
            ("平台", r["platform"]),
            ("SSH", r["host"]),
            ("Job ID", r.get("job_id")),
            ("Alias", r["alias"]),
            ("UUID", r.get("uuid")),
            ("已完成步数", r.get("completed_steps")),
            ("远端运行目录", r["remote_root"]),
            ("Checkpoint 目录", r.get("checkpoint_root")),
        ]:
            lines.append(f"| {k} | {value(v)} |")
        lines += ["", "## 配置", "", "| 字段 | 值 |", "|---|---|"]
        lines += [f"| {k} | {value(v)} |" for k, v in p.items()]
        e = r.get("latest_eval")
        if e:
            lines += [
                "",
                "## 最近一次验证",
                "",
                f"step **{value(e.get('step'))}**；loss **{value(e.get('loss'))}**；accuracy **{value(e.get('accuracy'))}**（0–1）。",
                "验证步数与训练最终步数可能不同。过拟合、稀疏 anchor、全量 anchor、独立采集的 BP/仪电特征不能直接混排。",
            ]
        lines += [
            "",
            "## 证据",
            "",
            "- [结构化记录](run.json)",
            "- [来源与 SHA256 清单](evidence-manifest.json)",
            f"- 原始本地目录：`{r['local_source']}`",
        ]
        manifest = json.loads((ROOT / r["id"] / "evidence-manifest.json").read_text())
        lines += [f"- [{x['file']}]({x['file']})" for x in manifest]
        if r.get("notes"):
            lines += ["", "## 说明", ""] + [f"- {n}" for n in r["notes"]]
        lines += [
            "",
            "证据文档为原文快照，内部相对路径以其原始目录为基准，来源见清单。未明确提供恢复验收时，不将 checkpoint 文件存在写成可恢复。",
            "",
        ]
        (ROOT / r["id"] / "README.md").write_text("\n".join(lines))

    counts = Counter(r["category"] for r in rows)
    lines = [
        "# DSpark training runs",
        "",
        f"归档 {len(rows)} 条记录："
        + "、".join(f"{k} {v}" for k, v in counts.items())
        + "。",
        "",
        inventory["scope"],
        "",
        "每个运行一个目录，包含 README、run.json、配置/结果快照和带 SHA256 的来源清单。平台 Job ID 与 run UUID 分开保存；准备、失败、取消、替代运行分别登记。",
        "",
        "## 当前训练",
        "",
        "各行时间见明细；只把实时平台 Running 的正式训练列在这里。",
    ]
    groups = [
        ("", [r for r in rows if r["status"] == "运行中" and r["category"] == "训练"]),
        (
            "已完成训练",
            [
                r
                for r in rows
                if r["category"] in ("训练", "过拟合实验")
                and r["status"].startswith("已完成")
            ],
        ),
        (
            "历史尝试与未提交计划",
            [
                r
                for r in rows
                if r["category"] in ("训练", "过拟合实验")
                and r["status"] != "运行中"
                and not r["status"].startswith("已完成")
            ],
        ),
        (
            "采集、探针与评测",
            [r for r in rows if r["category"] not in ("训练", "过拟合实验")],
        ),
    ]
    for title, group in groups:
        if title:
            lines += ["", f"## {title}"]
        lines += [
            "",
            "| Run | 状态 | GPU | LR | Global batch | 步数/预算 | 最近验证 step / acc / loss |",
            "|---|---|---:|---:|---:|---:|---|",
        ]
        for r in group:
            p = r["parameters"]
            e = r.get("latest_eval") or {}
            acc = (
                f"{e['accuracy'] * 100:.4f}%"
                if isinstance(e.get("accuracy"), (int, float))
                else "—"
            )
            loss = (
                f"{e['loss']:.6f}" if isinstance(e.get("loss"), (int, float)) else "—"
            )
            lines.append(
                f"| [{r['id']}]({r['id']}/README.md) | {r['status']} | {value(p.get('gpus'))} | {value(p.get('learning_rate'))} | {value(p.get('global_batch'))} | {value(r.get('completed_steps'))}/{value(p.get('max_steps'))} | {value(e.get('step'))} / {acc} / {loss} |"
            )
    lines += [
        "",
        "## 比较口径",
        "",
        "- BP 与仪电特征独立采集、硬件不同；跨平台对照不能仅归因于学习率。",
        "- 近期全量 anchor 调参组：1,326 train traces、17 validation traces；15,104 train windows 含 51 个既有重复 padding；32 rank 验证补齐为 160 windows。",
        "- Batch 512 组补齐至 15,360 windows，共 30 步；应按消费窗口量或 epoch 比较，不能与 batch 128 同步数直接对比。",
        "- 早期 stage1 为采样 anchor；全量 4,188 trace / 796 步、10 样本过拟合和未微调 MTP 评测另有口径，不能并入同一排名。",
        "- 日志中的准确率是离线指标，不等同于 serving 接受率或任务分数。",
        "",
        "## 更新",
        "",
        "修改对应目录的 `run.json`，添加带检查时间的证据并更新 `evidence-manifest.json`，然后运行：",
        "",
        "```bash",
        "python3 training-runs/dspark/refresh_index.py",
        "```",
        "",
        "只读监控按精确 Job ID 和 UUID 更新；不要用启动时的 run state=verified 代替平台状态。checkpoint、特征、完整日志和运行脚本继续保留在原目录，避免影响仍运行的任务。",
        "",
        "[范围与清单](inventory.json) · [来源聊天](codex://threads/01a1b06c-ef27-73b0-a81b-d4e54a0788bc)",
        "",
    ]
    (ROOT / "README.md").write_text("\n".join(lines))


if __name__ == "__main__":
    render()
