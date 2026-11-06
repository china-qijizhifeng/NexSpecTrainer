import json
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator, PercentFormatter

root = Path(__file__).resolve().parent
runs = [
    ("1e-5 | BP H100", root / "bp-1e-5", "#64748b"),
    ("1e-4 | BP H100", root / "bp-1e-4", "#2563eb"),
    ("5e-4 | YiCloud H200 (stopped)", root / "yicloud-5e-4", "#d97706"),
]
plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "font.size": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titleweight": "bold",
        "axes.edgecolor": "#cbd5e1",
        "savefig.facecolor": "white",
    }
)
fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
fig.subplots_adjust(top=0.80, bottom=0.18, hspace=0.42, wspace=0.26)
fig.suptitle(
    "AdamW learning-rate comparison",
    x=0.075,
    y=0.97,
    ha="left",
    fontsize=20,
    fontweight="bold",
)
fig.text(
    0.075,
    0.925,
    "1,326 traces / 32 GPUs / global batch 128 / all anchors / chunk64",
    color="#475569",
)
for label, path, color in runs:
    d = json.loads((path / "curves.json").read_text())
    train = d["train"]
    val = d["eval"]
    x = np.array([r["step"] for r in train])
    for ax, key in [(axes[0, 0], "acc"), (axes[0, 1], "loss")]:
        y = np.array([r[key] for r in train])
        ax.plot(x, y, color=color, alpha=0.2, lw=1)
        if len(y) >= 10:
            ax.plot(
                x[9:],
                np.convolve(y, np.ones(10) / 10, "valid"),
                color=color,
                lw=2,
                label=label,
            )
    for ax, key in [(axes[1, 0], "eval/avg_acc"), (axes[1, 1], "eval/avg_loss")]:
        ax.plot(
            [r["step"] for r in val],
            [r[key] for r in val],
            "-o",
            color=color,
            lw=2,
            ms=6,
        )
for ax, title in zip(
    axes.flat,
    [
        "Training accuracy",
        "Training loss (logged)",
        "Validation accuracy",
        "Validation loss",
    ],
):
    ax.set_title(title, loc="left")
    ax.set_xlabel("Optimizer step")
    ax.set_xlim(0, 120)
    ax.grid(axis="y", alpha=0.2)
    ax.yaxis.set_major_locator(MaxNLocator(5))
axes[0, 0].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
axes[1, 0].yaxis.set_major_formatter(PercentFormatter(1, decimals=1))
fig.legend(
    *axes[0, 0].get_legend_handles_labels(),
    loc="upper left",
    bbox_to_anchor=(0.067, 0.90),
    ncol=3,
    frameon=False,
    fontsize=10,
)
fig.text(
    0.075,
    0.035,
    "Snapshot: 2026-11-06 12:46 Beijing. BP: 118 updates; YiCloud 5e-4: stopped at 39.\nFaint lines: per update; bold training lines: trailing 10-update mean. Only observed validation points shown.\nYiCloud uses independently captured features and H200: not a strict LR-only comparison.\n2e-4 in progress (startup evidence only). No step-118 evaluation or serving-quality claim.",
    fontsize=9.5,
    color="#475569",
    linespacing=1.5,
)
fig.savefig(root / "adamw-lr-comparison.png", dpi=180)
fig.savefig(root / "adamw-lr-comparison.svg")
print(root / "adamw-lr-comparison.png")
