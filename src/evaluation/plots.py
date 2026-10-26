"""Summary bar charts: our top-1 accuracy at Q3 next to the paper's (Table 1.1)."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.paths import FIGURES as F, TABLES as T
from .paper import PAPER_1_2

OURS, GREY = "#2a6fdb", "#a8a8a8"


def bars(ax, labels, ours, paper, title):
    x = np.arange(len(labels))
    w = 0.38
    b1 = ax.bar(x - w / 2 - 0.01, ours, w, color=OURS, label="this reproduction")
    b2 = ax.bar(x + w / 2 + 0.01, [np.nan if p is None else p for p in paper], w, color=GREY, label="paper")
    for b, v in list(zip(b1, ours)) + list(zip(b2, paper)):
        if v is not None:
            ax.text(b.get_x() + b.get_width() / 2, v + 0.01, f"{v:.2f}", ha="center", va="bottom", fontsize=7, color="#333")
    ax.set_xticks(x, labels, rotation=30, ha="right", fontsize=8)
    ax.set_ylim(0, 1.25)
    ax.set_ylabel("top-1 accuracy at Q3")
    ax.set_title(title, fontsize=10)
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color="#e5e5e5", lw=0.6)
    ax.set_axisbelow(True)
    ax.legend(fontsize=7, frameon=False, ncol=2, loc="upper center")


def main():
    doms = [d for d in ("taxi", "dkg", "blockwords") if (T / f"exp1_2_{d}.json").exists()]
    if doms:
        fig, axes = plt.subplots(1, len(doms), figsize=(4 * len(doms), 3.4), squeeze=False)
        for ax, d in zip(axes[0], doms):
            res = json.loads((T / f"exp1_2_{d}.json").read_text())["all"]
            meths = ["SIPS", "BIRL (unbiased)", "BIRL (oracle)"]
            bars(ax, ["SIPS", "BIRL unbiased", "BIRL oracle"], [res[m]["top1_Q3"] for m in meths],
                 [PAPER_1_2[d][m][5] for m in meths], f"Exp 1.2: {d}")
        fig.tight_layout()
        fig.savefig(F / "summary_exp1_2.png", dpi=150)
    doms = [d for d in ("dkg", "blockwords") if (T / f"exp1_3_{d}.json").exists()]
    if doms:
        fig, axes = plt.subplots(1, len(doms), figsize=(6 * len(doms), 3.6), squeeze=False)
        for ax, d in zip(axes[0], doms):
            r = json.loads((T / f"exp1_3_{d}.json").read_text())
            labels = list(r["ours"])
            bars(ax, labels, [r["ours"][k] for k in labels], [r["paper"].get(k) for k in labels], f"Exp 1.3 robustness: {d}")
        fig.tight_layout()
        fig.savefig(F / "summary_exp1_3.png", dpi=150)

