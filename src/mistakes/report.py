"""Tables and figures for Experiments 2.1 / 2.2."""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .compare import p_true

KIND_TITLES = {"optimal": "optimal", "failure": "irreversible plan failure", "backtrack": "short-sighted (backtracking)",
               "action": "action mistakes", "goal": "goal mistakes"}


def kind_table(stimuli, posteriors, kinds) -> tuple[str, dict]:
    """Markdown table: mean P(true goal) per stimulus set and model."""
    pt = p_true(stimuli, posteriors)
    models = list(pt)
    stats = {k: {m: float(np.mean([v for v, s in zip(pt[m], stimuli) if s.kind == k])) for m in models} for k in kinds}
    stats["all"] = {m: float(np.mean(pt[m])) for m in models}
    lines = ["| stimulus set | " + " | ".join(models) + " |", "|---|" + "---|" * len(models)]
    for k in [*kinds, "all"]:
        lines.append(f"| {KIND_TITLES.get(k, k)} | " + " | ".join(f"{stats[k][m]:.2f}" for m in models) + " |")
    return "\n".join(lines), stats


def curves(stim, posteriors_for_stim: dict, path, title=None):
    """P(true goal) over time for every model on one stimulus (mean over runs, band = min..max)."""
    fig, ax = plt.subplots(figsize=(5.5, 3.2))
    for m, runs in posteriors_for_stim.items():
        y = np.array([r[:, stim.g] for r in runs])
        t = np.arange(y.shape[1])
        ax.plot(t, y.mean(0), label=m, lw=1.6)
        ax.fill_between(t, y.min(0), y.max(0), alpha=0.12)
    ax.set_xlabel("observed step")
    ax.set_ylabel("P(true goal)")
    ax.set_ylim(0, 1.02)
    ax.set_title(title or stim.name, fontsize=9)
    ax.legend(fontsize=7, frameon=False)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)


def save(path_stem, title, table, stats, extra=None):
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    path_stem.with_suffix(".json").write_text(json.dumps({"mean_P_true_goal": stats, **(extra or {})}, indent=1))
    path_stem.with_suffix(".md").write_text(f"# {title}\n\n{table}\n")
