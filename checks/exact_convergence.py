"""Stage 4 key correctness check: SIPS converges to the exact posterior as particle count grows.

    uv run python checks/exact_convergence.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from common.paths import RESULTS
from sips.convergence import LAYERS, sips_error, make_problems

KS = [3, 6, 12, 24, 48, 96, 192, 384, 768]
SEEDS = 20
OUT = RESULTS


def main():
    env, problems = make_problems()
    results = {}
    for name, layer in LAYERS.items():
        results[name] = []
        for k in KS:
            errs = [sips_error(env, problems, layer, k, seed) for seed in range(SEEDS)]
            results[name].append((k, float(np.mean(errs)), float(np.std(errs) / np.sqrt(SEEDS))))
            print(f"{name:36s} k={k:4d}  mean TV error {np.mean(errs):.4f} +- {np.std(errs)/np.sqrt(SEEDS):.4f}", flush=True)
    (OUT / "tables" / "exact_convergence.json").write_text(json.dumps(results, indent=2))
    fig, ax = plt.subplots(figsize=(5.5, 4))
    for name, rows in results.items():
        k, m, se = map(np.array, zip(*rows))
        ax.errorbar(k, m, yerr=se, marker="o", ms=3, capsize=2, label=name)
    k = np.array(KS)
    first = results[next(iter(results))][0][1]
    ax.plot(k, first * np.sqrt(k[0] / k), "k--", lw=1, label=r"$\propto 1/\sqrt{k}$")
    ax.set_xscale("log"); ax.set_yscale("log")
    ax.set_xlabel("number of particles k"); ax.set_ylabel("mean TV distance to exact posterior")
    ax.set_title(f"SIPS vs exact inference (tiny DKG map, {len(problems)} trajectories, {SEEDS} seeds)", fontsize=8)
    ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "exact_convergence.png", dpi=150)


if __name__ == "__main__":
    main()
