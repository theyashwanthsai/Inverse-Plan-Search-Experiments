"""Shared harness for the accuracy/speed (1.2) and robustness (1.3) experiments."""
import time
from dataclasses import dataclass

import numpy as np

from .agent import AgentParams, sample_trajectory
from .birl import BIRL, async_vi, exact_vi, oracle_sampler, uniform_sampler
from .planners import astar
from .smc import SIPS, SIPSConfig

QUARTILES = (0.25, 0.5, 0.75)


@dataclass
class Problem:
    domain_name: str
    domain: object
    goal: int
    states: list
    kind: str  # "optimal" / "suboptimal" / condition name


def optimal_trajectory(domain, g, admissible_h):
    plan, _ = astar(domain, domain.initial_state, g, admissible_h)
    return list(plan.states)


def agent_trajectories(domain, params: AgentParams, g, n, rng, max_steps=100, max_tries=200):
    """n trajectories from the agent model that reach the goal within max_steps (rejection sampling).
    Returns (trajectories, number of rejected samples)."""
    out, rejected = [], 0
    while len(out) < n:
        t = sample_trajectory(domain, params, g, rng, max_steps=max_steps)
        if domain.is_goal(t[-1], g):
            out.append(t)
        else:
            rejected += 1
            if rejected > max_tries:
                raise RuntimeError(f"agent rarely reaches goal {g} in {domain}")
    return out, rejected


def quartile_metrics(post: np.ndarray, g_true: int) -> dict:
    """P(g_true) and top-1 accuracy at the 1st/2nd/3rd quartile of the trajectory. Top-1 gives
    fractional credit 1/m when the true goal ties with m-1 others (so a flat posterior scores 1/|G|)."""
    T = len(post) - 1
    out = {}
    for i, q in enumerate(QUARTILES, start=1):
        p = post[max(1, int(round(q * T)))]
        tied = np.isclose(p, p.max(), atol=1e-9)
        out[f"P_Q{i}"] = float(p[g_true])
        out[f"top1_Q{i}"] = float(tied[g_true] / tied.sum())
    return out


def run_sips(problem: Problem, params: AgentParams, cfg: SIPSConfig, seed) -> dict:
    f = SIPS(problem.domain, params, cfg, np.random.default_rng(seed))
    post, c0, steps = f.run(problem.states, timed=True)
    T = len(problem.states) - 1
    return {**quartile_metrics(post, problem.goal), "C0": c0, "MC": float(np.mean(steps)),
            "AC": (c0 + sum(steps)) / T, "N": f.n_expanded}


def run_birl(problem: Problem, vt, c0: float, alpha: float = 1.0) -> dict:
    post, steps = BIRL(vt, alpha).run(problem.states, timed=True)
    T = len(problem.states) - 1
    return {**quartile_metrics(post, problem.goal), "C0": c0, "MC": float(np.mean(steps)),
            "AC": (c0 + sum(steps)) / T, "N": vt.n_backups}


def timed_vi(kind: str, domain, trajectories=None, n_iter=None, seed=0):
    t0 = time.perf_counter()
    if kind == "exact":
        vt = exact_vi(domain)
    elif kind == "unbiased":
        vt = async_vi(domain, uniform_sampler(domain), n_iter, np.random.default_rng(seed))
    elif kind == "oracle":
        vt = async_vi(domain, oracle_sampler(domain, trajectories), n_iter, np.random.default_rng(seed))
    else:
        raise ValueError(kind)
    return vt, time.perf_counter() - t0


METRIC_KEYS = ["P_Q1", "P_Q2", "P_Q3", "top1_Q1", "top1_Q2", "top1_Q3", "C0", "MC", "AC", "N"]


def aggregate(rows: list[dict]) -> dict:
    return {k: float(np.mean([r[k] for r in rows])) for k in METRIC_KEYS}


def format_table(results: dict, paper: dict | None = None) -> str:
    """Markdown table: one row per method, with the paper's numbers alongside where given."""
    head = "| Method | P(g) Q1 | Q2 | Q3 | Top-1 Q1 | Q2 | Q3 | C0 (s) | MC (s) | AC (s) | N |\n|---|---|---|---|---|---|---|---|---|---|---|\n"
    lines = []
    for meth, m in results.items():
        lines.append(f"| {meth} | {m['P_Q1']:.2f} | {m['P_Q2']:.2f} | {m['P_Q3']:.2f} | {m['top1_Q1']:.2f} | "
                     f"{m['top1_Q2']:.2f} | {m['top1_Q3']:.2f} | {m['C0']:.3g} | {m['MC']:.3g} | {m['AC']:.3g} | {m['N']:.0f} |")
        if paper and meth in paper:
            p = paper[meth]
            lines.append("| *paper* | " + " | ".join(str(x) for x in p) + " |")
    return head + "\n".join(lines)
