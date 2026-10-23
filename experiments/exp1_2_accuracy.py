"""Experiment 1.2: accuracy and speed of SIPS vs unbiased / oracle BIRL.

    uv run python experiments/exp1_2_accuracy.py dkg
    uv run python experiments/exp1_2_accuracy.py taxi
    uv run python experiments/exp1_2_accuracy.py blockwords
"""
import json
import sys
from pathlib import Path

import numpy as np

from sips.agent import AgentParams
from sips.evaluate import (Problem, aggregate, agent_trajectories, format_table, optimal_trajectory,
                           run_birl, run_sips, timed_vi)
from sips.smc import SIPSConfig

OUT = Path(__file__).resolve().parents[1] / "results"
N_SUBOPT = 2  # agent-model trajectories per goal per instance (plus 1 optimal)

# Thesis Table 1.1a: P(g) Q1-Q3, Top-1 Q1-Q3, C0, MC, AC, N
PAPER = {
    "dkg": {"SIPS": [0.37, 0.51, 0.61, 0.74, 0.74, 0.74, 3.30, 0.70, 0.86, 2099],
            "BIRL (unbiased)": [0.33, 0.33, 0.33, 0.33, 0.33, 0.33, 3326, 0.12, 154, 250000],
            "BIRL (oracle)": [0.37, 0.36, 0.42, 0.44, 0.60, 0.80, 150, 0.12, 7.01, 10000]},
    "taxi": {"SIPS": [0.44, 0.50, 0.62, 0.53, 0.56, 0.67, 13.0, 1.80, 2.55, 1429],
             "BIRL (unbiased)": [0.34, 0.35, 0.79, 0.33, 0.42, 0.92, 2.22, 0.00, 0.16, 10000],
             "BIRL (oracle)": [0.37, 0.47, 0.81, 0.42, 0.44, 0.86, 1.63, 0.00, 0.12, 2500]},
    "blockwords": {"SIPS": [0.47, 0.83, 0.90, 0.78, 0.84, 0.91, 20.8, 2.46, 4.15, 2506],
                   "BIRL (unbiased)": [0.20, 0.20, 0.21, 0.42, 0.49, 0.56, 687, 0.27, 63.6, 250000],
                   "BIRL (oracle)": [0.20, 0.29, 0.45, 0.73, 0.80, 0.96, 22.2, 0.05, 2.12, 10000]},
}


def domain_setup(name):
    """Returns (list of domain instances, admissible heuristic name, unbiased iterations, oracle iterations)."""
    if name == "dkg":
        from sips.envs.dkg_maps import experiment_maps
        return experiment_maps(), "maze", 250_000, 10_000
    if name == "taxi":
        from sips.envs.taxi import taxi_instances
        return taxi_instances(), "manhattan", 10_000, 2_500  # the paper's N for Taxi
    if name == "blockwords":
        from sips.envs.blockwords import blockwords_instances
        return blockwords_instances(), "hmax", 250_000, 10_000
    raise ValueError(name)


def make_dataset(domains, admissible, params, rng):
    problems, rejected = [], 0
    for d in domains:
        h = d.heuristics()[admissible]
        for g in range(d.n_goals):
            problems.append(Problem(d.name, d, g, optimal_trajectory(d, g, h), "optimal"))
            trajs, rej = agent_trajectories(d, params, g, N_SUBOPT, rng)
            rejected += rej
            problems += [Problem(d.name, d, g, t, "suboptimal") for t in trajs]
    return problems, rejected


def main(name):
    domains, admissible, n_unbiased, n_oracle = domain_setup(name)
    params = AgentParams(heuristic=domains[0].default_heuristic) if hasattr(domains[0], "default_heuristic") else AgentParams()
    rng = np.random.default_rng(2020)
    problems, rejected = make_dataset(domains, admissible, params, rng)
    print(f"{name}: {len(problems)} problems on {len(domains)} instances "
          f"({rejected} agent samples rejected for not reaching the goal within 100 steps)", flush=True)

    rows = {"SIPS": [], "BIRL (unbiased)": [], "BIRL (oracle)": [], "BIRL (exact VI, not in paper)": []}
    for d in domains:
        mine = [p for p in problems if p.domain is d]
        vts = {"BIRL (unbiased)": timed_vi("unbiased", d, n_iter=n_unbiased, seed=1),
               "BIRL (oracle)": timed_vi("oracle", d, [p.states for p in mine], n_iter=n_oracle, seed=2),
               "BIRL (exact VI, not in paper)": timed_vi("exact", d)}
        for i, p in enumerate(mine):
            cfg = SIPSConfig(n_particles=10 * d.n_goals, resample_threshold=0.25, rejuvenate=False)
            r = run_sips(p, params, cfg, seed=[3, i, domains.index(d)])
            rows["SIPS"].append({**r, "kind": p.kind})
            for meth, (vt, c0) in vts.items():
                rows[meth].append({**run_birl(p, vt, c0), "kind": p.kind})
        print(f"  {d.name}: done ({len(mine)} problems; unbiased VI {vts['BIRL (unbiased)'][1]:.1f}s)", flush=True)

    results = {m: aggregate(r) for m, r in rows.items()}
    by_kind = {kind: {m: aggregate([x for x in r if x["kind"] == kind]) for m, r in rows.items()}
               for kind in ("optimal", "suboptimal")}
    table = format_table(results, PAPER.get(name))
    print(table)
    for kind, res in by_kind.items():
        print(f"\n{kind} trajectories only:\n" + format_table(res))
    (OUT / "tables" / f"exp1_2_{name}.json").write_text(json.dumps({"all": results, **by_kind, "rows": rows,
                                                                  "n_problems": len(problems), "rejected": rejected}, indent=1))
    (OUT / "tables" / f"exp1_2_{name}.md").write_text(
        f"# Experiment 1.2: {name} ({len(problems)} problems)\n\n{table}\n\n"
        + "\n\n".join(f"## {k} only\n\n{format_table(v)}" for k, v in by_kind.items()) + "\n")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dkg")
