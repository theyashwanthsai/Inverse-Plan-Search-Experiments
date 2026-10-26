"""Experiment 1.3: robustness of SIPS to a mismatched agent model.

    uv run python experiments/exp1_3_robustness.py dkg
    uv run python experiments/exp1_3_robustness.py blockwords

SIPS always assumes r=2, q=0.95, gamma=0.1 and the default heuristic (Manhattan in DKG, h_add in
Block Words). Test data comes from agents that differ. Metric: top-1 accuracy at the third quartile.
The paper's human-data column (5 pilot subjects) is NOT reproduced.
"""
import json
import sys
from dataclasses import replace

import numpy as np

from sips.agent import AgentParams
from birl.birl import exact_vi
from common.paths import TABLES
from evaluation.datasets import boltzmann_trajectory, optimal_trajectory_random_ties
from evaluation.evaluate import Problem, agent_trajectories, run_sips
from evaluation.paper import PAPER_1_3
from sips.smc import SIPSConfig


PER_GOAL = {"dkg": 2, "blockwords": 2}  # x goals x instances = 30 trajectories per condition


def conditions(name, base: AgentParams):
    alt_h = "maze" if name == "dkg" else "goalcount"
    return {
        "r=1": replace(base, r=1), "r=2 (matched)": base, "r=4": replace(base, r=4),
        "q=0.8": replace(base, q=0.8), "q=0.9": replace(base, q=0.9),
        "gamma=0.5": replace(base, gamma=0.5), "gamma=0.02": replace(base, gamma=0.02),
        f"h={alt_h}": replace(base, heuristic=alt_h),
        "Boltzmann RL (alpha=50)": "boltzmann", "optimal": "optimal",
    }


def main(name):
    if name == "dkg":
        from envs.dkg.maps import experiment_maps
        domains, admissible = experiment_maps(), "maze"
    else:
        from envs.blockwords import blockwords_instances
        domains, admissible = blockwords_instances(), "hmax"
    base = AgentParams(heuristic=getattr(domains[0], "default_heuristic", "manhattan"))
    vts = {}
    results, rejected = {}, {}
    for ci, (cname, cond) in enumerate(conditions(name, base).items()):
        rng = np.random.default_rng([1003, ci])
        problems, rej = [], 0
        for d in domains:
            for g in range(d.n_goals):
                if cond == "boltzmann":
                    if d.name not in vts:
                        vts[d.name] = exact_vi(d)
                    trajs = []
                    while len(trajs) < PER_GOAL[name]:
                        t = boltzmann_trajectory(d, vts[d.name], g, 50.0, rng)
                        if d.is_goal(t[-1], g):
                            trajs.append(t)
                        else:
                            rej += 1
                elif cond == "optimal":
                    trajs = [optimal_trajectory_random_ties(d, g, d.heuristics()[admissible], rng) for _ in range(PER_GOAL[name])]
                else:
                    trajs, r = agent_trajectories(d, cond, g, PER_GOAL[name], rng, max_tries=2000)
                    rej += r
                problems += [Problem(d.name, d, g, t, cname) for t in trajs]
        accs = []
        for i, p in enumerate(problems):
            cfg = SIPSConfig(n_particles=10 * p.domain.n_goals, resample_threshold=0.25)
            accs.append(run_sips(p, base, cfg, seed=[7, ci, i])["top1_Q3"])
        results[cname] = float(np.mean(accs))
        rejected[cname] = rej
        print(f"{cname:26s} top-1@Q3 = {results[cname]:.2f}   (paper {PAPER_1_3[name].get(cname, '-')}; "
              f"n={len(problems)}, mean len {np.mean([len(p.states) - 1 for p in problems]):.1f}, rejected {rej})", flush=True)
    (TABLES / f"exp1_3_{name}.json").write_text(json.dumps({"ours": results, "paper": PAPER_1_3[name],
                                                                    "rejected": rejected}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dkg")
