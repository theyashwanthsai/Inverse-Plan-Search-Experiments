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
from pathlib import Path

import numpy as np

from sips.agent import AgentParams
from sips.birl import exact_vi
from sips.evaluate import Problem, agent_trajectories, run_sips
from sips.planners import prob_astar
from sips.smc import SIPSConfig

OUT = Path(__file__).resolve().parents[1] / "results"
PER_GOAL = {"dkg": 2, "blockwords": 2}  # x goals x instances = 30 trajectories per condition

PAPER = {  # thesis Table 1.1b, top-1 at Q3
    "dkg": {"r=1": 0.60, "r=2 (matched)": 0.73, "r=4": 0.73, "q=0.8": 0.53, "q=0.9": 0.60,
            "gamma=0.5": 0.67, "gamma=0.02": 0.77, "h=maze": 0.90, "Boltzmann RL (alpha=50)": 0.58, "optimal": 0.80},
    "blockwords": {"r=1": 0.90, "r=2 (matched)": 0.87, "r=4": 0.90, "q=0.8": 0.70, "q=0.9": 0.83,
                   "gamma=0.5": 0.83, "gamma=0.02": 0.87, "h=goalcount": 0.43, "Boltzmann RL (alpha=50)": 0.82,
                   "optimal": 0.80},
}


def boltzmann_trajectory(domain, vt, g, alpha, rng, max_steps=100):
    s = domain.initial_state
    states = [s]
    while not domain.is_goal(s, g) and len(states) <= max_steps:
        pol = vt.policy(s, g, alpha)
        acts = list(pol)
        s = domain.step(s, acts[rng.choice(len(acts), p=np.array(list(pol.values())))])
        states.append(s)
    return states


def optimal_trajectory_random_ties(domain, g, h, rng):
    """Probabilistic A* with unlimited budget and tiny gamma: an optimal plan, ties broken at random."""
    plan, _ = prob_astar(domain, domain.initial_state, g, h, gamma=0.01, budget=10**7, rng=rng)
    return list(plan.states)


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
        from sips.envs.dkg_maps import experiment_maps
        domains, admissible = experiment_maps(), "maze"
    else:
        from sips.envs.blockwords import blockwords_instances
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
        print(f"{cname:26s} top-1@Q3 = {results[cname]:.2f}   (paper {PAPER[name].get(cname, '-')}; "
              f"n={len(problems)}, mean len {np.mean([len(p.states) - 1 for p in problems]):.1f}, rejected {rej})", flush=True)
    (OUT / "tables" / f"exp1_3_{name}.json").write_text(json.dumps({"ours": results, "paper": PAPER[name],
                                                                    "rejected": rejected}, indent=1))


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "dkg")
