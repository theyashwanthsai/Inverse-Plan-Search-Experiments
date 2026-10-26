"""Stage 3 check: sample trajectories from the agent model and confirm it is boundedly rational
(sometimes backtracks, sometimes wastes keys) rather than optimal.

    uv run python checks/stage3_agent_behaviour.py
"""
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from common.paths import RESULTS
from sips.agent import AgentParams, sample_trajectory
from sips.behaviour import summarize
from envs.dkg import draw
from envs.dkg.maps import backtrack_map, failure_map
from common.seeding import make_rng

N = 200
OUT = RESULTS


def main():
    params = AgentParams()
    rng = make_rng(3)
    report = {}
    fig, axes = plt.subplots(2, 3, figsize=(12, 5))
    for row, env in enumerate([backtrack_map(), failure_map()]):
        trajs = {g: [sample_trajectory(env, params, g, rng, max_steps=100) for _ in range(N)] for g in range(env.n_goals)}
        report[env.name] = summarize(env, trajs, env.h_maze)
        blue = env.goal_names.index("blue")
        for col in range(3):
            draw(axes[row, col], env, trajs[blue][col], f"{env.name}: blue-seeking sample {col}")
    fig.tight_layout()
    fig.savefig(OUT / "figures" / "stage3_agent_samples.png", dpi=120)
    (OUT / "tables" / "stage3_agent_behaviour.json").write_text(json.dumps(report, indent=2))
    print(f"{'map':10s} {'goal':7s} {'reached':>8s} {'optimal':>8s} {'backtrk':>8s} {'unreach':>8s} {'len':>6s} {'opt':>4s}")
    for m, goals in report.items():
        for gname, st in goals.items():
            print(f"{m:10s} {gname:7s} {st['reached']:8.2f} {st['optimal']:8.2f} {st['backtracked']:8.2f} "
                  f"{st['goal_made_unreachable']:8.2f} {st['mean_len']:6.1f} {st['optimal_len']:4d}")


if __name__ == "__main__":
    main()
