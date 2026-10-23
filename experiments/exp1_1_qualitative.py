"""Experiment 1.1 (qualitative): goal inference from a backtracking and a failed trajectory.

    uv run python experiments/exp1_1_qualitative.py

No human data: the paper's correlations with humans (SIPS r=0.89, BIRL r=0.51) are NOT reproduced.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from sips.agent import AgentParams
from sips.birl import BIRL, async_vi, exact_vi, oracle_sampler
from sips.envs.dkg import GEM_RGB, draw
from sips.envs.dkg_maps import backtrack_map, failure_map
from sips.smc import SIPS, SIPSConfig

OUT = Path(__file__).resolve().parents[1] / "results"
SEEDS = 20

# Hand-scripted stimuli in the spirit of thesis Fig 1.1 (the paper's were also hand-designed).
SCRIPTS = {
    "backtrack": (["right"] * 2 + ["pickup"] + ["right"] + ["unlock_right"] + ["right"] * 4  # near key, door 1, to door 2
                  + ["left"] * 12 + ["pickup"]  # stuck at door 2: walk back for the far key
                  + ["right"] * 12 + ["unlock_right"] + ["right"] * 2 + ["pickup"]),
    "failure": (["right"] * 2 + ["pickup"] + ["right"] + ["unlock_right"] + ["right"] * 2  # key spent on blue door 1
                + ["left"] * 6),  # stuck: blue is now unreachable; the agent wanders back
}
MAPS = {"backtrack": backtrack_map, "failure": failure_map}


def states_from_script(env, actions):
    s = env.initial_state
    states = [s]
    for a in actions:
        assert a in env.legal_actions(s), (a, env.render_text(s))
        s = env.step(s, a)
        states.append(s)
    return states


def run_methods(env, states):
    params = AgentParams()
    res = {}
    for name, cfg in [("SIPS (no rejuv.)", SIPSConfig(n_particles=30, resample_threshold=0.25)),
                      ("SIPS (+ rejuv.)", SIPSConfig(n_particles=30, resample_threshold=0.25, rejuvenate=True))]:
        res[name] = np.array([SIPS(env, params, cfg, np.random.default_rng([seed, 11])).run(states) for seed in range(SEEDS)])
    # Near-exact reference for the same model: many particles, 3 seeds.
    big = SIPSConfig(n_particles=3000, resample_threshold=0.25)
    res["SIPS (k=3000 ref.)"] = np.array([SIPS(env, params, big, np.random.default_rng([seed, 13])).run(states) for seed in range(3)])
    res["BIRL (oracle)"] = np.array([BIRL(async_vi(env, oracle_sampler(env, [states]), 10_000, np.random.default_rng([seed, 12]))).run(states)
                                     for seed in range(SEEDS)])
    res["BIRL (exact VI)"] = BIRL(exact_vi(env)).run(states)[None]
    return res


def main():
    summary = {}
    for mname, make in MAPS.items():
        env = make()
        states = states_from_script(env, SCRIPTS[mname])
        res = run_methods(env, states)
        T = len(states) - 1
        fig, axes = plt.subplots(1, 1 + len(res), figsize=(4 + 3.2 * len(res), 3.0),
                                 gridspec_kw={"width_ratios": [1.6] + [1] * len(res)})
        draw(axes[0], env, states, f"{mname} map, blue-seeking trajectory (T={T})")
        for ax, (meth, post) in zip(axes[1:], res.items()):
            for g, gname in enumerate(env.goal_names):
                m, sd = post[:, :, g].mean(0), post[:, :, g].std(0)
                ax.plot(m, color=GEM_RGB[gname], label=gname)
                ax.fill_between(np.arange(T + 1), m - sd, m + sd, color=GEM_RGB[gname], alpha=0.15)
            ax.set_ylim(0, 1); ax.set_xlabel("timestep"); ax.set_title(meth, fontsize=9)
            for t, a in enumerate(SCRIPTS[mname]):
                if a == "pickup" or a.startswith("unlock"):
                    ax.axvline(t + 1, color="gray", lw=0.6, ls=":")
        axes[1].set_ylabel("P(goal | observations)")
        axes[-1].legend(fontsize=7)
        fig.tight_layout()
        fig.savefig(OUT / "figures" / f"exp1_1_{mname}.png", dpi=140)
        summary[mname] = {meth: {"mean_posterior_over_time": post.mean(0).round(3).tolist()} for meth, post in res.items()}
        blue = env.goal_names.index("blue")
        print(f"\n{mname}: mean P(blue) at selected timesteps (dotted lines in plot = pickups/unlocks)")
        ts = sorted({0, 3, 7, 9, T // 2, (3 * T) // 4, T})
        print(f"{'t':>18s}" + "".join(f"{t:>6d}" for t in ts))
        for meth, post in res.items():
            print(f"{meth:>18s}" + "".join(f"{post[:, t, blue].mean():6.2f}" for t in ts))
        yellow = env.goal_names.index("yellow")
        print(f"{'P(yellow) SIPS':>18s}" + "".join(f"{res['SIPS (no rejuv.)'][:, t, yellow].mean():6.2f}" for t in ts))
        print(f"{'P(yellow) BIRL-or':>18s}" + "".join(f"{res['BIRL (oracle)'][:, t, yellow].mean():6.2f}" for t in ts))
    (OUT / "tables" / "exp1_1.json").write_text(json.dumps(summary))


if __name__ == "__main__":
    main()
