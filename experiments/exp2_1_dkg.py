"""Experiment 2.1 (qualitative): Doors, Keys & Gems with plan and action mistakes.

    uv run python experiments/exp2_1_dkg.py

16 stimuli in 4 sets (optimal, irreversible plan failure, action mistakes, short-sighted
backtracking). Models: the full plan+action model (PA), each level lesioned, and a Boltzmann
agent. No human data, so there is no correlation with people; we report mean P(true goal) per
set. Parameters are not fit (no human data).
"""
import numpy as np

from common.paths import FIGURES, TABLES
from mistakes.compare import evaluate, model_zoo
from mistakes.report import curves, kind_table, save
from mistakes.stimuli import dkg_stimuli

RUNS, PER_GOAL = 5, 100  # 300 particles per run (the thesis: 10 runs x 300)
KINDS = ["optimal", "failure", "action", "backtrack"]


def main():
    stimuli, mp = dkg_stimuli()
    zoo = model_zoo(mp, PER_GOAL, plain_domain_of=lambda s: s.domain, levels="PA")
    post = evaluate(stimuli, zoo, runs=RUNS)
    table, stats = kind_table(stimuli, post, KINDS)
    print(table)

    # the thesis's key example: the agent wastes a key. Change in P(true goal) when the goal becomes unreachable.
    fail = [i for i, s in enumerate(stimuli) if s.kind == "failure"]
    drops = {}
    for m in post:
        deltas = []
        for i in fail:
            s = stimuli[i]
            alive = s.domain.states_that_can_reach(s.g)
            t = next(k for k, x in enumerate(s.states) if x not in alive)
            deltas.append(float(np.mean([r[t, s.g] - r[t - 1, s.g] for r in post[m][i]])))
        drops[m] = float(np.mean(deltas))
    print("\nmean change in P(true goal) at the step the goal becomes unreachable:",
          {m: round(v, 3) for m, v in drops.items()})

    FIGURES.mkdir(parents=True, exist_ok=True)
    for i in (fail[0], [i for i, s in enumerate(stimuli) if s.kind == "backtrack"][0]):
        curves(stimuli[i], {m: post[m][i] for m in post}, FIGURES / f"exp2_1_{stimuli[i].kind}.png")
    save(TABLES / "exp2_1", "Experiment 2.1: DKG, mean P(true goal) per stimulus set", table, stats,
         {"change_in_P_true_at_failure": drops, "stimuli": [s.name for s in stimuli]})


if __name__ == "__main__":
    main()
