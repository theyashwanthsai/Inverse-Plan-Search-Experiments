"""Experiment 3.2: open-ended goal inference over words, accuracy and cost (cf. thesis Table 3.1).

    uv run python experiments/exp3_2_engine.py

Goal space: all 458 dictionary words (3-7 letters, each block used once) that 9 blocks can spell.
Test data: 30 trajectories (3 start configurations x 10 target words drawn from the word prior),
sampled from the policy-based agent that the observer assumes (so this flatters every method).
Methods: exact enumeration over all words; the bottom-up proposal alone; open-ended SIPS (N particles).
Metric: P(true word), averaged over the observed steps and problems; time per observed action,
including the planner's heuristic evaluations (each run starts from a cold cache).
Human similarity (Exp 3.1) needs human data and is not reproduced.
"""
import time

import numpy as np

from common.paths import TABLES
from envs.blockwords import BlockWords
from openended import inference as inf
from openended.agent import PolicyModel, sample_trajectory
from openended.words import INITIAL_CONFIGS, LETTERS, WordModel
from evaluation.stats import mean_se

BETA, DEPTH = 2.0, 0  # the agent's (and the observer's) policy: Boltzmann over the heuristic
NS = [2, 5, 10, 20, 50]
SEEDS = 3
PER_CONFIG = 10


def make_problems(wm, rng):
    problems = []
    for c, cfg in enumerate(INITIAL_CONFIGS):
        d = BlockWords(cfg, words=wm.words, letters=LETTERS)
        pm = PolicyModel(d, BETA, DEPTH)
        targets = rng.choice(len(wm.words), size=PER_CONFIG, replace=False, p=wm.prior)
        for g in targets:
            for _ in range(50):
                tr = sample_trajectory(d, pm, int(g), rng)
                if d.is_goal(tr[-1], int(g)) and len(tr) > 4:
                    problems.append((c, int(g), tr))
                    break
    return problems


def fresh(c, wm):
    d = BlockWords(INITIAL_CONFIGS[c], words=wm.words, letters=LETTERS)
    return d, PolicyModel(d, BETA, DEPTH)


def timed(fn, c, wm):
    d, pm = fresh(c, wm)
    t0 = time.perf_counter()
    post = fn(d, pm)
    return post, (time.perf_counter() - t0)


def main():
    wm = WordModel()
    problems = make_problems(wm, np.random.default_rng(0))
    print(f"{len(wm.words)} words, {len(problems)} problems, mean length {np.mean([len(t) - 1 for _, _, t in problems]):.1f}", flush=True)
    rows = {}

    def record(name, fn):
        ps, ts = [], []
        for i, (c, g, tr) in enumerate(problems):
            for s in range(SEEDS if name != "exact" else 1):
                post, dt = timed(lambda d, pm: fn(d, pm, tr, np.random.default_rng([i, s])), c, wm)
                ps.append(post[1:, g].mean())
                ts.append(dt / (len(tr) - 1))
        rows[name] = {"P_true": mean_se(ps), "sec_per_action": float(np.mean(ts))}
        print(f"{name:28s} P(true)={rows[name]['P_true'][0]:.3f} +- {rows[name]['P_true'][1]:.3f}   {rows[name]['sec_per_action']:.3f} s/action", flush=True)

    record("exact", lambda d, pm, tr, rng: inf.exact(wm, pm, tr))
    for n in NS:
        record(f"proposal only, N={n}", lambda d, pm, tr, rng, n=n: inf.proposal_only(wm, tr, n, rng))
    for n in NS:
        record(f"open-ended SIPS, N={n}", lambda d, pm, tr, rng, n=n: inf.OpenEndedSIPS(wm, pm, n, rng).run(tr))

    lines = ["| method | P(true word) | s / action |", "|---|---|---|"]
    for k, v in rows.items():
        lines.append(f"| {k} | {v['P_true'][0]:.3f} ± {v['P_true'][1]:.3f} | {v['sec_per_action']:.3f} |")
    TABLES.mkdir(parents=True, exist_ok=True)
    (TABLES / "exp3_2.md").write_text("# Experiment 3.2: open-ended goal inference over 458 words\n\n" + "\n".join(lines) + "\n")
    import json
    (TABLES / "exp3_2.json").write_text(json.dumps(rows, indent=1))


if __name__ == "__main__":
    main()
