"""Experiment 2.2 (qualitative): Block Words with goal, plan and action mistakes.

    uv run python experiments/exp2_2_blockwords.py

16 stimuli in 4 sets (optimal, goal mistakes, short-sighted backtracking, action mistakes) over 5
candidate words. Models: the full goal+plan+action model (GPA), each level lesioned, and a
Boltzmann agent. No human data, so no correlation with people; we report mean P(true goal) per
set. Parameters are not fit (no human data).
"""
from common.paths import FIGURES, TABLES
from envs.blockwords import BlockWords
from envs.blockwords.env import INITIAL_CONFIGS
from mistakes.blockwords import WORDS
from mistakes.compare import evaluate, model_zoo
from mistakes.report import curves, kind_table, save
from mistakes.stimuli import blockwords_stimuli

RUNS, PER_GOAL = 3, 100  # 500 particles per run (the thesis: 10 runs x 500)
KINDS = ["optimal", "goal", "backtrack", "action"]


def plain_domain(stim):
    """The same world with only the 5 candidate words, for the Boltzmann baseline."""
    c = int(stim.domain.name[-1])
    return BlockWords(INITIAL_CONFIGS[c], words=WORDS, name=f"blockwords-plain{c}")


def main():
    stimuli, mp = blockwords_stimuli()
    zoo = model_zoo(mp, PER_GOAL, plain_domain, levels="GPA")
    post = evaluate(stimuli, zoo, runs=RUNS)
    table, stats = kind_table(stimuli, post, KINDS)
    print(table)
    FIGURES.mkdir(parents=True, exist_ok=True)
    for kind in ("goal", "backtrack"):
        i = next(i for i, s in enumerate(stimuli) if s.kind == kind)
        curves(stimuli[i], {m: post[m][i] for m in post}, FIGURES / f"exp2_2_{kind}.png")
    save(TABLES / "exp2_2", "Experiment 2.2: Block Words, mean P(true goal) per stimulus set", table, stats,
         {"stimuli": [s.name for s in stimuli]})


if __name__ == "__main__":
    main()
