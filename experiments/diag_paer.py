"""Diagnostic (not a paper table): the thesis's "paer" example. The agent wants "pear" but pursues
the mis-spelling "paer" at steps 3-6. P(pear) just after that window, and the top word, per model.

    uv run python experiments/diag_paer.py
"""
import numpy as np

from common.paths import TABLES
from mistakes.blockwords import setup, WORDS
from mistakes.stimuli import make
from mistakes.compare import model_zoo
from envs.blockwords import BlockWords
from envs.blockwords.env import INITIAL_CONFIGS

def plain(stim):
    c = int(stim.domain.name[-1]); return BlockWords(INITIAL_CONFIGS[c], words=WORDS, name=f"bwp{c}")

lines = ["| config | model | P(pear) at step 7 | top word at step 7 | P(reap) at step 7 | final P(pear) |", "|---|---|---|---|---|---|"]
for config in range(3):
    d, mp = setup(config)
    g = WORDS.index("pear"); tgt = d.goal_names.index("paer")
    rng = np.random.default_rng(config)
    st = make("goal", d, mp, g, d.heuristics()["hmax"], rng, ge_target=tgt)
    if st is None: print(config, "no stimulus"); continue
    zoo = model_zoo(mp, 100, plain, "GPA")
    print(f"\nconfig {config}: pear, 'paer' pursued at steps 3-6; {len(st.states)-1} steps")
    for name, f in zoo.items():
        posts = np.mean([f(st, [config, r]) for r in range(3)], axis=0)
        t_end = 7  # first observation after the corrupted window
        top_w = WORDS[int(posts[t_end].argmax())]; top_f = WORDS[int(posts[-1].argmax())]
        print(f"  {name:13s} P(pear)@7={posts[t_end,g]:.2f} (top {top_w}, reap={posts[t_end,1]:.2f})   final P(pear)={posts[-1,g]:.2f} (top {top_f})")
        lines.append(f"| {config} | {name} | {posts[t_end,g]:.2f} | {top_w} | {posts[t_end,1]:.2f} | {posts[-1,g]:.2f} |")

TABLES.mkdir(parents=True, exist_ok=True)
(TABLES / "diag_paer.md").write_text("# 'paer' diagnostic (3 runs of 500 particles, one stimulus per start configuration)\n\n" + "\n".join(lines) + "\n")
