"""The open-ended goal space: every dictionary word these blocks can spell, a word prior, and a
character n-gram proposal."""
import math
import re
from collections import Counter
from pathlib import Path

import numpy as np

from common.paths import ROOT

LETTERS = "acdeoprst"  # 9 blocks -> 458 words (the thesis: 145-807 per scenario)
DATA = ROOT / "data" / f"words_{LETTERS}.tsv"
DICT = Path("/usr/share/dict/american-english")
TREEBANK = Path.home() / "nltk_data" / "corpora" / "treebank" / "raw"


def build_table(letters=LETTERS, lo=3, hi=7, path=DATA):
    """Dictionary words (each block used at most once) with their count in the NLTK Treebank sample
    (200 WSJ articles, about 100k tokens). Written once; later runs read the file."""
    ok = {w.strip() for w in DICT.read_text().split() if w.strip().isascii() and w.strip().isalpha() and w.strip().islower()}
    valid = sorted(w for w in ok if lo <= len(w) <= hi and set(w) <= set(letters) and len(set(w)) == len(w))
    counts = Counter()
    for f in TREEBANK.glob("wsj_*"):
        counts.update(re.findall(r"[a-z]+", f.read_text(encoding="latin-1").lower()))
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"{w}\t{counts[w]}\n" for w in valid))


def load_table(path=DATA):
    if not path.exists():
        build_table(path=path)
    rows = [l.split("\t") for l in path.read_text().splitlines()]
    return [r[0] for r in rows], np.array([int(r[1]) for r in rows])


class WordModel:
    """Prior p(w) ~ (count + 1)^tau, and a 5-gram character model q0(w) fit to the same weights."""

    def __init__(self, tau=0.5, n=5, smooth=0.1):
        self.words, counts = load_table()
        self.index = {w: i for i, w in enumerate(self.words)}
        weight = (counts + 1.0) ** tau
        self.prior = weight / weight.sum()
        self.n, self.smooth = n, smooth
        ctx = Counter()
        grams = Counter()
        for w, wt in zip(self.words, weight):
            s = "^" * (n - 1) + w + "$"
            for i in range(n - 1, len(s)):
                grams[(s[i - n + 1:i], s[i])] += wt
                ctx[s[i - n + 1:i]] += wt
        self._grams, self._ctx, self._V = grams, ctx, len(LETTERS) + 1
        q = np.array([self._logp(w) for w in self.words])
        q = np.exp(q - q.max())
        self.ngram = q / q.sum()  # normalised over the finite goal space

    def _logp(self, w):
        s = "^" * (self.n - 1) + w + "$"
        lp = 0.0
        for i in range(self.n - 1, len(s)):
            c = s[i - self.n + 1:i]
            lp += math.log((self._grams.get((c, s[i]), 0.0) + self.smooth) / (self._ctx.get(c, 0.0) + self.smooth * self._V))
        return lp

    def proposal(self, fragments, eps=0.05) -> np.ndarray:
        """q(w | what is on the table): the n-gram auto-completion, restricted (up to eps) to words
        that contain every built fragment as a substring."""
        fit = np.array([1.0 if all(f in w for f in fragments) else eps for w in self.words])
        q = self.ngram * fit
        return q / q.sum()


# Three start configurations (towers bottom-to-top) using all 9 blocks; none spells a goal word.
INITIAL_CONFIGS = [
    [["r", "t", "a"], ["d", "e", "c"], ["p", "o"], ["s"]],
    [["a", "s"], ["c", "r", "e"], ["d", "t"], ["p", "o"]],
    [["e", "d", "a"], ["s", "p"], ["r", "c", "o"], ["t"]],
]
assert all(sorted(x for t in c for x in t) == sorted(LETTERS) for c in INITIAL_CONFIGS)
