"""Goal inference over an open-ended set of words.

  exact          one likelihood per word in the dictionary, every step (cost grows with |G|)
  proposal_only  N words from the bottom-up proposal, weighted by prior / proposal, ignoring the actions
  OpenEndedSIPS  proposal + top-down reweighting by the actions: guess bottom-up, filter top-down
"""
import math

import numpy as np

from .agent import PolicyModel
from .words import WordModel


def fragments(state, min_len=2) -> list:
    """Words already partly built: each tower read top to bottom (a goal word has w[0] on top)."""
    return ["".join(reversed(t)) for t in state[0] if len(t) >= min_len]


def _norm(logw):
    w = np.exp(logw - np.max(logw))
    return w / w.sum()


def exact(wm: WordModel, pm: PolicyModel, states):
    """Posterior over all words after each observation, shape [T+1, |G|]."""
    logp = np.log(wm.prior)
    out = [_norm(logp)]
    for s, s2 in zip(states[:-1], states[1:]):
        a = pm.domain.action_between(s, s2)
        logp = logp + np.array([pm.logpi(s, a, g) for g in range(len(wm.words))])
        out.append(_norm(logp))
    return np.array(out)


def proposal_only(wm: WordModel, states, n, rng):
    out = [wm.prior]
    for s in states[1:]:
        q = wm.proposal(fragments(s))
        idx = rng.choice(len(q), size=n, p=q)
        post = np.zeros(len(wm.words))
        np.add.at(post, idx, wm.prior[idx] / q[idx])
        out.append(post / post.sum())
    return np.array(out)


class OpenEndedSIPS:
    def __init__(self, wm: WordModel, pm: PolicyModel, n, rng):
        self.wm, self.pm, self.n, self.rng = wm, pm, n, rng
        self.log_evidence: list = []  # log of the mean pool weight after each observation: estimates log P(observations)

    def _history_loglik(self, states, g) -> float:
        return sum(self.pm.logpi(s, self.pm.domain.action_between(s, s2), g) for s, s2 in zip(states[:-1], states[1:]))

    def run(self, states):
        """Posterior over all words after each observation, shape [T+1, |G|]. Particles are words."""
        wm, G = self.wm, len(self.wm.words)
        # Weight scale: a particle stands for one sample whose weight averages Z_t (the marginal
        # likelihood), so n survivors and n fresh proposals are on the same scale.
        idx = self.rng.choice(G, size=self.n, p=wm.prior)  # time 0: sampled from the prior
        parts = {int(g): math.log(c) for g, c in zip(*np.unique(idx, return_counts=True))}  # word -> log weight
        out = [wm.prior]
        for t in range(1, len(states)):
            s, s2 = states[t - 1], states[t]
            a = self.pm.domain.action_between(s, s2)
            # 1. reweight the survivors by the new action
            parts = {g: lw + self.pm.logpi(s, a, g) for g, lw in parts.items()}
            # 2. bottom-up: propose n new words, importance-weighted back to the posterior
            q = wm.proposal(fragments(s2))
            new = self.rng.choice(G, size=self.n, p=q)
            pool = dict(parts)
            for g in new:
                g = int(g)
                lw = math.log(wm.prior[g]) - math.log(q[g]) + self._history_loglik(states[: t + 1], g)
                pool[g] = np.logaddexp(pool[g], lw) if g in pool else lw
            out.append(self._posterior(pool, G))
            self.log_evidence.append(float(np.logaddexp.reduce(np.array(list(pool.values()))) - math.log(2 * self.n)))
            # 3. resample back to n particles; merge identical words
            gs = list(pool)
            w = _norm(np.array([pool[g] for g in gs]))
            draw = self.rng.choice(len(gs), size=self.n, p=w)
            total = np.logaddexp.reduce(np.array([pool[g] for g in gs]))
            counts = np.bincount(draw, minlength=len(gs))
            parts = {gs[i]: total - math.log(2 * self.n) + math.log(c) for i, c in enumerate(counts) if c}  # mean weight of the 2n pool
        return np.array(out)

    @staticmethod
    def _posterior(pool, G):
        post = np.zeros(G)
        gs = list(pool)
        post[gs] = _norm(np.array([pool[g] for g in gs]))
        return post
