"""SIPS for the three-level agent: each particle also carries the goal the agent is pursuing now.

The pursued goal is a latent Markov chain sampled from its prior (bootstrap proposal); the action
likelihood is evaluated under it. Resampling is inherited; MH rejuvenation is not implemented for
this model.
"""
import math
from dataclasses import dataclass

import numpy as np

from envs.base import Domain
from sips.agent import Mind, action_probs, advance, plan_update
from sips.smc import SIPS, Particle, SIPSConfig, StepRecord
from .agent import MistakeParams, goal_step


@dataclass
class MParticle(Particle):
    ge: int = -1

    def copy(self) -> "MParticle":
        return MParticle(self.g, self.mind, self.logw, list(self.hist), self.ge)


class MistakeSIPS(SIPS):
    def __init__(self, domain: Domain, mp: MistakeParams, config: SIPSConfig, rng, goal_prior=None):
        assert not config.rejuvenate, "MH rejuvenation is not implemented for the mistake model"
        self.mp = mp
        self.n_obs = mp.n_candidates(domain)
        prior = np.full(self.n_obs, 1.0 / self.n_obs) if goal_prior is None else goal_prior
        super().__init__(domain, mp.base, config, rng, prior)

    def init(self, s0):
        k, G = self.cfg.n_particles, self.n_obs
        self.obs = [s0]
        if self.cfg.stratified and k % G == 0:
            goals = np.repeat(np.arange(G), k // G)
            logw = self.log_prior[goals] - np.log(1.0 / G)
        else:
            goals = self.rng.choice(G, size=k, p=np.exp(self.log_prior))
            logw = np.zeros(k)
        self.particles = [MParticle(int(g), Mind(), float(w), [], int(g)) for g, w in zip(goals, logw)]

    def _extend(self, p: MParticle, s_prev, a) -> StepRecord:
        ge = goal_step(self.mp, self.domain, p.g, p.ge, self.rng)
        if ge != p.ge:
            p.mind = Mind()
        p.ge = ge
        mind, replanned, n = plan_update(self.domain, self.params, s_prev, p.mind, ge, self.rng)
        self.n_expanded += n
        planned = mind.planned_action(s_prev)
        ll = math.log(action_probs(self.domain, self.params, s_prev, mind, ge)[a])
        rec = StepRecord(p.mind, mind, replanned, ll, planned is not None and planned != a)
        p.mind = advance(mind, s_prev, a)
        p.hist.append(rec)
        return rec

    def goal_posterior(self) -> np.ndarray:
        post = np.zeros(self.n_obs)
        for p, w in zip(self.particles, self.normalized_weights()):
            post[p.g] += w
        return post

    def pursued_posterior(self) -> np.ndarray:
        """P(goal the agent is pursuing right now) over all domain goals."""
        post = np.zeros(self.domain.n_goals)
        for p, w in zip(self.particles, self.normalized_weights()):
            post[p.ge] += w
        return post
