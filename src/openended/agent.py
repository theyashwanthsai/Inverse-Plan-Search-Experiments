"""Policy-based planner of Chapter 3: a Boltzmann policy over lookahead-corrected heuristic values.

    pi(a | s, g) ~ exp(-beta * (1 + V_d(step(s, a), g)))
    V_0(s, g) = h(s, g);  V_d(s, g) = 0 at a goal, else min_a 1 + V_{d-1}(step(s, a), g)

`depth` d is the planning budget: more lookahead, better-informed policy. States and actions are
observed directly, so no plan is ever inferred; the likelihood is closed form.
"""
import math

import numpy as np


class PolicyModel:
    def __init__(self, domain, beta=2.0, depth=1, heuristic="hadd"):
        self.domain, self.beta, self.depth = domain, beta, depth
        self.h = domain.heuristics()[heuristic]
        self._v: dict = {}
        self._logpi: dict = {}

    def V(self, s, g, d):
        key = (s, g, d)
        if key not in self._v:
            if self.domain.is_goal(s, g):
                v = 0.0
            elif d == 0:
                v = self.h(s, g)
            else:
                v = min(1.0 + self.V(t, g, d - 1) for _, t in self.domain.successors(s))
            self._v[key] = v
        return self._v[key]

    def policy(self, s, g) -> dict:
        acts, scores = [], []
        for a, t in self.domain.successors(s):
            acts.append(a)
            scores.append(-self.beta * (1.0 + self.V(t, g, self.depth)))
        z = np.array(scores)
        z = np.exp(z - z.max())
        return dict(zip(acts, z / z.sum()))

    def logpi(self, s, a, g) -> float:
        key = (s, a, g)
        if key not in self._logpi:
            self._logpi[key] = math.log(max(self.policy(s, g)[a], 1e-300))
        return self._logpi[key]


def sample_trajectory(domain, model: PolicyModel, g, rng, max_steps=30):
    s, states = domain.initial_state, [domain.initial_state]
    while not domain.is_goal(s, g) and len(states) <= max_steps:
        pol = model.policy(s, g)
        acts = list(pol)
        s = domain.step(s, acts[rng.choice(len(acts), p=np.array(list(pol.values())))])
        states.append(s)
    return states
