"""Exact filter for the three-level agent: forward filtering over (goal, pursued goal, plan).
Tiny domains only (same truncated-budget requirement as sips.exact)."""
from collections import defaultdict

import numpy as np

from sips.agent import Mind, action_probs, advance
from sips.planners import enumerate_prob_astar
from .agent import MistakeParams


def goal_transitions(mp: MistakeParams, domain, g: int, ge: int) -> dict:
    if ge == g:
        opts = mp.similar_to(domain, g)
        if mp.eps_g == 0 or not opts:
            return {g: 1.0}
        out = {g: 1.0 - mp.eps_g}
        for o in opts:
            out[o] = out.get(o, 0.0) + mp.eps_g / len(opts)
        return out
    return {g: mp.recover, ge: 1.0 - mp.recover}


def exact_goal_posterior(domain, mp: MistakeParams, states):
    base = mp.base
    assert base.theta_d == (0.0, 0.0, 1.0) and base.budget_max is not None
    pmf = base.budget_pmf(base.budget_max)
    pmf = pmf / pmf.sum()
    h = domain.heuristics()[base.heuristic]
    G = mp.n_candidates(domain)
    cache: dict = {}

    def plan_dist(s, ge):
        if (s, ge) not in cache:
            cache[(s, ge)] = enumerate_prob_astar(domain, s, ge, h, base.gamma, pmf)
        return cache[(s, ge)]

    belief = {(g, g, Mind()): 1.0 / G for g in range(G)}
    out = [np.full(G, 1.0 / G)]
    for s_prev, s_next in zip(states[:-1], states[1:]):
        a = domain.action_between(s_prev, s_next)
        new: dict = defaultdict(float)
        for (g, ge, mind), p in belief.items():
            for ge2, pg in goal_transitions(mp, domain, g, ge).items():
                m0 = Mind() if ge2 != ge else mind
                if m0.planned_action(s_prev) is None:
                    options = [(Mind(plan, 0), pp) for plan, pp in plan_dist(s_prev, ge2).items()]
                else:
                    options = [(m0, 1.0)]
                for m, pp in options:
                    pa = action_probs(domain, base, s_prev, m, ge2)[a]
                    new[(g, ge2, advance(m, s_prev, a))] += p * pg * pp * pa
        z = sum(new.values())
        belief = {k: v / z for k, v in new.items()}
        post = np.zeros(G)
        for (g, _, _), p in belief.items():
            post[g] += p
        out.append(post)
    return np.array(out)
