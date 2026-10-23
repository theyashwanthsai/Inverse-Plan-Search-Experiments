"""Exact goal posterior for the SIPS agent model, by forward filtering over (goal, current plan).

With theta_d = keep-unless-forced, the only hidden variables are the goal and the current plan
(and where the agent is along it). Each forced replan draws a plan from P(plan | s, g), which we
enumerate exactly with `enumerate_prob_astar` under a truncated budget prior. Tiny maps only.
"""
from collections import defaultdict

import numpy as np

from .agent import AgentParams, Mind, action_probs, advance
from .planners import enumerate_prob_astar


def exact_goal_posterior(domain, params: AgentParams, states, goal_prior=None, return_belief: bool = False):
    """Goal posterior after each observation, shape [T+1, G]. With return_belief, also the final
    joint posterior over (goal, current plan)."""
    assert params.theta_d == (0.0, 0.0, 1.0), "exact filter implemented for replan-only-when-forced"
    assert params.budget_max is not None, "needs a truncated budget prior"
    pmf = params.budget_pmf(params.budget_max)
    pmf = pmf / pmf.sum()
    h = domain.heuristics()[params.heuristic]
    G = domain.n_goals
    prior = np.full(G, 1.0 / G) if goal_prior is None else np.asarray(goal_prior)
    plan_cache: dict = {}

    def plan_dist(s, g):
        if (s, g) not in plan_cache:
            plan_cache[(s, g)] = enumerate_prob_astar(domain, s, g, h, params.gamma, pmf)
        return plan_cache[(s, g)]

    belief = {(g, Mind()): prior[g] for g in range(G)}
    out = [prior.copy()]
    for s_prev, s_next in zip(states[:-1], states[1:]):
        a = domain.action_between(s_prev, s_next)
        new: dict = defaultdict(float)
        for (g, mind), p in belief.items():
            if mind.planned_action(s_prev) is None:
                options = [(Mind(plan, 0), pp) for plan, pp in plan_dist(s_prev, g).items()]
            else:
                options = [(mind, 1.0)]
            for m, pp in options:
                pa = action_probs(domain, params, s_prev, m, g)[a]
                new[(g, advance(m, s_prev, a))] += p * pp * pa
        z = sum(new.values())
        belief = {k: v / z for k, v in new.items()}
        post = np.zeros(G)
        for (g, _), p in belief.items():
            post[g] += p
        out.append(post)
    return (np.array(out), belief) if return_belief else np.array(out)
