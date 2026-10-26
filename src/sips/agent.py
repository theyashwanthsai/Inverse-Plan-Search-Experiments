"""The boundedly-rational incremental planner: the generative model that SIPS inverts.

Per timestep, from state s with current plan ("mind"):
  1. plan update: d ~ Categorical([replan, refine, keep]; theta_d); replan is forced when the plan
     has no action for s. If (re)planning: budget eta ~ NegBin, run probabilistic A* for eta expansions.
  2. action selection: epsilon-greedy around the plan's action.
  3. deterministic transition.
"""
from dataclasses import dataclass
from math import comb

import numpy as np

from envs.base import Domain, State
from .planners import Plan, prob_astar

REPLAN, REFINE, KEEP = 0, 1, 2


@dataclass(frozen=True)
class AgentParams:
    r: int = 2
    q: float = 0.95
    gamma: float = 0.1
    epsilon: float = 0.05
    heuristic: str = "manhattan"
    theta_d: tuple = (0.0, 0.0, 1.0)  # P(replan), P(refine), P(keep) when not forced
    budget_max: int | None = None  # truncate the budget prior (only for exact-inference tests)

    def budget_pmf(self, upto: int) -> np.ndarray:
        """P(eta = b) for b = 0..upto. eta counts 'continue' events (prob q) before the r-th
        'stop' (prob 1-q), so mean = r q / (1-q): 38 node expansions for r=2, q=0.95."""
        b = np.arange(upto + 1)
        return np.array([comb(k + self.r - 1, k) for k in b]) * (1 - self.q) ** self.r * self.q ** b

    def sample_budget(self, rng) -> int:
        if self.budget_max is None:
            return int(rng.negative_binomial(self.r, 1 - self.q))
        pmf = self.budget_pmf(self.budget_max)
        return int(rng.choice(len(pmf), p=pmf / pmf.sum()))


@dataclass(frozen=True)
class Mind:
    """The agent's current plan and how far along it is. plan.states[idx] is where it expects to be."""
    plan: Plan | None = None
    idx: int = 0

    def planned_action(self, s: State):
        if self.plan is not None and self.idx < len(self.plan) and self.plan.states[self.idx] == s:
            return self.plan.actions[self.idx]
        return None


def plan_update(domain: Domain, params: AgentParams, s: State, mind: Mind, g: int, rng):
    """Returns (new mind, replanned?, node expansions used)."""
    d = int(rng.choice(3, p=params.theta_d))
    if mind.planned_action(s) is None:
        d = REPLAN  # forced: the old plan says nothing about this state
    if d == KEEP:
        return mind, False, 0
    h = domain.heuristics()[params.heuristic]
    budget = params.sample_budget(rng)
    if d == REPLAN:
        plan, n = prob_astar(domain, s, g, h, params.gamma, budget, rng)
        return Mind(plan, 0), True, n
    # REFINE: keep the rest of the current plan and extend it from where it ends.
    rest = Plan(mind.plan.states[mind.idx:], mind.plan.actions[mind.idx:])
    ext, n = prob_astar(domain, rest.states[-1], g, h, params.gamma, budget, rng)
    return Mind(Plan(rest.states + ext.states[1:], rest.actions + ext.actions), 0), True, n


def action_probs(domain: Domain, params: AgentParams, s: State, mind: Mind, g: int) -> dict:
    """Epsilon-greedy around the planned action. With no planned action (empty plan, or goal
    already achieved) the agent acts uniformly at random."""
    legal = domain.legal_actions(s)
    a_plan = None if domain.is_goal(s, g) else mind.planned_action(s)
    if a_plan is None or len(legal) == 1:
        return {a: 1.0 / len(legal) for a in legal}
    other = params.epsilon / (len(legal) - 1)
    return {a: (1 - params.epsilon if a == a_plan else other) for a in legal}


def advance(mind: Mind, s: State, a: str) -> Mind:
    """Move along the plan if the agent did what it planned."""
    if mind.planned_action(s) == a:
        return Mind(mind.plan, mind.idx + 1)
    return mind  # a slip: the next state won't match the plan, which forces a replan


def simulate(domain: Domain, params: AgentParams, g: int, rng, s0: State | None = None,
             mind: Mind | None = None, max_steps: int = 100):
    """Generator yielding (state_before, action, state_after) one step at a time.
    Can be resumed from any (s0, mind), which is exactly what a SIPS particle stores."""
    s = domain.initial_state if s0 is None else s0
    mind = Mind() if mind is None else mind
    for _ in range(max_steps):
        if domain.is_goal(s, g):
            return
        mind, _, _ = plan_update(domain, params, s, mind, g, rng)
        probs = action_probs(domain, params, s, mind, g)
        acts = list(probs)
        a = acts[rng.choice(len(acts), p=np.array(list(probs.values())))]
        mind = advance(mind, s, a)
        s_next = domain.step(s, a)
        yield s, a, s_next
        s = s_next


def sample_trajectory(domain, params, g, rng, max_steps=100) -> list:
    """List of states s_0..s_T."""
    states = [domain.initial_state]
    for _, _, s_next in simulate(domain, params, g, rng, max_steps=max_steps):
        states.append(s_next)
    return states
