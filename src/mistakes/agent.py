"""Chapter 2 agent: mistakes at three levels.

  goal   with prob eps_g per step a correct goal is swapped for a similar one; a corrupted goal
         returns to the true one with prob `recover` per step. A change of goal discards the plan.
  plan   the Chapter 1 budget-limited planner (`base`, no refinement: theta_d keeps the plan).
  action with prob base.epsilon a random other legal action replaces the planned one.

Goal indices below are domain goal indices. The first `n_obs` are the candidates an observer
chooses between; any further goals exist only as targets of corruption (e.g. "paer").
"""
from dataclasses import dataclass, replace

import numpy as np

from envs.base import Domain
from sips.agent import AgentParams, Mind, action_probs, advance, plan_update

TINY = 1e-4  # stands in for "never" when a level is lesioned, so likelihoods never hit exactly 0


@dataclass(frozen=True)
class MistakeParams:
    base: AgentParams = AgentParams(epsilon=0.05)  # r, q, gamma, heuristic; epsilon is eps_a
    eps_g: float = 0.0
    recover: float = 0.3
    similar: tuple = ()  # similar[g] = tuple of goal indices g can be corrupted into
    n_obs: int | None = None  # number of candidate goals; None = all domain goals

    def n_candidates(self, domain: Domain) -> int:
        return domain.n_goals if self.n_obs is None else self.n_obs

    def similar_to(self, domain: Domain, g: int) -> tuple:
        if self.similar:
            return self.similar[g]
        return tuple(h for h in range(self.n_candidates(domain)) if h != g)


def lesion(mp: MistakeParams, level: str) -> MistakeParams:
    """Switch one level off. 'G': no goal mistakes. 'P': near-optimal planning (huge budget, tiny
    gamma). 'A': (almost) no action slips."""
    if level == "G":
        return replace(mp, eps_g=0.0)
    if level == "P":
        return replace(mp, base=replace(mp.base, q=0.9999, gamma=0.01))
    if level == "A":
        return replace(mp, base=replace(mp.base, epsilon=TINY))
    raise ValueError(level)


def goal_step(mp: MistakeParams, domain: Domain, g: int, ge: int, rng) -> int:
    """One transition of the (possibly corrupted) goal the agent is currently pursuing."""
    if ge == g:
        if mp.eps_g > 0 and rng.random() < mp.eps_g:
            options = mp.similar_to(domain, g)
            if options:
                return int(options[rng.integers(len(options))])
        return g
    return g if rng.random() < mp.recover else ge


def simulate(domain: Domain, mp: MistakeParams, g: int, rng, max_steps: int = 100, ge_script=None):
    """Yields (state_before, action, state_after, pursued_goal). `ge_script(t)` may force the
    pursued goal at step t (used to build stimuli); it returns None to leave it to the model."""
    s, mind, ge = domain.initial_state, Mind(), g
    for t in range(max_steps):
        if domain.is_goal(s, g):
            return
        new_ge = goal_step(mp, domain, g, ge, rng)
        if ge_script is not None and ge_script(t) is not None:
            new_ge = ge_script(t)
        if new_ge != ge:
            mind = Mind()
        ge = new_ge
        mind, _, _ = plan_update(domain, mp.base, s, mind, ge, rng)
        probs = action_probs(domain, mp.base, s, mind, ge)
        acts = list(probs)
        a = acts[rng.choice(len(acts), p=np.array(list(probs.values())))]
        mind = advance(mind, s, a)
        s_next = domain.step(s, a)
        yield s, a, s_next, ge
        s = s_next


def sample_trajectory(domain, mp, g, rng, max_steps=100, ge_script=None):
    """(states s_0..s_T, pursued goal at each step)."""
    states, goals = [domain.initial_state], []
    for _, _, s_next, ge in simulate(domain, mp, g, rng, max_steps, ge_script):
        states.append(s_next)
        goals.append(ge)
    return states, goals
