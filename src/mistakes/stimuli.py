"""Stimulus sets for Experiments 2.1 / 2.2: trajectories with a known kind of mistake.

Each stimulus is (domain, true goal, states, kind). Kinds:
  optimal   A* path with random tie-breaking
  backtrack the agent's own short-sighted planning, nothing else switched on; kept only if it
            reaches the goal in >= 3 extra steps
  failure   as backtrack, but kept only if it makes the goal unreachable (DKG: a wasted key)
  action    an optimal path with one or two random slips, re-planned optimally after each slip
  goal      the agent pursues a mis-spelled goal for a few steps, then goes back to the true one
"""
from dataclasses import dataclass, replace

import numpy as np

from sips.planners import astar, prob_astar
from .agent import sample_trajectory

MAX_STEPS = 40
MIN_LEN = 6  # shorter trajectories cannot be ambiguous
FAIL_TAIL = 8  # steps kept after the goal became unreachable


@dataclass
class Stimulus:
    domain: object
    g: int
    states: list
    kind: str

    @property
    def name(self) -> str:
        return f"{self.domain.name}/{self.domain.goal_names[self.g]}/{self.kind}"


def _opt_len(domain, g, h) -> int:
    return len(astar(domain, domain.initial_state, g, h)[0])


def optimal(domain, g, h, rng) -> list:
    plan, _ = prob_astar(domain, domain.initial_state, g, h, gamma=0.01, budget=10**7, rng=rng)
    return list(plan.states)


def with_slips(domain, g, h, rng, n_slips=1) -> list:
    """Optimal path, except at n_slips random steps a different legal action is taken."""
    L = _opt_len(domain, g, h)
    slip_at = set(rng.choice(np.arange(1, max(2, L - 1)), size=min(n_slips, max(1, L - 2)), replace=False).tolist())
    s, states, plan_acts, t = domain.initial_state, [domain.initial_state], None, 0
    while not domain.is_goal(s, g) and t < 4 * L:
        if t in slip_at:
            planned = None if not plan_acts else plan_acts[0]
            options = [a for a in domain.legal_actions(s) if a != planned]
            a = options[int(rng.integers(len(options)))]
            plan_acts = None
        else:
            if not plan_acts:
                plan = astar(domain, s, g, h)[0]
                if plan is None:  # a slip made the goal unreachable
                    break
                plan_acts = list(plan.actions)
            a = plan_acts.pop(0)
        s = domain.step(s, a)
        states.append(s)
        t += 1
    return states


def make(kind, domain, mp, g, h, rng, tries=300, ge_target=None, window=(3, 7)):
    """One stimulus of the given kind, or None if no sample qualified."""
    gen = replace(mp, eps_g=0.0, base=replace(mp.base, epsilon=0.02))
    L = _opt_len(domain, g, h)
    if kind == "optimal":
        return Stimulus(domain, g, optimal(domain, g, h, rng), kind)
    if kind == "action":
        return Stimulus(domain, g, with_slips(domain, g, h, rng, n_slips=int(rng.integers(1, 3))), kind)
    if kind == "goal":
        a, b = window
        script = lambda t: ge_target if a <= t < b else g
        for _ in range(tries):
            states, _ = sample_trajectory(domain, replace(gen, base=replace(gen.base, q=0.995)), g, rng,
                                          max_steps=MAX_STEPS, ge_script=script)
            if domain.is_goal(states[-1], g) and len(states) - 1 > L + 2:
                return Stimulus(domain, g, states, kind)
        return None
    if kind in ("backtrack", "failure"):
        alive = domain.states_that_can_reach(g) if kind == "failure" else None
        for _ in range(tries):
            states, _ = sample_trajectory(domain, gen, g, rng, max_steps=MAX_STEPS)
            if kind == "backtrack" and domain.is_goal(states[-1], g) and len(states) - 1 >= L + 3:
                return Stimulus(domain, g, states, kind)
            if kind == "failure" and any(x not in alive for x in states):
                t_fail = next(i for i, x in enumerate(states) if x not in alive)
                return Stimulus(domain, g, states[: t_fail + FAIL_TAIL + 1], kind)
        return None
    raise ValueError(kind)


def build_set(kinds, combos, make_one, per_kind=4, seed=0):
    """Take the first `per_kind` of a random ordering of combos (domain, g, ...) that yield a stimulus of each kind."""
    rng = np.random.default_rng(seed)
    out = []
    for kind in kinds:
        got = 0
        for ci in np.tile(rng.permutation(len(combos)), 3):  # a combo may be used again, with a new sample
            st = make_one(kind, combos[ci], rng)
            if st is not None and len(st.states) > MIN_LEN:
                out.append(st)
                got += 1
            if got == per_kind:
                break
        assert got == per_kind, f"only {got} stimuli of kind {kind}"
    return out


def dkg_stimuli(seed=0):
    from envs.dkg.maps import experiment_maps
    from .agent import MistakeParams
    from sips.agent import AgentParams
    maps = experiment_maps()
    mp = MistakeParams(base=AgentParams(epsilon=0.05, heuristic="manhattan"))
    combos = [(m, g) for g in range(3) for m in maps]
    def one(kind, c, rng):
        d, g = c
        return make(kind, d, mp, g, d.heuristics()["maze"], rng)
    return build_set(["optimal", "failure", "action", "backtrack"], combos, one, seed=seed), mp


def blockwords_stimuli(seed=0):
    from .blockwords import setup
    setups = [setup(c) for c in range(3)]
    combos = [(setups[c], g) for g in range(5) for c in range(3)]
    def one(kind, c, rng):
        (d, mp), g = c
        kw = {}
        if kind == "goal":
            kw["ge_target"] = int(rng.choice(mp.similar[g]))
        return make(kind, d, mp, g, d.heuristics()["hmax"], rng, **kw)
    return build_set(["optimal", "goal", "backtrack", "action"], combos, one, seed=seed), setups[0][1]
