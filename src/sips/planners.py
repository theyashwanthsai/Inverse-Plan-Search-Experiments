"""Planners: optimal A*, the agent's probabilistic budgeted A*, and an exact enumerator of the
latter's output distribution (used only to check SIPS against exact inference on tiny maps)."""
import heapq
import itertools
from dataclasses import dataclass

import numpy as np

from envs.base import Domain, Heuristic, State


@dataclass(frozen=True)
class Plan:
    """A (possibly partial) plan: states[0] --actions[0]--> states[1] ... --> states[-1]."""
    states: tuple
    actions: tuple

    def __len__(self) -> int:
        return len(self.actions)


def _path(parent: dict, last: State) -> Plan:
    states, actions = [last], []
    while parent[last] is not None:
        last, a = parent[last]
        states.append(last)
        actions.append(a)
    return Plan(tuple(reversed(states)), tuple(reversed(actions)))


def astar(domain: Domain, s0: State, g: int, h: Heuristic, max_expansions: int = 10**7):
    """Plain A* with unit action costs. Returns (plan, n_expanded); plan is None if unreachable."""
    tie = itertools.count()
    open_heap = [(h(s0, g), next(tie), s0)]
    gcost = {s0: 0}
    parent: dict = {s0: None}
    closed = set()
    n = 0
    while open_heap and n < max_expansions:
        _, _, s = heapq.heappop(open_heap)
        if s in closed:
            continue
        closed.add(s)
        n += 1
        if domain.is_goal(s, g):
            return _path(parent, s), n
        for a, t in domain.successors(s):
            new_g = gcost[s] + 1
            if t not in closed and new_g < gcost.get(t, np.inf):
                gcost[t] = new_g
                parent[t] = (s, a)
                heapq.heappush(open_heap, (new_g + h(t, g), next(tie), t))
    return None, n


class _Search:
    """State of one probabilistic A* run, kept explicit so the exact enumerator can branch on it."""

    def __init__(self, domain, s0, g, h):
        self.domain, self.g, self.h = domain, g, h
        self.frontier = {s0: 0}  # state -> cost so far
        self.parent: dict = {s0: None}
        self.closed = set()
        self.last = None  # last expanded state
        self.done = False  # goal expanded or frontier exhausted

    def copy(self):
        new = object.__new__(_Search)
        new.domain, new.g, new.h = self.domain, self.g, self.h
        new.frontier, new.parent, new.closed = dict(self.frontier), dict(self.parent), set(self.closed)
        new.last, new.done = self.last, self.done
        return new

    def expansion_probs(self, gamma: float):
        """P(expand s) proportional to exp(-f(s)/gamma), f = cost so far + heuristic."""
        nodes = list(self.frontier)
        f = np.array([self.frontier[s] + self.h(s, self.g) for s in nodes])
        logits = -(f - f.min()) / gamma
        p = np.exp(logits)
        return nodes, p / p.sum()

    def expand(self, s):
        cost = self.frontier.pop(s)
        self.closed.add(s)
        self.last = s
        if self.domain.is_goal(s, self.g):
            self.done = True
            return
        for a, t in self.domain.successors(s):
            if t in self.closed:
                continue
            if cost + 1 < self.frontier.get(t, np.inf):
                self.frontier[t] = cost + 1
                self.parent[t] = (s, a)
        if not self.frontier:
            self.done = True

    def plan(self, s0) -> Plan:
        return Plan((s0,), ()) if self.last is None else _path(self.parent, self.last)


def prob_astar(domain: Domain, s0: State, g: int, h: Heuristic, gamma: float, budget: int, rng):
    """Probabilistic A* with a node-expansion budget.
    Returns (partial plan to the last expanded state, number of expansions)."""
    search = _Search(domain, s0, g, h)
    n = 0
    while n < budget and not search.done:
        nodes, p = search.expansion_probs(gamma)
        search.expand(nodes[rng.choice(len(nodes), p=p)])
        n += 1
    return search.plan(s0), n


def enumerate_prob_astar(domain: Domain, s0: State, g: int, h: Heuristic, gamma: float, budget_pmf: np.ndarray) -> dict:
    """Exact distribution over plans returned by prob_astar when budget ~ budget_pmf
    (budget_pmf[b] = P(budget = b), finite support). Exponential cost: tiny problems only."""
    tail = np.concatenate([np.cumsum(budget_pmf[::-1])[::-1], [0.0]])  # tail[b] = P(budget >= b)
    out: dict = {}

    def add(plan, p):
        out[plan] = out.get(plan, 0.0) + p

    def recurse(search: _Search, n: int, p_path: float):
        # The search has made n expansions. It stops here if the budget is exactly n, or (for any
        # budget >= n) if it is already done.
        if search.done:
            add(search.plan(s0), p_path * tail[n])
            return
        if n < len(budget_pmf):
            add(search.plan(s0), p_path * budget_pmf[n])
        if n + 1 >= len(tail) or tail[n + 1] == 0:
            return
        nodes, probs = search.expansion_probs(gamma)
        for s, q in zip(nodes, probs):
            if q < 1e-300:
                continue
            child = search.copy()
            child.expand(s)
            recurse(child, n + 1, p_path * q)

    recurse(_Search(domain, s0, g, h), 0, 1.0)
    return out
