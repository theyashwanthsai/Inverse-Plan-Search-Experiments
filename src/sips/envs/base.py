"""The interface every domain implements, so planners, SIPS and BIRL stay domain-agnostic.

A state is any hashable value. Actions are strings. Every action costs 1.
A goal is an index into `domain.goal_names`; `domain.is_goal(s, g)` tests it.
"""
from collections import deque
from typing import Any, Callable

State = Any  # any hashable value
Heuristic = Callable[[State, int], float]


class Domain:
    goal_names: list[str]
    initial_state: State

    def legal_actions(self, s: State) -> list[str]:
        raise NotImplementedError

    def step(self, s: State, a: str) -> State:
        raise NotImplementedError

    def is_goal(self, s: State, g: int) -> bool:
        raise NotImplementedError

    def sample_state(self, rng) -> State:
        """A uniformly random state from the (unconstrained) state space. Used by unbiased BIRL."""
        raise NotImplementedError

    def heuristics(self) -> dict[str, Heuristic]:
        raise NotImplementedError

    # ---- shared helpers ----
    @property
    def n_goals(self) -> int:
        return len(self.goal_names)

    def successors(self, s: State) -> list[tuple[str, State]]:
        return [(a, self.step(s, a)) for a in self.legal_actions(s)]

    def action_between(self, s: State, s_next: State) -> str:
        """The action that takes s to s_next. Transitions are deterministic, so observing
        consecutive states identifies the action."""
        for a in self.legal_actions(s):
            if self.step(s, a) == s_next:
                return a
        raise ValueError("no legal action connects these states")

    def reachable_states(self, s0: State | None = None, stop_at_goals: bool = False) -> list[State]:
        s0 = self.initial_state if s0 is None else s0
        seen = {s0}
        queue = deque([s0])
        while queue:
            s = queue.popleft()
            if stop_at_goals and any(self.is_goal(s, g) for g in range(self.n_goals)):
                continue
            for _, t in self.successors(s):
                if t not in seen:
                    seen.add(t)
                    queue.append(t)
        return list(seen)

    def states_that_can_reach(self, g: int, s0: State | None = None) -> set:
        """All states reachable from s0 from which goal g is still achievable (backward BFS)."""
        states = self.reachable_states(s0)
        preds: dict = {s: [] for s in states}
        for s in states:
            for _, t in self.successors(s):
                preds[t].append(s)
        alive = {s for s in states if self.is_goal(s, g)}
        queue = deque(alive)
        while queue:
            t = queue.popleft()
            for s in preds[t]:
                if s not in alive:
                    alive.add(s)
                    queue.append(s)
        return alive
