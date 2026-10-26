"""Joint planning for CLIPS: optimal cost-to-goal V*(s) for every (goal, cost function), and the
Boltzmann joint policy built from it.

The thesis estimates V* with Real-Time Adaptive A* (budget 2^18) and says this gets "close to" V*
in real time (App. 4.8.1). Our maps are small enough to compute V* exactly: build the reachable
joint state space once, then run Dijkstra backwards from the goal states for each hypothesis.
"""
from dataclasses import dataclass

import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import dijkstra

from .env import HANDOVER, HUMAN, MOVE, PICKUP, ROBOT, UNLOCK, WAIT, MDKG


@dataclass(frozen=True)
class CostFn:
    """Action costs per agent (thesis App. 4.8.1). Handovers count as 'other' actions."""
    name: str
    human_pickup: float = 5.0
    robot_pickup: float = 1.0
    wait: float = 0.6
    human_other: float = 1.0
    robot_other: float = 1.0
    robot_unlock: float = 1.0

    def table(self) -> np.ndarray:
        t = np.zeros((2, 5))
        t[HUMAN] = [self.human_other, self.wait, self.human_pickup, self.human_other, self.human_other]
        t[ROBOT] = [self.robot_other, self.wait, self.robot_pickup, self.robot_unlock, self.robot_other]
        assert (MOVE, WAIT, PICKUP, UNLOCK, HANDOVER) == (0, 1, 2, 3, 4)
        return t


# The four cost functions of the thesis, under a uniform prior.
COST_FNS = [
    CostFn("equal"),
    CostFn("human_lazy", human_other=2.0),
    CostFn("robot_unlock_costly", robot_unlock=5.0),
    CostFn("human_lazy_robot_unlock_costly", human_other=2.0, robot_unlock=5.0),
]


class JointPlanner:
    """All states reachable from the map's start, with V* for each (goal, cost function, mode).

    mode "joint": both agents act optimally (the human's model of cooperation).
    mode "solo": the robot only waits (the simulated human's fallback when the robot is unhelpful).
    """

    def __init__(self, env: MDKG, cost_fns=COST_FNS):
        self.env, self.cost_fns = env, list(cost_fns)
        self.states, self.index, edges = env.reachable()
        e = np.array([(i, j, actor, kind) for i, j, actor, kind, _ in edges], dtype=np.int64)
        self.src, self.dst, self.actor, self.kind = e.T
        self.action = [a for *_, a in edges]
        # Outgoing edges of each state, in the order legal_actions produced them.
        order = np.argsort(self.src, kind="stable")
        assert (order == np.arange(len(order))).all(), "BFS emits edges grouped by source"
        self.first = np.searchsorted(self.src, np.arange(len(self.states) + 1))
        self.gem = np.array([s[7] for s in self.states])
        self._V: dict = {}

    @property
    def n_states(self) -> int:
        return len(self.states)

    def edge_costs(self, c: int) -> np.ndarray:
        return self.cost_fns[c].table()[self.actor, self.kind]

    def V(self, g: int, c: int, mode: str = "joint") -> np.ndarray:
        key = (g, c, mode)
        if key not in self._V:
            w = self.edge_costs(c)
            keep = np.ones(len(w), dtype=bool)
            if mode == "solo":
                keep = (self.actor == HUMAN) | (self.kind == WAIT)
            # Reverse graph: edge t -> s with the cost of the action s -> t. Action costs are > 0,
            # and no two actions join the same pair of states, so no duplicate edges get summed.
            n = self.n_states
            m = csr_matrix((w[keep], (self.dst[keep], self.src[keep])), shape=(n, n))
            goals = np.flatnonzero(self.gem == g)
            self._V[key] = dijkstra(m, directed=True, indices=goals, min_only=True)
        return self._V[key]

    def actions(self, s) -> tuple[list[str], np.ndarray, np.ndarray]:
        """(action names, successor indices, edge indices) for the agent to act at s."""
        i = self.index[s]
        lo, hi = self.first[i], self.first[i + 1]
        return self.action[lo:hi], self.dst[lo:hi], np.arange(lo, hi)

    def Q(self, s, g: int, c: int, mode: str = "joint") -> tuple[list[str], np.ndarray]:
        """Q*(s, a) = cost(a) + V*(s') for the acting agent's actions (inf = goal unreachable)."""
        acts, nxt, eidx = self.actions(s)
        q = self.edge_costs(c)[eidx] + self.V(g, c, mode)[nxt]
        if mode == "solo" and s[0] == ROBOT:
            q = np.where(np.array(acts) == "wait", q, np.inf)
        return acts, q

    def greedy_rollout(self, s, g: int, c: int, max_steps: int = 60):
        """Follow argmin Q* from s (ties: first action in legal-action order) until the goal or
        max_steps. Returns the list of (state, action) pairs."""
        out = []
        for _ in range(max_steps):
            if self.env.is_goal(s, g) or s[7] >= 0:
                break
            acts, q = self.Q(s, g, c)
            if not np.isfinite(q).any():
                break
            a = acts[int(np.argmin(q))]
            out.append((s, a))
            s = self.env.step(s, a)
        return out


def boltzmann_logprobs(q: np.ndarray, betas: np.ndarray) -> np.ndarray:
    """log pi_beta(a) for pi_beta(a) proportional to exp(-beta Q(a)), one row per beta.
    Actions with Q = inf get -inf; if every action has Q = inf the policy is uniform."""
    finite = np.isfinite(q)
    if not finite.any():
        return np.full((len(betas), len(q)), -np.log(len(q)))
    logits = -np.outer(betas, np.where(finite, q - q[finite].min(), 0.0))
    logits[:, ~finite] = -np.inf
    m = logits.max(axis=1, keepdims=True)
    return logits - (m + np.log(np.exp(logits - m).sum(axis=1, keepdims=True)))
