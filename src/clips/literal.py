"""LLM-based literal listener baselines (thesis Sec. 4.5.2, App. 4.8.1).

A literal listener reads the instruction alone: no goal inference, no use of the human's actions.

1. Command inference. Enumerate every assistant-directed command possible in the current state
   (subsets of up to K = 3 salient robot actions, same pruning and lifting as CLIPS), score the
   utterance against each with the LLM, and normalise: P(c | u, s) under a uniform prior.
2. Literal assistance. Draw m = 10 commands by systematic sampling, and for each one plan
   (i) the cheapest joint plan that carries the command out, then (ii) the human's remaining plan
   to the true goal with the robot standing still. Metrics are averaged over the samples.
   The thesis lets "only the principal move" in (ii). We read that as the robot staying put but
   still able to hand over a key it holds: otherwise "pick up the red key" leaves the key stuck
   with the robot and the human can never finish.

A lifted command can have several groundings ("pick up a red key" with two red keys):
  naive      plans every grounding and averages the metrics over them;
  efficient  plans the grounding whose command plan is cheapest.

The thesis's literal listener uses commands that only involve the assistant; so does ours.
Its plans come from A* with a node budget; ours are exact (Dijkstra over the joint state graph).
"""
import heapq
import itertools
from collections import defaultdict
from dataclasses import dataclass

import numpy as np

from .assist import precision_recall
from .commands import EXAMPLES, lift, valid
from .env import HELD_R, HUMAN, ON_MAP, ROBOT, MDKG
from .planner import JointPlanner


def assistant_actions(env: MDKG, s) -> list[tuple[str, str, str]]:
    """Ground salient robot actions still possible from s: pick up a key on the map, hand over a
    key on the map or held by the robot, unlock a locked door for which a matching key is left."""
    keyloc, locked = s[5], s[6]
    out = []
    for k, loc in enumerate(keyloc):
        if loc == ON_MAP:
            out.append(("pickup", "robot", env.key_names[k]))
        if loc in (ON_MAP, HELD_R):
            out.append(("handover", "robot", env.key_names[k]))
    for d, (_, _, color) in enumerate(env.doors):
        if (locked >> d) & 1 and any(loc in (ON_MAP, HELD_R) and env.keys[k][2] == color for k, loc in enumerate(keyloc)):
            out.append(("unlock", "robot", env.door_names[d]))
    return out


def literal_commands(env: MDKG, s, K: int = 3) -> dict[str, list[tuple]]:
    """Lifted command -> the ground action sets it can stand for."""
    out: dict[str, list[tuple]] = defaultdict(list)
    for n in range(1, K + 1):
        for sub in itertools.combinations(assistant_actions(env, s), n):
            if valid(sub):
                out[lift(env, sub)].append(sub)
    return dict(out)


def systematic_sample(probs: np.ndarray, m: int, rng: np.random.Generator) -> np.ndarray:
    """m indices drawn by systematic sampling over the probabilities sorted in decreasing order."""
    order = np.argsort(-probs, kind="stable")
    cum = np.cumsum(probs[order])
    x = (np.arange(m) + rng.uniform()) / m
    return order[np.minimum(np.searchsorted(cum, x, side="right"), len(order) - 1)]


def plan_command(P: JointPlanner, s, ground: tuple, table: np.ndarray, robot_only: bool = False):
    """Cheapest plan from s in which every action of `ground` happens (in any order), without
    ending the game. Both agents act unless robot_only (then the human only waits).
    Returns (list of (state, action), cost), or None if impossible."""
    env = P.env
    bit = {a: 1 << i for i, a in enumerate(ground)}
    full = (1 << len(ground)) - 1
    start = (P.index[s], 0)
    dist, prev = {start: 0.0}, {}
    heap = [(0.0, start)]
    while heap:
        d, node = heapq.heappop(heap)
        if d > dist[node]:
            continue
        i, mask = node
        if mask == full:
            path = []
            while node != start:
                node, a = prev[node]
                path.append((P.states[node[0]], a))
            return path[::-1], d
        si = P.states[i]
        for e in range(P.first[i], P.first[i + 1]):
            j, a = P.dst[e], P.action[e]
            if P.gem[j] >= 0 or (robot_only and si[0] == HUMAN and a != "wait"):
                continue
            nxt = (j, mask | bit.get(env.salient(si, a), 0))
            nd = d + table[P.actor[e], P.kind[e]]
            if nd < dist.get(nxt, np.inf):
                dist[nxt], prev[nxt] = nd, (node, a)
                heapq.heappush(heap, (nd, nxt))
    return None


@dataclass
class Outcome:
    precision: float
    recall: float
    steps: int
    human_cost: float
    reached: bool


def run_plan(P: JointPlanner, s, ground: tuple | None, g: int, c: int, options: str, gold: set) -> tuple[Outcome, float]:
    """Carry out a ground command (joint plan, else robot-only), then the human finishes alone.
    Returns the outcome and the cost of the command plan (inf if it could not be carried out)."""
    table = P.cost_fns[c].table()
    path, cmd_cost = [], np.inf
    if ground:
        found = plan_command(P, s, ground, table) or plan_command(P, s, ground, table, robot_only=True)
        if found:
            path, cmd_cost = found
            s = P.env.step(*path[-1])
    for _ in range(200):  # the human finishes; the robot stays put but may hand over keys
        if s[7] >= 0:
            break
        acts, q = P.Q(s, g, c, "stay")
        if not np.isfinite(q).any():
            break
        a = acts[int(np.argmin(q))]
        path.append((s, a))
        s = P.env.step(s, a)
    verb = "pickup" if options == "keys" else "unlock"
    chosen = {sal[2] for st, a in path if st[0] == ROBOT and (sal := P.env.salient(st, a)) and sal[0] == verb}
    prec, rec = precision_recall(chosen, gold)
    human = sum(table[HUMAN, P.env.kind(a)] for st, a in path if st[0] == HUMAN)
    return Outcome(prec, rec, len(path), float(human), s[7] == g), cmd_cost


class LiteralListener:
    def __init__(self, planner: JointPlanner, scorer, K: int = 3, examples=EXAMPLES, m: int = 10, seed: int = 0):
        self.P, self.scorer, self.K, self.examples, self.m, self.seed = planner, scorer, K, examples, m, seed

    def command_posterior(self, s, utterance: str) -> tuple[list[str], np.ndarray, dict]:
        cmds = literal_commands(self.P.env, s, self.K)
        names = sorted(cmds)
        lw = self.scorer.loglik(utterance, names, self.examples)
        w = np.exp(lw - lw.max())
        return names, w / w.sum(), cmds

    def assist(self, s, utterance: str, g: int, c: int, options: str, gold: set) -> dict:
        """Mean metrics over m sampled commands, for the naive and efficient variants."""
        names, post, groundings = self.command_posterior(s, utterance)
        picks = systematic_sample(post, self.m, np.random.default_rng(self.seed))
        memo: dict = {}

        def outcome(ground):
            if ground not in memo:
                memo[ground] = run_plan(self.P, s, ground, g, c, options, gold)
            return memo[ground]

        res = {"naive": [], "efficient": []}
        for i in picks:
            runs = [outcome(gr) for gr in groundings[names[i]]]
            res["efficient"].append(min(runs, key=lambda oc: oc[1])[0])
            res["naive"].append(Outcome(*(np.mean([getattr(o, f) for o, _ in runs])
                                          for f in ("precision", "recall", "steps", "human_cost", "reached"))))
        top = int(np.argmax(post))
        return {v: {f: float(np.mean([getattr(o, f) for o in outs]))
                    for f in ("precision", "recall", "steps", "human_cost", "reached")}
                for v, outs in res.items()} | {"top_command": names[top], "top_prob": float(post[top]),
                                                "n_commands": len(names)}
