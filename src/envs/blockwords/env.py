"""Block Words: Blocks World with lettered blocks; each goal is a tower spelling a word top-to-bottom,
i.e. on(w[0], w[1]), on(w[1], w[2]), ... (adapted from Ramirez & Geffner). 7 blocks give
37,633 table arrangements + 7 x 4,051 holding states = 65,990 states.

State: (towers, held) where towers is a sorted tuple of bottom-to-top tuples and held is a letter
or "" (hand empty).
"""
from functools import lru_cache

from ..base import Domain

LETTERS = tuple("acdeprw")
WORDS = ["draw", "ward", "pace", "cape", "wear"]


def _canon(towers):
    return tuple(sorted(t for t in towers if t))


class BlockWords(Domain):
    default_heuristic = "hadd"

    def __init__(self, init_towers, words=WORDS, name="blockwords", letters=LETTERS):
        self.name = name
        self.letters = tuple(letters)
        self.words = list(words)
        self.goal_names = list(words)
        self.initial_state = (_canon(tuple(tuple(t) for t in init_towers)), "")
        self.goal_facts = [frozenset(("on", w[i], w[i + 1]) for i in range(len(w) - 1)) for w in words]
        self._h_cache: dict = {}

    # ---------------- dynamics ----------------
    def legal_actions(self, s):
        towers, held = s
        acts = []
        if held == "":
            for t in towers:
                acts.append(f"pick-up {t[-1]}" if len(t) == 1 else f"unstack {t[-1]} {t[-2]}")
        else:
            acts.append(f"put-down {held}")
            acts += [f"stack {held} {t[-1]}" for t in towers]
        return acts

    def step(self, s, a):
        towers, held = s
        op, *args = a.split()
        if op in ("pick-up", "unstack"):
            x = args[0]
            new = [t[:-1] if t[-1] == x else t for t in towers]
            return (_canon(new), x)
        if op == "put-down":
            return (_canon(list(towers) + [(held,)]), "")
        if op == "stack":
            y = args[1]
            return (_canon([t + (held,) if t[-1] == y else t for t in towers]), "")
        raise ValueError(a)

    def is_goal(self, s, g):
        return self.goal_facts[g] <= facts_of(s)

    def sample_state(self, rng):
        states = all_states(self.letters)
        return states[int(rng.integers(len(states)))]

    # ---------------- heuristics ----------------
    def _relaxed(self, s, g, combine):
        key = (s, g, combine)
        if key not in self._h_cache:
            self._h_cache[key] = relaxed_cost(facts_of(s), self.goal_facts[g], combine, _ground_actions(self.letters))
        return self._h_cache[key]

    def h_add(self, s, g):
        """Additive relaxed-plan heuristic (Bonet & Geffner). Informative but inadmissible."""
        return self._relaxed(s, g, "add")

    def h_max(self, s, g):
        """Max relaxed heuristic: admissible, used for optimal A*."""
        return self._relaxed(s, g, "max")

    def h_goal_count(self, s, g):
        """Number of unsatisfied goal facts: the naive heuristic of the robustness experiment."""
        return float(len(self.goal_facts[g] - facts_of(s)))

    def heuristics(self):
        return {"hadd": self.h_add, "hmax": self.h_max, "goalcount": self.h_goal_count}


@lru_cache(maxsize=None)
def facts_of(s) -> frozenset:
    towers, held = s
    f = set()
    for t in towers:
        f.add(("ontable", t[0]))
        f.add(("clear", t[-1]))
        f.update(("on", t[i + 1], t[i]) for i in range(len(t) - 1))
    f.add(("holding", held) if held else ("handempty",))
    return frozenset(f)


@lru_cache(maxsize=None)
def _ground_actions(letters):
    acts = []  # (preconditions, add effects) of the delete relaxation
    for x in letters:
        acts.append(({("clear", x), ("ontable", x), ("handempty",)}, {("holding", x)}))
        acts.append(({("holding", x)}, {("ontable", x), ("clear", x), ("handempty",)}))
        for y in letters:
            if x != y:
                acts.append(({("holding", x), ("clear", y)}, {("on", x, y), ("clear", x), ("handempty",)}))
                acts.append(({("on", x, y), ("clear", x), ("handempty",)}, {("holding", x), ("clear", y)}))
    return [(tuple(p), tuple(a)) for p, a in acts]


ACTIONS = _ground_actions(LETTERS)


def relaxed_cost(state_facts, goal, combine: str, actions=None) -> float:
    """Fixpoint of fact costs under the delete relaxation; h = sum (h_add) or max (h_max) over goal facts."""
    cost = dict.fromkeys(state_facts, 0.0)
    agg = sum if combine == "add" else max
    changed = True
    while changed:
        changed = False
        for pre, add in (ACTIONS if actions is None else actions):
            if all(p in cost for p in pre):
                c = agg(cost[p] for p in pre) + 1.0
                for f in add:
                    if c < cost.get(f, float("inf")):
                        cost[f] = c
                        changed = True
    if not all(f in cost for f in goal):
        return float("inf")
    return float(agg(cost[f] for f in goal)) if goal else 0.0


@lru_cache(maxsize=None)
def all_states(letters) -> list:
    """Every Block Words state over these letters (all are mutually reachable)."""
    d = BlockWords([[x] for x in letters], words=[], letters=letters)
    return sorted(d.reachable_states(), key=repr)


# Three fixed initial configurations (bottom-to-top), none of which satisfies a goal.
INITIAL_CONFIGS = [
    [["r", "a"], ["d", "e", "c"], ["p", "w"]],
    [["a", "w"], ["c", "r", "e"], ["d"], ["p"]],
    [["e", "d", "a"], ["w", "p"], ["r", "c"]],
]


def blockwords_instances():
    return [BlockWords(cfg, name=f"blockwords-{i}") for i, cfg in enumerate(INITIAL_CONFIGS)]
