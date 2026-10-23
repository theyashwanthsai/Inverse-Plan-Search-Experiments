"""Taxi (Dietterich 2000), 5x5 grid, 125 states per instance: taxi cell (25) x passenger location
(4 landmarks or in the taxi). Goal g = passenger dropped off at landmark g; the three goals are the
three landmarks other than the passenger's start.

    +---------+
    |R: | : :G|
    | : | : : |
    | : : : : |
    | | : | : |
    |Y| : |B: |
    +---------+
"""
from .base import Domain

LANDMARKS = {"R": (0, 0), "G": (0, 4), "Y": (4, 0), "B": (4, 3)}
NAMES = list(LANDMARKS)
IN_TAXI = "taxi"
# Walls block east-west movement between these cell pairs (row, col) <-> (row, col+1).
WALLS = {((0, 1), (0, 2)), ((1, 1), (1, 2)), ((3, 0), (3, 1)), ((4, 0), (4, 1)), ((3, 2), (3, 3)), ((4, 2), (4, 3))}
MOVES = {"north": (-1, 0), "south": (1, 0), "west": (0, -1), "east": (0, 1)}


def _wall(a, b):
    return (a, b) in WALLS or (b, a) in WALLS


class Taxi(Domain):
    default_heuristic = "manhattan"

    def __init__(self, taxi_start, passenger_start: str, name="taxi"):
        self.name = name
        self.goal_names = [n for n in NAMES if n != passenger_start]
        self.initial_state = (taxi_start, passenger_start)

    def legal_actions(self, s):
        (r, c), p = s
        acts = []
        for a, (dr, dc) in MOVES.items():
            t = (r + dr, c + dc)
            if 0 <= t[0] < 5 and 0 <= t[1] < 5 and not _wall((r, c), t):
                acts.append(a)
        if p != IN_TAXI and LANDMARKS[p] == (r, c):
            acts.append("pickup")
        if p == IN_TAXI and (r, c) in LANDMARKS.values():
            acts.append("dropoff")
        return acts

    def step(self, s, a):
        (r, c), p = s
        if a in MOVES:
            dr, dc = MOVES[a]
            return ((r + dr, c + dc), p)
        if a == "pickup":
            return ((r, c), IN_TAXI)
        if a == "dropoff":
            return ((r, c), next(n for n, xy in LANDMARKS.items() if xy == (r, c)))
        raise ValueError(a)

    def is_goal(self, s, g):
        return s[1] == self.goal_names[g]

    def sample_state(self, rng):
        cell = (int(rng.integers(5)), int(rng.integers(5)))
        return (cell, (NAMES + [IN_TAXI])[int(rng.integers(5))])

    def h_manhattan(self, s, g):
        """Manhattan distance via the passenger, + pickup and dropoff. Ignores walls; admissible."""
        if self.is_goal(s, g):
            return 0.0
        (r, c), p = s
        gr, gc = LANDMARKS[self.goal_names[g]]
        if p == IN_TAXI:
            return abs(r - gr) + abs(c - gc) + 1.0
        pr, pc = LANDMARKS[p]
        return abs(r - pr) + abs(c - pc) + 1 + abs(pr - gr) + abs(pc - gc) + 1.0

    def heuristics(self):
        return {"manhattan": self.h_manhattan}


def taxi_instances():
    """One instance per passenger start landmark, with the taxi starting in a fixed interior cell."""
    starts = {"R": (2, 2), "G": (3, 1), "Y": (1, 3), "B": (2, 0)}
    return [Taxi(starts[p], p, name=f"taxi-{p}") for p in NAMES]
