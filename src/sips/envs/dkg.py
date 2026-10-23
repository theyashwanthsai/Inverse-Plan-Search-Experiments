"""Doors, Keys & Gems (DKG).

Map legend:
    #  wall          .  floor        S  agent start
    a-z  key of that colour        A-Z  locked door of that colour (key 'r' opens door 'R')
    1-9  gem (names given separately, e.g. {"1": "yellow"})

State is a plain tuple, for speed and hashability:
    (row, col, held, locked, keys, gems)
    held   - tuple: number of keys held, per colour
    locked - bitmask over doors, bit i set = door i still locked
    keys   - bitmask over keys, bit i set = key i still on the map
    gems   - bitmask over gems, bit i set = gem i still on the map (cleared = acquired)

Keys are single-use: unlocking a door consumes one key of the door's colour. Because a key
spent on the wrong door is gone, an agent can make its goal permanently unreachable.
"""
from collections import deque

import numpy as np

from .base import Domain

MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
UNLOCKS = {"unlock_" + d: delta for d, delta in MOVES.items()}


class DKG(Domain):
    def __init__(self, rows: list[str], gem_names: dict[str, str], name: str = "dkg"):
        self.name = name
        self.rows = rows
        self.height, self.width = len(rows), len(rows[0])
        assert all(len(r) == self.width for r in rows), "map rows must have equal length"
        self.wall = np.zeros((self.height, self.width), dtype=bool)
        self.colors: list[str] = sorted({ch.lower() for row in rows for ch in row if ch.isalpha() and ch != "S"})
        self.doors: list[tuple[int, int, int]] = []  # (row, col, colour index)
        self.keys: list[tuple[int, int, int]] = []
        self.gems: list[tuple[int, int]] = []
        gem_chars = []
        start = None
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.wall[r, c] = True
                elif ch == "S":
                    start = (r, c)
                elif ch.isupper():
                    self.doors.append((r, c, self.colors.index(ch.lower())))
                elif ch.islower():
                    self.keys.append((r, c, self.colors.index(ch)))
                elif ch.isdigit():
                    gem_chars.append((ch, r, c))
        assert start is not None, "map needs a start cell S"
        gem_chars.sort()
        self.gems = [(r, c) for _, r, c in gem_chars]
        self.goal_names = [gem_names[ch] for ch, _, _ in gem_chars]
        self.door_at = {(r, c): i for i, (r, c, _) in enumerate(self.doors)}
        self.key_at = {(r, c): i for i, (r, c, _) in enumerate(self.keys)}
        self.gem_at = {(r, c): i for i, (r, c) in enumerate(self.gems)}
        self.initial_state = (
            start[0], start[1],
            tuple(0 for _ in self.colors),
            (1 << len(self.doors)) - 1,
            (1 << len(self.keys)) - 1,
            (1 << len(self.gems)) - 1,
        )
        self._maze_dist = [self._bfs_ignoring_doors(r, c) for r, c in self.gems]

    # ---------------- dynamics ----------------
    def _blocked(self, r: int, c: int, locked: int) -> bool:
        if not (0 <= r < self.height and 0 <= c < self.width) or self.wall[r, c]:
            return True
        d = self.door_at.get((r, c))
        return d is not None and (locked >> d) & 1 == 1

    def legal_actions(self, s) -> list[str]:
        r, c, held, locked, keys, gems = s
        acts = []
        for name, (dr, dc) in MOVES.items():
            if not self._blocked(r + dr, c + dc, locked):
                acts.append(name)
        k = self.key_at.get((r, c))
        g = self.gem_at.get((r, c))
        if (k is not None and (keys >> k) & 1) or (g is not None and (gems >> g) & 1):
            acts.append("pickup")
        for name, (dr, dc) in UNLOCKS.items():
            d = self.door_at.get((r + dr, c + dc))
            if d is not None and (locked >> d) & 1 and held[self.doors[d][2]] > 0:
                acts.append(name)
        return acts

    def step(self, s, a: str):
        r, c, held, locked, keys, gems = s
        if a in MOVES:
            dr, dc = MOVES[a]
            assert not self._blocked(r + dr, c + dc, locked), f"illegal move {a}"
            return (r + dr, c + dc, held, locked, keys, gems)
        if a == "pickup":
            k = self.key_at.get((r, c))
            if k is not None and (keys >> k) & 1:
                color = self.keys[k][2]
                held = held[:color] + (held[color] + 1,) + held[color + 1:]
                return (r, c, held, locked, keys & ~(1 << k), gems)
            g = self.gem_at.get((r, c))
            assert g is not None and (gems >> g) & 1, "nothing to pick up"
            return (r, c, held, locked, keys, gems & ~(1 << g))
        if a in UNLOCKS:
            dr, dc = UNLOCKS[a]
            d = self.door_at[(r + dr, c + dc)]
            color = self.doors[d][2]
            assert (locked >> d) & 1 and held[color] > 0, "cannot unlock"
            held = held[:color] + (held[color] - 1,) + held[color + 1:]
            return (r, c, held, locked & ~(1 << d), keys, gems)
        raise ValueError(f"unknown action {a}")

    def is_goal(self, s, g: int) -> bool:
        return not (s[5] >> g) & 1

    def sample_state(self, rng):
        """Uniform over the product space: free cell x door bits x key bits x held counts x gem bits."""
        while True:
            r, c = int(rng.integers(self.height)), int(rng.integers(self.width))
            locked = int(rng.integers(1 << len(self.doors)))
            if self._blocked(r, c, locked):
                continue
            n_per_color = [sum(1 for k in self.keys if k[2] == ci) for ci in range(len(self.colors))]
            held = tuple(int(rng.integers(n + 1)) for n in n_per_color)
            keys = int(rng.integers(1 << len(self.keys)))
            gems = int(rng.integers(1 << len(self.gems)))
            return (r, c, held, locked, keys, gems)

    def state_space_size(self) -> int:
        free = int((~self.wall).sum())
        n_per_color = [sum(1 for k in self.keys if k[2] == ci) for ci in range(len(self.colors))]
        held = int(np.prod([n + 1 for n in n_per_color])) if n_per_color else 1
        return free * 2 ** len(self.doors) * 2 ** len(self.keys) * held * 2 ** len(self.gems)

    # ---------------- heuristics ----------------
    def _bfs_ignoring_doors(self, r0: int, c0: int) -> np.ndarray:
        dist = np.full((self.height, self.width), np.inf)
        dist[r0, c0] = 0
        queue = deque([(r0, c0)])
        while queue:
            r, c = queue.popleft()
            for dr, dc in MOVES.values():
                rr, cc = r + dr, c + dc
                if 0 <= rr < self.height and 0 <= cc < self.width and not self.wall[rr, cc] and dist[rr, cc] == np.inf:
                    dist[rr, cc] = dist[r, c] + 1
                    queue.append((rr, cc))
        return dist

    def h_manhattan(self, s, g: int) -> float:
        """Manhattan distance to the gem, +1 for the pickup. Ignores walls, doors and keys."""
        if self.is_goal(s, g):
            return 0.0
        gr, gc = self.gems[g]
        return abs(s[0] - gr) + abs(s[1] - gc) + 1.0

    def h_maze(self, s, g: int) -> float:
        """Shortest walking distance to the gem through walls but ignoring doors, +1 for the pickup.
        Admissible (doors only add cost), so plain A* with it is optimal."""
        if self.is_goal(s, g):
            return 0.0
        return float(self._maze_dist[g][s[0], s[1]]) + 1.0

    def heuristics(self):
        return {"manhattan": self.h_manhattan, "maze": self.h_maze}

    # ---------------- rendering ----------------
    def render_text(self, s) -> str:
        r0, c0, held, locked, keys, gems = s
        grid = [list(row.replace("S", ".")) for row in self.rows]
        for i, (r, c, _) in enumerate(self.doors):
            if not (locked >> i) & 1:
                grid[r][c] = "."
        for i, (r, c, _) in enumerate(self.keys):
            if not (keys >> i) & 1:
                grid[r][c] = "."
        for i, (r, c) in enumerate(self.gems):
            if not (gems >> i) & 1:
                grid[r][c] = "."
        grid[r0][c0] = "@"
        inv = ", ".join(f"{n}x{self.colors[ci]}" for ci, n in enumerate(held) if n)
        return "\n".join("".join(row) for row in grid) + f"\nheld keys: [{inv}]"


COLOR_RGB = {"r": "#d62728", "b": "#1f77b4", "y": "#e6b800", "g": "#2ca02c", "p": "#9467bd", "o": "#ff7f0e"}
GEM_RGB = {"red": "#d62728", "blue": "#1f77b4", "yellow": "#e6b800", "green": "#2ca02c", "purple": "#9467bd"}


def draw(ax, env: DKG, states=None, title: str = ""):
    """Draw the map and optionally a trajectory (list of states) on a matplotlib axis."""
    from matplotlib.patches import Rectangle, RegularPolygon

    ax.set_xlim(-0.5, env.width - 0.5)
    ax.set_ylim(env.height - 0.5, -0.5)
    ax.set_aspect("equal")
    ax.set_xticks([])
    ax.set_yticks([])
    for r in range(env.height):
        for c in range(env.width):
            if env.wall[r, c]:
                ax.add_patch(Rectangle((c - 0.5, r - 0.5), 1, 1, color="#444444"))
    for r, c, ci in env.doors:
        ax.add_patch(Rectangle((c - 0.45, r - 0.45), 0.9, 0.9, facecolor=COLOR_RGB.get(env.colors[ci], "k"), alpha=0.35, edgecolor="k"))
    for r, c, ci in env.keys:
        ax.plot(c, r, marker="$⚷$", markersize=14, color=COLOR_RGB.get(env.colors[ci], "k"))
    for i, (r, c) in enumerate(env.gems):
        ax.add_patch(RegularPolygon((c, r), 4, radius=0.35, color=GEM_RGB.get(env.goal_names[i], "k")))
    sr, sc = env.initial_state[:2]
    ax.plot(sc, sr, "ko", markersize=8)
    if states:
        ys = [s[0] for s in states]
        xs = [s[1] for s in states]
        jitter = np.linspace(-0.15, 0.15, len(states))  # makes back-and-forth visible
        ax.plot(np.array(xs) + jitter, np.array(ys) + jitter, "-", color="k", lw=1.2, alpha=0.7)
    ax.set_title(title, fontsize=9)
