"""Multi-agent Doors, Keys & Gems (mDKG), the CLIPS domain (thesis Ch. 4).

A human (the principal) and a robot (the assistant) take turns acting. The human wants exactly
one of the gems. The robot can pick up keys, unlock doors and hand keys over, but cannot pick up
gems. Agents may share a cell and walk past each other.

Map legend:
    #  wall          .  floor        @  human start      &  robot start
    a-z  key of that colour        A-Z  locked door of that colour (key 'r' opens door 'R')
    1-9  gem (names given separately, e.g. {"1": "red"})

Keys and doors are named key1, key2, ... and door1, door2, ... in reading order (row by row),
so that commands can refer to individual objects.

State is a plain tuple:
    (turn, hr, hc, rr, rc, keyloc, locked, gem)
    turn   - 0 if the human acts next, 1 if the robot does
    keyloc - tuple, one entry per key: 0 on the map, 1 held by human, 2 held by robot, 3 used up
    locked - bitmask over doors, bit i set = door i still locked
    gem    - index of the gem the human picked up, or -1. A picked gem ends the episode.
"""
from collections import deque

import numpy as np

HUMAN, ROBOT = 0, 1
AGENTS = ("human", "robot")
MOVES = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
UNLOCKS = {"unlock_" + d: delta for d, delta in MOVES.items()}
ON_MAP, HELD_H, HELD_R, USED = 0, 1, 2, 3
COLOR_NAMES = {"r": "red", "b": "blue", "y": "yellow", "g": "green", "p": "purple", "o": "orange"}

# Action kinds, used to price actions under each cost function.
MOVE, WAIT, PICKUP, UNLOCK, HANDOVER = range(5)


class MDKG:
    def __init__(self, rows: list[str], gem_names: dict[str, str], name: str = "mdkg"):
        self.name = name
        self.rows = rows
        self.height, self.width = len(rows), len(rows[0])
        assert all(len(r) == self.width for r in rows), "map rows must have equal length"
        self.wall = np.zeros((self.height, self.width), dtype=bool)
        self.doors: list[tuple[int, int, str]] = []  # (row, col, colour letter)
        self.keys: list[tuple[int, int, str]] = []
        gem_chars = []
        human = robot = None
        for r, row in enumerate(rows):
            for c, ch in enumerate(row):
                if ch == "#":
                    self.wall[r, c] = True
                elif ch == "@":
                    human = (r, c)
                elif ch == "&":
                    robot = (r, c)
                elif ch.isupper():
                    self.doors.append((r, c, ch.lower()))
                elif ch.islower():
                    self.keys.append((r, c, ch))
                elif ch.isdigit():
                    gem_chars.append((ch, r, c))
        assert human is not None and robot is not None, "map needs @ (human) and & (robot)"
        gem_chars.sort()
        self.gems = [(r, c) for _, r, c in gem_chars]
        self.goal_names = [gem_names[ch] for ch, _, _ in gem_chars]
        self.key_names = [f"key{i + 1}" for i in range(len(self.keys))]
        self.door_names = [f"door{i + 1}" for i in range(len(self.doors))]
        self.door_at = {(r, c): i for i, (r, c, _) in enumerate(self.doors)}
        self.key_at = {(r, c): i for i, (r, c, _) in enumerate(self.keys)}
        self.gem_at = {(r, c): i for i, (r, c) in enumerate(self.gems)}
        self.initial_state = (
            HUMAN, human[0], human[1], robot[0], robot[1],
            tuple(ON_MAP for _ in self.keys),
            (1 << len(self.doors)) - 1,
            -1,
        )

    @property
    def n_goals(self) -> int:
        return len(self.goal_names)

    def color_of(self, obj: str) -> str:
        """Colour name of 'keyN' or 'doorN'."""
        kind, idx = obj.rstrip("0123456789"), int(obj.lstrip("keydor")) - 1
        letter = self.keys[idx][2] if kind == "key" else self.doors[idx][2]
        return COLOR_NAMES[letter]

    # ---------------- dynamics ----------------
    def _blocked(self, r: int, c: int, locked: int) -> bool:
        if not (0 <= r < self.height and 0 <= c < self.width) or self.wall[r, c]:
            return True
        d = self.door_at.get((r, c))
        return d is not None and (locked >> d) & 1 == 1

    @staticmethod
    def _pos(s, agent: int):
        return (s[1], s[2]) if agent == HUMAN else (s[3], s[4])

    def legal_actions(self, s) -> list[str]:
        """Actions of the agent whose turn it is. Empty once a gem has been picked up."""
        turn, _, _, _, _, keyloc, locked, gem = s
        if gem >= 0:
            return []
        r, c = self._pos(s, turn)
        held = HELD_H if turn == HUMAN else HELD_R
        acts = [name for name, (dr, dc) in MOVES.items() if not self._blocked(r + dr, c + dc, locked)]
        acts.append("wait")
        k = self.key_at.get((r, c))
        if k is not None and keyloc[k] == ON_MAP:
            acts.append("pickup")
        elif turn == HUMAN and (r, c) in self.gem_at:
            acts.append("pickup")
        for name, (dr, dc) in UNLOCKS.items():
            d = self.door_at.get((r + dr, c + dc))
            if d is not None and (locked >> d) & 1 and self._held_key(keyloc, held, self.doors[d][2]) is not None:
                acts.append(name)
        orr, occ = self._pos(s, 1 - turn)
        if abs(orr - r) + abs(occ - c) <= 1:
            acts.extend("handover_" + self.key_names[k] for k, loc in enumerate(keyloc) if loc == held)
        return acts

    def _held_key(self, keyloc, held: int, color: str):
        """Lowest-index key of this colour held by the agent (the one an unlock uses up)."""
        for k, loc in enumerate(keyloc):
            if loc == held and self.keys[k][2] == color:
                return k
        return None

    def step(self, s, a: str):
        turn, hr, hc, rr, rc, keyloc, locked, gem = s
        r, c = self._pos(s, turn)
        held = HELD_H if turn == HUMAN else HELD_R
        nxt = 1 - turn
        if a in MOVES:
            dr, dc = MOVES[a]
            assert not self._blocked(r + dr, c + dc, locked), f"illegal move {a}"
            r, c = r + dr, c + dc
        elif a == "wait":
            pass
        elif a == "pickup":
            k = self.key_at.get((r, c))
            if k is not None and keyloc[k] == ON_MAP:
                keyloc = keyloc[:k] + (held,) + keyloc[k + 1:]
            else:
                assert turn == HUMAN and (r, c) in self.gem_at, "nothing to pick up"
                gem = self.gem_at[(r, c)]
        elif a in UNLOCKS:
            dr, dc = UNLOCKS[a]
            d = self.door_at[(r + dr, c + dc)]
            k = self._held_key(keyloc, held, self.doors[d][2])
            assert (locked >> d) & 1 and k is not None, "cannot unlock"
            keyloc = keyloc[:k] + (USED,) + keyloc[k + 1:]
            locked &= ~(1 << d)
        elif a.startswith("handover_"):
            k = self.key_names.index(a[len("handover_"):])
            assert keyloc[k] == held, "agent does not hold that key"
            keyloc = keyloc[:k] + ((HELD_R if turn == HUMAN else HELD_H),) + keyloc[k + 1:]
        else:
            raise ValueError(f"unknown action {a}")
        if turn == HUMAN:
            hr, hc = r, c
        else:
            rr, rc = r, c
        return (nxt, hr, hc, rr, rc, keyloc, locked, gem)

    @staticmethod
    def kind(a: str) -> int:
        if a in MOVES:
            return MOVE
        if a == "wait":
            return WAIT
        if a == "pickup":
            return PICKUP
        if a in UNLOCKS:
            return UNLOCK
        return HANDOVER

    def salient(self, s, a: str):
        """The communicable (salient) form of an action, or None.
        Salient: picking up a key, unlocking a door, handing over a key (thesis App. 4.8.1).
        Returned as (verb, actor, object name)."""
        turn = s[0]
        actor = AGENTS[turn]
        if a == "pickup":
            k = self.key_at.get(self._pos(s, turn))
            if k is not None and s[5][k] == ON_MAP:
                return ("pickup", actor, self.key_names[k])
            return None
        if a in UNLOCKS:
            r, c = self._pos(s, turn)
            dr, dc = UNLOCKS[a]
            return ("unlock", actor, self.door_names[self.door_at[(r + dr, c + dc)]])
        if a.startswith("handover_"):
            return ("handover", actor, a[len("handover_"):])
        return None

    def is_goal(self, s, g: int) -> bool:
        return s[7] == g

    # ---------------- state space ----------------
    def reachable(self, s0=None):
        """BFS over every state reachable from s0. Returns (states, index, edges) where edges is a
        list of (src index, dst index, actor, action kind, action name)."""
        s0 = self.initial_state if s0 is None else s0
        index = {s0: 0}
        states = [s0]
        edges = []
        queue = deque([s0])
        while queue:
            s = queue.popleft()
            i = index[s]
            for a in self.legal_actions(s):
                t = self.step(s, a)
                j = index.get(t)
                if j is None:
                    j = index[t] = len(states)
                    states.append(t)
                    queue.append(t)
                edges.append((i, j, s[0], self.kind(a), a))
        return states, index, edges

    # ---------------- rendering ----------------
    def render_text(self, s) -> str:
        _, hr, hc, rr, rc, keyloc, locked, gem = s
        grid = [list(row.replace("@", ".").replace("&", ".")) for row in self.rows]
        for i, (r, c, _) in enumerate(self.doors):
            if not (locked >> i) & 1:
                grid[r][c] = "."
        for i, (r, c, _) in enumerate(self.keys):
            if keyloc[i] != ON_MAP:
                grid[r][c] = "."
        if gem >= 0:
            r, c = self.gems[gem]
            grid[r][c] = "."
        grid[rr][rc] = "&"
        grid[hr][hc] = "@" if (hr, hc) != (rr, rc) else "%"
        inv = {who: [self.key_names[k] + f"({self.keys[k][2]})" for k, loc in enumerate(keyloc) if loc == held]
               for who, held in (("human", HELD_H), ("robot", HELD_R))}
        return "\n".join("".join(row) for row in grid) + f"\nheld: {inv}"
