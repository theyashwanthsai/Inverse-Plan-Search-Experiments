"""Hand-designed DKG maps. All keys/doors share one colour ('k') unless noted, so any key opens any door.

Goal order is the gem digit order: gem '1' is goal 0, '2' is goal 1, '3' is goal 2.
"""
from .env import DKG

GEMS = {"1": "yellow", "2": "blue", "3": "red"}

# Backtracking (cf. thesis Fig 1.1a). Yellow is behind one door, blue behind that same door plus a
# second one. The optimal blue plan fetches the far-left key first. A myopic blue-seeker grabs the
# near key, opens door 1 (which looks like going for yellow), finds door 2 locked, and has to walk
# all the way back for the second key. Red needs no key.
BACKTRACK = [
    "#################",
    "######3######1###",
    "#k....S.k.K...K2#",
    "#################",
]

# Failure (cf. thesis Fig 1.1b). The near key should be spent on the bottom door, which guards two
# more keys; blue sits behind two doors. A myopic agent spends the near key on the first blue door
# (it gets closer to blue in Manhattan distance) and is then stuck: blue becomes unreachable.
# Yellow and red each sit behind one door, so picking up the near key is consistent with every gem.
FAILURE = [
    "###############",
    "#3K...S.k.K.K2#",
    "#####.#########",
    "#1K...#########",
    "#####K#########",
    "####k.k########",
    "###############",
]

# Tiny map for exact-inference checks. Yellow and red are both reached by first moving left
# (red needs the key and the door), blue is to the right.
TINY = [
    "#######",
    "#1kS.2#",
    "###K###",
    "###3###",
    "#######",
]


# Extra maps for the accuracy/speed and robustness experiments (1.2, 1.3).
# Three rooms off a corridor, one door each; three keys, one of them inside the gem-less middle room.
ROOMS = [
    "#############",
    "#1..#.k.#..2#",
    "#...#...#...#",
    "##K###.###K##",
    "#k....S....k#",
    "######.######",
    "#3..K.......#",
    "#############",
]

# Blue needs two keys and there are only two: spending one on red's door makes blue impossible.
TWOKEY = [
    "###########",
    "#2..K..K..#",
    "########..#",
    "#k...S....#",
    "#.###.###K#",
    "#.#1..#..3#",
    "#k#####...#",
    "###########",
]

# A more open maze: yellow and blue each sit behind one door, red is free but far.
OPEN = [
    "#############",
    "#3.....#...1#",
    "#.###..K....#",
    "#.#k#..###.##",
    "#...#.S..#..#",
    "###.####.#.##",
    "#2..K....#..#",
    "#######k....#",
    "#############",
]


# A larger map (6 doors, 6 keys) used only for the state-space scaling diagnostic of unbiased BIRL.
LARGE = [
    "#################",
    "#1....#...k.....#",
    "#.###.#.#######.#",
    "#.#k..K.......#.#",
    "#.#####.#####.#.#",
    "#.....#.#.3.#.K.#",
    "###K#.#.#.#.#.###",
    "#k..#...S.#K#..k#",
    "#.#####.#.#.###.#",
    "#....K..k.#...#.#",
    "#.#######.###.K.#",
    "#k........#.....#",
    "###########2#####",
    "#################",
]


def large_map() -> DKG:
    return DKG(LARGE, GEMS, name="large")


def rooms_map() -> DKG:
    return DKG(ROOMS, GEMS, name="rooms")


def twokey_map() -> DKG:
    return DKG(TWOKEY, GEMS, name="twokey")


def open_map() -> DKG:
    return DKG(OPEN, GEMS, name="open")


def experiment_maps() -> list[DKG]:
    return [backtrack_map(), failure_map(), rooms_map(), twokey_map(), open_map()]


def backtrack_map() -> DKG:
    return DKG(BACKTRACK, GEMS, name="backtrack")


def failure_map() -> DKG:
    return DKG(FAILURE, GEMS, name="failure")


def tiny_map() -> DKG:
    return DKG(TINY, GEMS, name="tiny")
