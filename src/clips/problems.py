"""Maps and goal-assistance problems for multi-agent Doors, Keys & Gems.

The thesis's 30 problems and maps are not published in the text, so these are our own, built to
cover its problem types (Table 4.2): ambiguous and partial instructions, assistant-only and joint
instructions, low and high goal uncertainty, and key vs door assistance.

In every problem the human acts while the robot waits, then the human speaks once, as in the
thesis's mDKG problems.
"""
from dataclasses import dataclass

from .env import MDKG

GEMS = {"1": "red", "2": "yellow", "3": "blue", "4": "green"}

MAPS = {
    # Two red keys; the red gem's plan uses key1 (left), the blue gem's uses key2 (right).
    # The yellow gem needs no help.
    "red_keys": [
        "#########",
        "#1#2#3#4#",
        "#R#.#R#B#",
        "#...@...#",
        "#.#####.#",
        "#r..&.rb#",
        "#########",
    ],
    # The red gem is behind two red doors, the yellow gem behind a red and a blue door,
    # the blue gem behind a yellow door; the green gem needs no help.
    "two_keys": [
        "#########",
        "#1#2#3#4#",
        "#R#B#Y#.#",
        "#R#R#.#.#",
        "#...@...#",
        "#.#####.#",
        "#r.r&.by#",
        "#########",
    ],
}


def make_map(name: str) -> MDKG:
    return MDKG(MAPS[name], GEMS, name)


@dataclass(frozen=True)
class Problem:
    name: str
    map: str
    category: str  # problem type, after thesis Table 4.2
    human_actions: tuple[str, ...]
    utterance: str
    true_goal: str
    true_cost: int  # index into planner.COST_FNS
    options: str  # "keys" (robot should pick up keys) or "doors" (robot should unlock doors)
    # The literal listener only parses assistant-directed commands, so a joint instruction is
    # replaced by its assistant-directed part for that baseline (thesis App. 4.8.1).
    literal_utterance: str | None = None


PROBLEMS = [
    # ---- red_keys ----
    Problem("rk_ambiguous_left", "red_keys", "ambiguous / assistant / low uncertainty",
            ("left", "left"), "Can you get me the red key?", "red", 2, "keys"),
    Problem("rk_ambiguous_right", "red_keys", "ambiguous / assistant / low uncertainty",
            ("right",), "Can you get me the red key?", "blue", 2, "keys"),
    Problem("rk_no_actions", "red_keys", "ambiguous / assistant / high uncertainty",
            (), "Could you bring me a red key?", "red", 2, "keys"),
    Problem("rk_door_blue", "red_keys", "ambiguous / assistant / low uncertainty",
            ("right", "right", "right"), "Can you open the blue door?", "green", 0, "doors"),
    Problem("rk_that_door", "red_keys", "ambiguous / assistant / low uncertainty",
            ("left", "left", "left"), "Please unlock that door for me.", "red", 0, "doors"),
    # ---- two_keys ----
    Problem("tk_partial_red", "two_keys", "partial / assistant / low uncertainty",
            ("left", "left", "left"), "Can you get me a red key?", "red", 2, "keys"),
    Problem("tk_partial_door", "two_keys", "partial / assistant / high uncertainty",
            ("left",), "Can you unlock the red door?", "yellow", 0, "doors"),
    Problem("tk_red_doors", "two_keys", "ambiguous / assistant / high uncertainty",
            (), "Please open the red doors.", "red", 0, "doors"),
    Problem("tk_joint", "two_keys", "partial / joint / low uncertainty",
            ("left", "left", "left", "down"), "I'm getting a red key, can you get the blue key?", "yellow", 2, "keys",
            literal_utterance="Can you get the blue key?"),
    Problem("tk_yellow_door", "two_keys", "ambiguous / assistant / low uncertainty",
            ("right",), "Could you unlock the yellow door?", "blue", 1, "doors"),
]


def observed_states(env: MDKG, p: Problem) -> list:
    """States while the human acts and the robot waits. The human speaks in the last state
    (the robot's turn, or the start state if the human has not acted)."""
    s = env.initial_state
    states: list = [s]
    for i, a in enumerate(p.human_actions):
        s = env.step(s, a)
        states.append(s)
        if i < len(p.human_actions) - 1:
            s = env.step(s, "wait")
            states.append(s)
    return states
