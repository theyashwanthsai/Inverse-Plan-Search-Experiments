import numpy as np

from sips.envs.blockwords import BlockWords, all_states, LETTERS, relaxed_cost, facts_of
from sips.envs.taxi import Taxi
from sips.planners import astar
from test_dkg import bfs_plan


def test_taxi_dynamics_and_optimality():
    d = Taxi((2, 2), "R")
    assert len(d.reachable_states()) == 125
    s = d.initial_state
    assert "pickup" not in d.legal_actions(s)
    for g in range(3):
        plan, _ = astar(d, s, g, d.h_manhattan)
        assert len(plan) == len(bfs_plan(d, s, g))
    # wall between (0,1) and (0,2)
    assert "east" not in d.legal_actions(((0, 1), "R"))


def test_blockwords_state_count_and_goal():
    assert len(all_states(LETTERS)) == 37633 + 7 * 4051
    d = BlockWords([["w"], ["a"], ["r"], ["d"], ["c"], ["e"], ["p"]], words=["draw"])
    s = d.initial_state
    for a in ["pick-up a", "stack a w", "pick-up r", "stack r a", "pick-up d", "stack d r"]:
        assert a in d.legal_actions(s), a
        s = d.step(s, a)
    assert d.is_goal(s, 0)


def test_blockwords_hmax_admissible_and_astar_optimal():
    d = BlockWords([["r", "a"], ["d", "e", "c"], ["p", "w"]], words=["draw", "pace"])
    for g in range(2):
        opt = len(bfs_plan(d, d.initial_state, g))
        assert d.h_max(d.initial_state, g) <= opt
        assert len(astar(d, d.initial_state, g, d.h_max)[0]) == opt
    assert relaxed_cost(facts_of(d.initial_state), frozenset(), "add") == 0
