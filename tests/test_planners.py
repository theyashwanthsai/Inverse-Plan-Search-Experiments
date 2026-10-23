import numpy as np
import pytest

from sips.envs.dkg_maps import backtrack_map, failure_map, tiny_map
from sips.planners import astar, enumerate_prob_astar, prob_astar
from test_dkg import bfs_plan

MAPS = [backtrack_map(), failure_map(), tiny_map()]


@pytest.mark.parametrize("env", MAPS, ids=lambda e: e.name)
@pytest.mark.parametrize("hname", ["manhattan", "maze"])
def test_astar_is_optimal(env, hname):
    h = env.heuristics()[hname]
    for g in range(env.n_goals):
        plan, _ = astar(env, env.initial_state, g, h)
        assert len(plan) == len(bfs_plan(env, env.initial_state, g))
        assert env.is_goal(plan.states[-1], g)


@pytest.mark.parametrize("env", MAPS, ids=lambda e: e.name)
def test_large_budget_small_gamma_matches_astar(env):
    h = env.h_manhattan
    rng = np.random.default_rng(0)
    for g in range(env.n_goals):
        opt, _ = astar(env, env.initial_state, g, h)
        for _ in range(5):
            plan, _ = prob_astar(env, env.initial_state, g, h, gamma=0.01, budget=10**6, rng=rng)
            assert env.is_goal(plan.states[-1], g)
            assert len(plan) == len(opt)


def test_small_budget_gives_short_partial_plans_towards_goal():
    env = backtrack_map()
    blue = env.goal_names.index("blue")
    h = env.h_manhattan
    rng = np.random.default_rng(1)
    s0 = env.initial_state
    for _ in range(50):
        plan, n = prob_astar(env, s0, blue, h, gamma=0.1, budget=6, rng=rng)
        assert n == 6 and len(plan) < 6
        assert not env.is_goal(plan.states[-1], blue)
        assert h(plan.states[-1], blue) < h(s0, blue), "partial plan should head towards the gem"


def test_enumeration_matches_sampling():
    env = tiny_map()
    h = env.h_manhattan
    pmf = np.array([0.05, 0.15, 0.2, 0.2, 0.2, 0.2])  # budgets 0..5
    for g in range(env.n_goals):
        exact = enumerate_prob_astar(env, env.initial_state, g, h, 0.5, pmf)
        assert abs(sum(exact.values()) - 1) < 1e-12
        rng = np.random.default_rng(g)
        counts: dict = {}
        n = 20000
        for _ in range(n):
            b = rng.choice(len(pmf), p=pmf)
            plan, _ = prob_astar(env, env.initial_state, g, h, 0.5, b, rng)
            counts[plan] = counts.get(plan, 0) + 1
        tv = 0.5 * sum(abs(exact.get(p, 0) - counts.get(p, 0) / n) for p in set(exact) | set(counts))
        assert tv < 0.02, tv
