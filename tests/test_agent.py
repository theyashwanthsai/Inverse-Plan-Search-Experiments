import numpy as np

from sips.agent import AgentParams, Mind, action_probs, sample_trajectory
from sips.behaviour import summarize
from sips.envs.dkg_maps import backtrack_map, failure_map, tiny_map


def test_budget_pmf_mean_matches_negbin():
    p = AgentParams()
    pmf = p.budget_pmf(2000)
    assert abs(pmf.sum() - 1) < 1e-9
    assert abs((np.arange(len(pmf)) * pmf).sum() - p.r * p.q / (1 - p.q)) < 1e-6
    rng = np.random.default_rng(0)
    assert abs(np.mean([p.sample_budget(rng) for _ in range(20000)]) - 38) < 1.0


def test_epsilon_greedy_probabilities():
    env = tiny_map()
    p = AgentParams(epsilon=0.1)
    s = env.initial_state
    probs = action_probs(env, p, s, Mind(), 0)  # no plan -> uniform
    assert np.allclose(list(probs.values()), 1 / len(probs))


def test_agent_is_boundedly_rational_not_optimal():
    rng = np.random.default_rng(3)
    params = AgentParams()
    bt, fl = backtrack_map(), failure_map()
    blue = 1
    s_bt = summarize(bt, {blue: [sample_trajectory(bt, params, blue, rng) for _ in range(40)]}, bt.h_maze)["blue"]
    s_fl = summarize(fl, {blue: [sample_trajectory(fl, params, blue, rng) for _ in range(40)]}, fl.h_maze)["blue"]
    assert s_bt["optimal"] < 1 and s_bt["backtracked"] > 0 and s_bt["reached"] > 0
    assert s_fl["goal_made_unreachable"] > 0, "agent should sometimes waste the key on the failure map"
    red = summarize(bt, {2: [sample_trajectory(bt, params, 2, rng) for _ in range(40)]}, bt.h_maze)["red"]
    assert red["optimal"] > 0, "easy goals should often be reached optimally"
