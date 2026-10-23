import numpy as np

from sips.birl import BIRL, async_vi, exact_vi, oracle_sampler, uniform_sampler
from sips.envs.dkg_maps import failure_map, tiny_map
from sips.planners import astar


def _max_err(env, vt, ref, states):
    return max(abs(vt.V[g].get(s, 0) - ref.V[g].get(s, 0)) for g in range(env.n_goals) for s in states)


def test_exact_vi_satisfies_bellman_equation():
    env = tiny_map()
    vt = exact_vi(env)
    for s in env.reachable_states():
        for g in range(env.n_goals):
            if not env.is_goal(s, g):
                assert abs(vt.V[g][s] - vt.q_values(s, g)[1].max()) < 1e-9


def test_async_variants_approach_exact_vi_on_tiny_map():
    env = tiny_map()
    ref = exact_vi(env)
    reach = env.reachable_states()
    rng = np.random.default_rng(0)
    errs = [_max_err(env, async_vi(env, uniform_sampler(env), n, rng), ref, reach) for n in (500, 5000, 30000)]
    assert errs[0] > errs[1] > errs[2] and errs[2] < 1e-6, errs
    trajs = [astar(env, env.initial_state, g, env.h_maze)[0].states for g in range(env.n_goals)]
    orc = async_vi(env, oracle_sampler(env, trajs), 5000, rng)
    # The oracle is only expected to be right along each trajectory, for that trajectory's goal.
    err = max(abs(orc.V[g].get(s, 0) - ref.V[g].get(s, 0)) for g, t in enumerate(trajs) for s in t)
    assert err < 1e-6


def test_unbiased_birl_unconverged_and_flat_on_large_map():
    env = failure_map()
    blue = env.goal_names.index("blue")
    traj = list(astar(env, env.initial_state, blue, env.h_maze)[0].states)
    ref = exact_vi(env)
    vt = async_vi(env, uniform_sampler(env), 250_000, np.random.default_rng(0))
    s0 = env.initial_state
    assert ref.V[blue][s0] > 0.01 and vt.V[blue].get(s0, 0.0) == 0.0, "value never propagates back to the start"
    post = BIRL(vt).run(traj)
    assert np.abs(post[int(0.75 * (len(traj) - 1))] - 1 / 3).max() < 0.05, "near-flat posterior"
    assert BIRL(ref).run(traj)[-2][blue] > 0.4, "while converged VI is informative"
