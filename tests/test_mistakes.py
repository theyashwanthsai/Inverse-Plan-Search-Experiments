"""The three-level agent and its inference."""
import numpy as np

from envs.dkg.maps import tiny_map
from mistakes.agent import MistakeParams, lesion, sample_trajectory
from mistakes.exact import exact_goal_posterior as exact_m
from mistakes.smc import MistakeSIPS
from sips.agent import AgentParams
from sips.convergence import TEST_PARAMS
from sips.smc import SIPS, SIPSConfig

MP = MistakeParams(base=TEST_PARAMS, eps_g=0.15, recover=0.4)


def _trajs(env, mp, n=2, seed=0, max_steps=10):
    rng = np.random.default_rng(seed)
    return [sample_trajectory(env, mp, g, rng, max_steps=max_steps)[0] for g in range(env.n_goals) for _ in range(n)]


def test_no_goal_mistakes_reproduces_chapter1_sips_exactly():
    """eps_g = 0 draws no extra random numbers, so the same seed gives the identical posterior."""
    env = tiny_map()
    mp = MistakeParams(base=TEST_PARAMS)
    for i, traj in enumerate(_trajs(env, mp, n=1)):
        cfg = SIPSConfig(n_particles=30)
        a = SIPS(env, TEST_PARAMS, cfg, np.random.default_rng(i)).run(traj)
        b = MistakeSIPS(env, mp, cfg, np.random.default_rng(i)).run(traj)
        np.testing.assert_allclose(a, b)


def test_goal_corruption_happens_and_is_corrected():
    env = tiny_map()
    rng = np.random.default_rng(0)
    pursued = np.concatenate([sample_trajectory(env, MP, 0, rng, max_steps=30)[1] for _ in range(60)])
    assert (pursued != 0).any() and (pursued == 0).mean() > 0.6  # corrupted sometimes, mostly correct


def test_lesions_switch_off_one_level():
    assert lesion(MP, "G").eps_g == 0
    assert lesion(MP, "A").base.epsilon < 1e-3 and lesion(MP, "A").eps_g == MP.eps_g
    assert lesion(MP, "P").base.q > MP.base.q and lesion(MP, "P").eps_g == MP.eps_g


def test_mistake_sips_converges_to_exact_filter():
    env = tiny_map()
    trajs = _trajs(env, MP, n=1, seed=3)
    exact = [exact_m(env, MP, t) for t in trajs]

    def err(k, seed):
        return np.mean([0.5 * np.abs(MistakeSIPS(env, MP, SIPSConfig(n_particles=k), np.random.default_rng([seed, i, k]))
                                     .run(t)[1:] - e[1:]).sum(axis=1).mean()
                        for i, (t, e) in enumerate(zip(trajs, exact))])
    small = np.mean([err(6, s) for s in range(6)])
    large = np.mean([err(600, s) for s in range(6)])
    assert large < 0.05 and large < small / 2.5, (small, large)
