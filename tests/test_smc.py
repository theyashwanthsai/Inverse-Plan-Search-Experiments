"""Correctness of SIPS against exact inference on a tiny map."""
from collections import Counter, defaultdict

import numpy as np

from sips.convergence import LAYERS, TEST_PARAMS, sips_error, make_problems
from sips.exact import exact_goal_posterior
from sips.smc import SIPS, SIPSConfig


def test_sips_error_shrinks_with_particles_for_every_layer():
    env, problems = make_problems(n_per_goal=1)
    for name, layer in LAYERS.items():
        small = np.mean([sips_error(env, problems, layer, 6, s) for s in range(6)])
        large = np.mean([sips_error(env, problems, layer, 300, s) for s in range(6)])
        assert large < 0.05 and large < small / 2.5, (name, small, large)


def _chain(env, traj, p_goal_move, n_iter, seed, g0=None):
    """Run the rejuvenation kernel alone, as an MCMC chain on (goal, plan history) given traj.
    A low goal temperature makes Q(g) clearly non-uniform, so a wrong proposal ratio would show."""
    cfg = SIPSConfig(p_goal_move=p_goal_move, goal_temp=0.5)
    f = SIPS(env, TEST_PARAMS, cfg, np.random.default_rng(seed))
    f.obs = list(traj)
    p = f._regenerate(int(f.rng.integers(env.n_goals)) if g0 is None else g0, 0, None)
    samples = []
    for _ in range(n_iter):
        p = f.rejuvenate(p)
        samples.append(p)
    return samples, f


def _ambiguous_traj():
    """A trajectory prefix where the exact posterior is still spread out and the agent has left the
    start cell (at the start all gems are equally far, so the goal proposal would be uniform)."""
    env, problems = make_problems(n_per_goal=1)
    traj, exact = problems[0]
    ok = [t for t in range(1, len(traj)) if traj[t][:2] != traj[0][:2] and exact[t].max() < 0.8]
    t = min(ok, key=lambda t: exact[t].max())
    return env, traj[: t + 1], exact[t]


def test_mh_kernel_leaves_exact_goal_posterior_invariant():
    env, traj, exact = _ambiguous_traj()
    samples, f = _chain(env, traj, p_goal_move=0.5, n_iter=6000, seed=0)
    counts = np.bincount([p.g for p in samples[500:]], minlength=env.n_goals)
    freq = counts / counts.sum()
    assert 0.5 * np.abs(freq - exact).sum() < 0.04, (freq, exact)
    assert f.n_accept[0] > 50 and f.n_accept[1] > 50, "both kernels must actually move"


def test_replanning_kernel_targets_exact_plan_posterior():
    """With the goal fixed, the error-driven replanning kernel must sample the current plan from
    its exact posterior P(plan_T | g, observations)."""
    env, traj, _ = _ambiguous_traj()
    _, belief = exact_goal_posterior(env, TEST_PARAMS, traj, return_belief=True)
    g = 0
    cond = defaultdict(float)
    for (gg, mind), p in belief.items():
        if gg == g:
            cond[mind] += p
    z = sum(cond.values())
    samples, f = _chain(env, traj, p_goal_move=0.0, n_iter=6000, seed=1, g0=g)
    emp = Counter(p.mind for p in samples[500:])
    n = sum(emp.values())
    tv = 0.5 * sum(abs(cond.get(m, 0) / z - emp.get(m, 0) / n) for m in set(cond) | set(emp))
    assert tv < 0.06, tv
    assert f.n_accept[1] > 100
