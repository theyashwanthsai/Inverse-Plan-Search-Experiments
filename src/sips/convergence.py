"""SIPS vs exact inference on a tiny map, as a function of particle count."""
import numpy as np

from .agent import AgentParams, sample_trajectory
from .envs.dkg_maps import tiny_map
from .exact import exact_goal_posterior
from .smc import SIPS, SIPSConfig

# Agent parameters for the exact check: truncated budget prior (so enumeration is exact) and
# noisier search/actions than the experiments, so the posterior stays uncertain for several steps.
TEST_PARAMS = AgentParams(r=2, q=0.7, gamma=0.5, epsilon=0.1, budget_max=5)

LAYERS = {
    "1: importance weights only": dict(resample_threshold=None, rejuvenate=False),
    "2: + resampling (ESS<0.5k)": dict(resample_threshold=0.5, rejuvenate=False),
    "3: + resampling + MH rejuvenation": dict(resample_threshold=0.5, rejuvenate=True),
}


def make_problems(n_per_goal: int = 2, seed: int = 0, max_steps: int = 12):
    env = tiny_map()
    rng = np.random.default_rng(seed)
    trajs = [sample_trajectory(env, TEST_PARAMS, g, rng, max_steps=max_steps)
             for g in range(env.n_goals) for _ in range(n_per_goal)]
    return env, [(t, exact_goal_posterior(env, TEST_PARAMS, t)) for t in trajs]


def sips_error(env, problems, layer: dict, k: int, seed: int) -> float:
    """Mean total-variation distance between SIPS and exact goal posteriors, over problems and time."""
    errs = []
    for i, (traj, exact) in enumerate(problems):
        f = SIPS(env, TEST_PARAMS, SIPSConfig(n_particles=k, **layer), np.random.default_rng([seed, i, k]))
        post = f.run(traj)
        errs.append(0.5 * np.abs(post[1:] - exact[1:]).sum(axis=1).mean())
    return float(np.mean(errs))
