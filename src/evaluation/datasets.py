"""Test-trajectory generation shared by the experiments."""
import numpy as np

from sips.planners import prob_astar
from .evaluate import Problem, agent_trajectories, optimal_trajectory

N_SUBOPT = 2  # agent-model trajectories per goal per instance (plus 1 optimal)


def domain_setup(name):
    """Returns (list of domain instances, admissible heuristic name, unbiased iterations, oracle iterations)."""
    if name == "dkg":
        from envs.dkg.maps import experiment_maps
        return experiment_maps(), "maze", 250_000, 10_000
    if name == "taxi":
        from envs.taxi import taxi_instances
        return taxi_instances(), "manhattan", 10_000, 2_500  # the paper's N for Taxi
    if name == "blockwords":
        from envs.blockwords import blockwords_instances
        return blockwords_instances(), "hmax", 250_000, 10_000
    raise ValueError(name)


def make_dataset(domains, admissible, params, rng):
    problems, rejected = [], 0
    for d in domains:
        h = d.heuristics()[admissible]
        for g in range(d.n_goals):
            problems.append(Problem(d.name, d, g, optimal_trajectory(d, g, h), "optimal"))
            trajs, rej = agent_trajectories(d, params, g, N_SUBOPT, rng)
            rejected += rej
            problems += [Problem(d.name, d, g, t, "suboptimal") for t in trajs]
    return problems, rejected


def boltzmann_trajectory(domain, vt, g, alpha, rng, max_steps=100):
    s = domain.initial_state
    states = [s]
    while not domain.is_goal(s, g) and len(states) <= max_steps:
        pol = vt.policy(s, g, alpha)
        acts = list(pol)
        s = domain.step(s, acts[rng.choice(len(acts), p=np.array(list(pol.values())))])
        states.append(s)
    return states


def optimal_trajectory_random_ties(domain, g, h, rng):
    """Probabilistic A* with unlimited budget and tiny gamma: an optimal plan, ties broken at random."""
    plan, _ = prob_astar(domain, domain.initial_state, g, h, gamma=0.01, budget=10**7, rng=rng)
    return list(plan.states)
