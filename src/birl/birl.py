"""Bayesian IRL baseline: Boltzmann-rational policies from value iteration, exact posterior over goals.

Each goal g is an indicator reward: entering a state where g holds pays 1, and such states are
terminal (V = 0 there). Discount 0.9, Boltzmann policy pi(a|s,g) proportional to exp(alpha Q_g(s,a)).

V tables are dicts defaulting to 0, so a state that value iteration never backed up looks
worthless; that is exactly why unconverged VI gives a near-uniform policy and a flat posterior.
"""
import time

import numpy as np

from envs.base import Domain


class ValueTables:
    def __init__(self, domain: Domain, discount: float = 0.9):
        self.domain, self.discount = domain, discount
        self.V = [dict() for _ in range(domain.n_goals)]
        self._succ: dict = {}
        self.n_backups = 0

    def successors(self, s):
        if s not in self._succ:
            self._succ[s] = self.domain.successors(s)
        return self._succ[s]

    def q_values(self, s, g: int):
        acts, qs = [], []
        for a, t in self.successors(s):
            acts.append(a)
            qs.append(1.0 if self.domain.is_goal(t, g) else self.discount * self.V[g].get(t, 0.0))
        return acts, np.array(qs)

    def backup(self, s) -> float:
        """Bellman backup of s for every goal; returns the largest change."""
        self.n_backups += 1
        delta = 0.0
        for g in range(self.domain.n_goals):
            if self.domain.is_goal(s, g):
                continue
            _, q = self.q_values(s, g)
            new = float(q.max()) if len(q) else 0.0
            delta = max(delta, abs(new - self.V[g].get(s, 0.0)))
            self.V[g][s] = new
        return delta

    def policy(self, s, g: int, alpha: float) -> dict:
        acts, q = self.q_values(s, g)
        w = np.exp(alpha * (q - q.max()))
        return dict(zip(acts, w / w.sum()))


def exact_vi(domain: Domain, discount: float = 0.9, tol: float = 1e-10, max_sweeps: int = 10_000) -> ValueTables:
    """Synchronous-order sweeps over every state reachable from the start, until converged."""
    vt = ValueTables(domain, discount)
    states = domain.reachable_states()
    for _ in range(max_sweeps):
        if max(vt.backup(s) for s in states) < tol:
            break
    return vt


def async_vi(domain: Domain, sampler, n_iter: int, rng, discount: float = 0.9) -> ValueTables:
    """Asynchronous VI: back up one sampled state per iteration."""
    vt = ValueTables(domain, discount)
    for _ in range(n_iter):
        vt.backup(sampler(rng))
    return vt


def uniform_sampler(domain: Domain):
    return domain.sample_state


def oracle_sampler(domain: Domain, trajectories, p_uniform: float = 0.1):
    """Biased sampler for oracle BIRL: states on the observed trajectories and their one-step
    successors, mixed with a little uniform sampling. Needs all test trajectories in advance."""
    pool = set()
    for traj in trajectories:
        for s in traj:
            pool.add(s)
            pool.update(t for _, t in domain.successors(s))
    pool = list(pool)

    def sample(rng):
        if rng.random() < p_uniform:
            return domain.sample_state(rng)
        return pool[int(rng.integers(len(pool)))]

    return sample


class BIRL:
    """Exact goal posterior under Boltzmann-rational policies from precomputed value tables."""

    def __init__(self, vt: ValueTables, alpha: float = 1.0, goal_prior=None):
        self.vt, self.alpha = vt, alpha
        G = vt.domain.n_goals
        self.log_prior = np.log(np.full(G, 1.0 / G) if goal_prior is None else np.asarray(goal_prior))

    def run(self, states, timed: bool = False):
        domain = self.vt.domain
        logp = self.log_prior.copy()
        post = [np.exp(logp)]
        step_times = []
        for s, s_next in zip(states[:-1], states[1:]):
            t0 = time.perf_counter()
            a = domain.action_between(s, s_next)
            for g in range(domain.n_goals):
                logp[g] += np.log(self.vt.policy(s, g, self.alpha)[a])
            p = np.exp(logp - logp.max())
            post.append(p / p.sum())
            step_times.append(time.perf_counter() - t0)
        return (np.array(post), step_times) if timed else np.array(post)
