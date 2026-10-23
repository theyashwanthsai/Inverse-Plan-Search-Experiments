"""Sequential Inverse Plan Search (SIPS): a particle filter over (goal, plan history).

Observations are the agent's states s_1..s_T (s_0 is known). Transitions are deterministic, so
consecutive states identify the action a_t. Each particle simulates its hypothesized agent one step
(plan update, only if needed) and is reweighted by P(a_t | s_{t-1}, plan_t): the epsilon-greedy
action probability. This equals the marginal likelihood of observing s_t, summing over actions.

Three layers, switched by config:
  1. importance weighting only            (resample_threshold=None, rejuvenate=False)
  2. + multinomial/systematic resampling  (resample_threshold=c)
  3. + Metropolis-Hastings rejuvenation   (rejuvenate=True)
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from .agent import AgentParams, Mind, action_probs, advance, plan_update
from .envs.base import Domain


@dataclass
class StepRecord:
    """What happened inside one particle at observation step t."""
    mind_in: Mind  # plan before the update at this step
    mind: Mind  # plan used to choose the action
    replanned: bool
    loglik: float  # log P(observed action | state, plan)
    slipped: bool  # observed action differs from the planned action


@dataclass
class Particle:
    g: int
    mind: Mind = field(default_factory=Mind)
    logw: float = 0.0
    hist: list = field(default_factory=list)

    def loglik(self, start: int = 0) -> float:
        return sum(r.loglik for r in self.hist[start:])

    def copy(self) -> "Particle":
        return Particle(self.g, self.mind, self.logw, list(self.hist))


@dataclass
class SIPSConfig:
    n_particles: int = 30
    stratified: bool = True  # n_particles/|G| particles per goal (the paper's "10 per goal")
    resample_threshold: float | None = 0.25  # resample when ESS/k < c; None disables
    rejuvenate: bool = False
    p_goal_move: float = 0.5  # mixture weight of the heuristic goal proposal
    goal_temp: float = 1.0  # softmax temperature for the goal proposal
    time_temp: float = 2.0  # temperature of the divergence-time proposal


class SIPS:
    def __init__(self, domain: Domain, params: AgentParams, config: SIPSConfig, rng, goal_prior=None):
        self.domain, self.params, self.cfg, self.rng = domain, params, config, rng
        G = domain.n_goals
        self.log_prior = np.log(np.full(G, 1.0 / G) if goal_prior is None else np.asarray(goal_prior))
        self.obs: list = []
        self.particles: list[Particle] = []
        self.n_expanded = 0  # node expansions across all planning calls (cost measure N)
        self.n_resample = 0
        self.n_accept = [0, 0]  # accepted goal moves, accepted replan moves
        self.n_proposed = [0, 0]

    # ------------------------------------------------------------ setup
    def init(self, s0):
        k, G = self.cfg.n_particles, self.domain.n_goals
        self.obs = [s0]
        if self.cfg.stratified and k % G == 0:
            # Equal counts per goal; the importance weight P(g) / (1/G) corrects for a non-uniform prior.
            goals = np.repeat(np.arange(G), k // G)
            logw = self.log_prior[goals] - np.log(1.0 / G)
        else:
            goals = self.rng.choice(G, size=k, p=np.exp(self.log_prior))
            logw = np.zeros(k)
        self.particles = [Particle(int(g), Mind(), float(w)) for g, w in zip(goals, logw)]

    # ------------------------------------------------------------ core step
    def _extend(self, p: Particle, s_prev, a) -> StepRecord:
        """Simulate one step of particle p's agent with the action clamped to the observed one."""
        mind, replanned, n = plan_update(self.domain, self.params, s_prev, p.mind, p.g, self.rng)
        self.n_expanded += n
        planned = mind.planned_action(s_prev)
        ll = math.log(action_probs(self.domain, self.params, s_prev, mind, p.g)[a])
        rec = StepRecord(p.mind, mind, replanned, ll, planned is not None and planned != a)
        p.mind = advance(mind, s_prev, a)
        p.hist.append(rec)
        return rec

    def observe(self, s_next):
        s_prev = self.obs[-1]
        a = self.domain.action_between(s_prev, s_next)
        if self.cfg.resample_threshold is not None and self.ess() / len(self.particles) < self.cfg.resample_threshold:
            self.resample()
            if self.cfg.rejuvenate:
                self.particles = [self.rejuvenate(p) for p in self.particles]
        for p in self.particles:
            p.logw += self._extend(p, s_prev, a).loglik
        self.obs.append(s_next)

    def run(self, states, timed: bool = False):
        """Filter a whole trajectory. Returns array [T+1, G] of goal posteriors (row 0 = prior),
        and, if timed, the start-up time and the per-step wall-clock times."""
        t0 = time.perf_counter()
        self.init(states[0])
        startup = time.perf_counter() - t0
        post, step_times = [self.goal_posterior()], []
        for s in states[1:]:
            t0 = time.perf_counter()
            self.observe(s)
            step_times.append(time.perf_counter() - t0)
            post.append(self.goal_posterior())
        return (np.array(post), startup, step_times) if timed else np.array(post)

    # ------------------------------------------------------------ weights
    def normalized_weights(self) -> np.ndarray:
        lw = np.array([p.logw for p in self.particles])
        w = np.exp(lw - lw.max())
        return w / w.sum()

    def ess(self) -> float:
        w = self.normalized_weights()
        return 1.0 / np.sum(w**2)

    def goal_posterior(self) -> np.ndarray:
        post = np.zeros(self.domain.n_goals)
        for p, w in zip(self.particles, self.normalized_weights()):
            post[p.g] += w
        return post

    def resample(self):
        """Systematic resampling; afterwards all weights are equal (their average is kept)."""
        w = self.normalized_weights()
        k = len(w)
        positions = (self.rng.random() + np.arange(k)) / k
        idx = np.minimum(np.searchsorted(np.cumsum(w), positions), k - 1)
        lw = np.array([p.logw for p in self.particles])
        mean_lw = lw.max() + math.log(np.mean(np.exp(lw - lw.max())))
        self.particles = [self.particles[i].copy() for i in idx]
        for p in self.particles:
            p.logw = mean_lw
        self.n_resample += 1

    # ------------------------------------------------------------ rejuvenation (MH)
    def _regenerate(self, g: int, start: int, prefix: Particle | None) -> Particle:
        """Sample steps start..t from the agent prior under goal g, actions clamped to the observed
        ones. Keeps prefix.hist[:start] unchanged. Returns a new particle (logw not set)."""
        if start == 0 or prefix is None:
            q = Particle(g)
        else:
            q = Particle(g, prefix.hist[start].mind_in, 0.0, list(prefix.hist[:start]))
        for t in range(start, len(self.obs) - 1):
            s_prev, s_next = self.obs[t], self.obs[t + 1]
            self._extend(q, s_prev, self.domain.action_between(s_prev, s_next))
        return q

    def _goal_proposal(self) -> np.ndarray:
        """Q(g) = softmax(-h(last observed state, g) / T): goals that look close are proposed more."""
        h = self.domain.heuristics()[self.params.heuristic]
        s = self.obs[-1]
        logits = -np.array([h(s, g) for g in range(self.domain.n_goals)]) / self.cfg.goal_temp
        logits = np.where(np.isfinite(logits), logits, -1e9)
        q = np.exp(logits - logits.max())
        return q / q.sum()

    def _time_proposal(self, p: Particle):
        """Distribution over this particle's replanning times, peaked near where its hypothesized
        actions first diverge from the observed ones (or the present, if they never diverge)."""
        times = [t for t, r in enumerate(p.hist) if r.replanned]
        t_div = next((t for t, r in enumerate(p.hist) if r.slipped), len(p.hist) - 1)
        logits = -np.abs(np.array(times) - t_div) / self.cfg.time_temp
        q = np.exp(logits - logits.max())
        return times, q / q.sum()

    def rejuvenate(self, p: Particle) -> Particle:
        if not p.hist:
            return p
        if self.rng.random() < self.cfg.p_goal_move:
            # Heuristic goal proposal: g' ~ Q, whole plan history redrawn from the prior given g'.
            # Prior terms of the plans cancel, leaving
            #   alpha = P(g') L(new) Q(g) / (P(g) L(old) Q(g')).
            self.n_proposed[0] += 1
            Q = self._goal_proposal()
            g_new = int(self.rng.choice(len(Q), p=Q))
            new = self._regenerate(g_new, 0, None)
            log_alpha = (self.log_prior[g_new] + new.loglik() + math.log(Q[p.g])
                         - self.log_prior[p.g] - p.loglik() - math.log(Q[g_new]))
            kind = 0
        else:
            # Error-driven replanning: pick a replanning time t* near the divergence point, redraw
            # the plans from t* onward from the prior. The reverse move must pick the same t* in the
            # new history, so the auxiliary ratio Q(t*|new)/Q(t*|old) enters alpha.
            self.n_proposed[1] += 1
            times, Qt = self._time_proposal(p)
            j = int(self.rng.choice(len(times), p=Qt))
            t_star = times[j]
            new = self._regenerate(p.g, t_star, p)
            new_times, new_Qt = self._time_proposal(new)
            if t_star not in new_times:  # reverse move impossible -> reject
                return p
            log_alpha = (new.loglik(t_star) - p.loglik(t_star)
                         + math.log(new_Qt[new_times.index(t_star)]) - math.log(Qt[j]))
            kind = 1
        if math.log(self.rng.random() + 1e-300) < log_alpha:
            new.logw = p.logw
            self.n_accept[kind] += 1
            return new
        return p
