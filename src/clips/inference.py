"""CLIPS goal inference (thesis Algorithm 3), by exact enumeration.

Hypotheses are (goal g, cost function C, rationality beta). Goals and cost functions have uniform
priors; beta has a Gamma(0.5, 1) prior discretised on 33 values 2^-3 .. 2^5 (thesis App. 4.8.1).
Enumerating beta is the same computation as the thesis's Rao-Blackwellisation over it.

Per observed step the weight of each hypothesis is multiplied by:
  P(a_h | s, pi)                    the human's action under the Boltzmann joint policy
  sum_c P(u | c) P(c | s, pi)        if the human spoke (LLM scores P(u | c))
The robot's own actions are interventions, do(a_r), and are never conditioned on (Eq. 4.8).
P(s_t | s_t-1, a) is 1 (deterministic dynamics) and P(d_t) = p_speak is the same for every
hypothesis in Ch. 4, so both drop out.

Ablations from the same code: use_utterance=False is action-only inverse planning (the exact
Boltzmann variant of SIPS); use_actions=False is language-only inverse planning.
"""
from dataclasses import dataclass, field

import numpy as np

from .commands import EXAMPLES, command_distribution, salient_actions
from .env import HUMAN
from .planner import JointPlanner, boltzmann_logprobs

BETAS = 2.0 ** np.arange(-3, 5.0001, 0.25)


def beta_prior(shape: float = 0.5, rate: float = 1.0) -> np.ndarray:
    """Gamma(shape, rate) mass on the log-spaced grid: density times bin width (bin width is
    proportional to beta on a log grid)."""
    w = BETAS ** (shape - 1) * np.exp(-rate * BETAS) * BETAS
    return w / w.sum()


def _logsumexp(x, axis=None):
    m = np.max(x, axis=axis, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    with np.errstate(divide="ignore"):  # log(0) = -inf for hypotheses with zero weight
        out = m + np.log(np.sum(np.exp(x - m), axis=axis, keepdims=True))
    return np.squeeze(out, axis=axis) if axis is not None else out.item()


@dataclass
class Belief:
    """log weights over (goal, cost function, beta), unnormalised."""
    logw: np.ndarray
    goal_history: list = field(default_factory=list)  # goal posterior after each update
    commands: dict = field(default_factory=dict)  # (g, c) -> {command: P(c | pi)} at the utterance
    scores: dict = field(default_factory=dict)  # command -> LLM log-score

    def hypothesis_posterior(self) -> np.ndarray:
        """P(g, C), marginalising beta."""
        lw = _logsumexp(self.logw, axis=2)
        return np.exp(lw - _logsumexp(lw))

    def goal_posterior(self) -> np.ndarray:
        return self.hypothesis_posterior().sum(axis=1)


class GoalInference:
    def __init__(self, planner: JointPlanner, scorer=None, use_actions: bool = True,
                 use_utterance: bool = True, K: int = 3, examples=EXAMPLES,
                 beta_shape: float = 0.5, beta_rate: float = 1.0):
        self.P, self.scorer = planner, scorer
        self.use_actions, self.use_utterance = use_actions, use_utterance
        self.K, self.examples = K, examples
        self.log_beta_prior = np.log(beta_prior(beta_shape, beta_rate))

    @property
    def shape(self):
        return (self.P.env.n_goals, len(self.P.cost_fns), len(BETAS))

    def init(self) -> Belief:
        G, C, _ = self.shape
        logw = np.zeros(self.shape) - np.log(G) - np.log(C) + self.log_beta_prior[None, None, :]
        b = Belief(logw)
        b.goal_history.append(b.goal_posterior())
        return b

    def observe_action(self, b: Belief, s, a: str) -> None:
        """Condition on the human's action a in state s. Robot actions: do not call this."""
        assert s[0] == HUMAN, "only the human's actions are evidence (the robot's are do(a_r))"
        if self.use_actions:
            G, C, _ = self.shape
            for g in range(G):
                for c in range(C):
                    acts, q = self.P.Q(s, g, c)
                    b.logw[g, c] += boltzmann_logprobs(q, BETAS)[:, acts.index(a)]
        b.goal_history.append(b.goal_posterior())

    def utterance_model(self, s) -> dict:
        """P(c | s, pi) for every (g, C): roll out the joint policy, keep salient actions,
        enumerate commands."""
        env = self.P.env
        G, C, _ = self.shape
        return {(g, c): command_distribution(env, salient_actions(env, self.P.greedy_rollout(s, g, c)), self.K)
                for g in range(G) for c in range(C)}

    def observe_utterance(self, b: Belief, s, utterance: str) -> None:
        """Condition on an utterance spoken in state s: weight *= sum_c P(u | c) P(c | s, pi)."""
        if self.use_utterance:
            b.commands = self.utterance_model(s)
            union = sorted({cmd for dist in b.commands.values() for cmd in dist})
            if union:
                scores = self.scorer.loglik(utterance, union, self.examples)
                b.scores = dict(zip(union, scores))
                for (g, c), dist in b.commands.items():
                    if dist:
                        term = _logsumexp(np.array([b.scores[cmd] + np.log(p) for cmd, p in dist.items()]))
                    else:
                        term = -np.inf  # this plan has nothing to communicate: no utterance expected
                    b.logw[g, c] += term
                if not np.isfinite(b.logw).any():
                    raise ValueError("utterance has zero likelihood under every hypothesis")
        b.goal_history.append(b.goal_posterior())

    def run(self, states, human_actions, utterance: str | None) -> Belief:
        """Observe a problem: human actions at the human-turn states, then the utterance in the
        final state."""
        b = self.init()
        human_states = [s for s in states[:-1] if s[0] == HUMAN]
        assert len(human_states) == len(human_actions)
        for s, a in zip(human_states, human_actions):
            self.observe_action(b, s, a)
        if utterance is not None:
            self.observe_utterance(b, states[-1], utterance)
        return b
