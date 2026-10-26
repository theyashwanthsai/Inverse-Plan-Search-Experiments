"""Goal assistance (thesis Algorithm 4) and its evaluation against a simulated human.

The robot picks argmin_a sum_i w_i Q_i(s, a) over goal/cost hypotheses (the Q_MDP approximation),
with the posterior frozen after the observations ("offline" mode, as in the thesis experiments).
Hypotheses with weight < 0.02 are dropped, as in the thesis.

The simulated human (App. 4.8.1, simplified): follows the optimal joint policy to the true goal
under the true cost function, but switches to doing everything alone once the robot's actions are
10x more likely under a uniformly random policy than under the joint policy (Boltzmann, beta = 1).
The thesis's human also follows a scripted "ground truth" plan first; ours starts on the optimal
joint policy.

Our addition: the human also gives up when a joint state repeats on their turn. Q_MDP with a
fixed posterior can make the robot cycle between two goals' plans forever, and the likelihood-ratio
rule alone rarely fires on such a cycle because a beta = 1 policy is nearly flat.
"""
from dataclasses import dataclass

import numpy as np

from .env import HUMAN, ROBOT
from .planner import JointPlanner, boltzmann_logprobs

Q_CAP = 1e3  # stands in for an infinite cost-to-go when averaging over hypotheses


def qmdp_action(P: JointPlanner, post_gc: np.ndarray, s, cutoff: float = 0.02) -> str:
    """The assistance policy: minimise the posterior-weighted Q over the robot's actions."""
    w = np.where(post_gc >= cutoff, post_gc, 0.0)
    w = w / w.sum()
    total, acts = None, None
    for (g, c), wi in np.ndenumerate(w):
        if wi == 0:
            continue
        acts, q = P.Q(s, g, c)
        total = wi * np.minimum(q, Q_CAP) + (0 if total is None else total)
    assert total is not None and acts is not None
    return acts[int(np.argmin(total))]


@dataclass
class Episode:
    reached: bool
    steps: int  # actions taken by both agents after the utterance
    human_cost: float
    robot_options: set  # keys the robot picked up / doors it unlocked
    fallback: bool  # the simulated human gave up on the robot


def simulate(P: JointPlanner, post_gc: np.ndarray, s, true_goal: int, true_cost: int, options: str,
             max_steps: int = 120, give_up_ratio: float = 10.0) -> Episode:
    env = P.env
    table = P.cost_fns[true_cost].table()
    verb = "pickup" if options == "keys" else "unlock"
    robot_opts, human_cost, steps, log_lr, solo = set(), 0.0, 0, 0.0, False
    seen = set()
    while steps < max_steps and s[7] < 0:
        if s[0] == ROBOT:
            a = qmdp_action(P, post_gc, s)
            acts, q = P.Q(s, true_goal, true_cost)
            log_lr += boltzmann_logprobs(q, np.array([1.0]))[0, acts.index(a)] + np.log(len(acts))
            if log_lr < -np.log(give_up_ratio):
                solo = True
            sal = env.salient(s, a)
            if sal is not None and sal[0] == verb:
                robot_opts.add(sal[2])
        else:
            if s in seen:
                solo = True
            seen.add(s)
            acts, q = P.Q(s, true_goal, true_cost, "solo" if solo else "joint")
            a = acts[int(np.argmin(q))]
            human_cost += table[HUMAN, env.kind(a)]
        s = env.step(s, a)
        steps += 1
    return Episode(s[7] == true_goal, steps, human_cost, robot_opts, solo)


def gold_options(P: JointPlanner, s, true_goal: int, true_cost: int, options: str) -> set:
    """The assistance options in the optimal joint plan for the true goal and cost function."""
    verb = "pickup" if options == "keys" else "unlock"
    return {sal[2] for st, a in P.greedy_rollout(s, true_goal, true_cost)
            if (sal := P.env.salient(st, a)) is not None and sal[0] == verb and sal[1] == "robot"}


def precision_recall(chosen: set, gold: set) -> tuple[float, float]:
    if not chosen:
        return (1.0 if not gold else 0.0), (1.0 if not gold else 0.0)
    hit = len(chosen & gold)
    return hit / len(chosen), (hit / len(gold) if gold else 1.0)
