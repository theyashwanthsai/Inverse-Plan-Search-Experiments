"""Open-ended goal inference over words."""
import numpy as np
import pytest

from envs.blockwords import BlockWords
from openended import inference as inf
from openended.agent import PolicyModel, sample_trajectory
from openended.words import INITIAL_CONFIGS, LETTERS, WordModel


@pytest.fixture(scope="module")
def world():
    wm = WordModel()
    d = BlockWords(INITIAL_CONFIGS[0], words=wm.words, letters=LETTERS)
    rng, g = np.random.default_rng(0), wm.index["rose"]
    while True:  # a trajectory that gets there
        traj = sample_trajectory(d, PolicyModel(d, 2.0, 0), g, rng)
        if d.is_goal(traj[-1], g) and len(traj) > 6:
            return wm, d, traj


def test_word_model_is_a_distribution_and_proposal_follows_fragments():
    wm = WordModel()
    assert len(wm.words) > 400
    assert abs(wm.prior.sum() - 1) < 1e-9 and abs(wm.ngram.sum() - 1) < 1e-9
    q = wm.proposal(["ro"])
    assert abs(q.sum() - 1) < 1e-9
    has = [wm.index[w] for w in wm.words if "ro" in w]
    assert q[has].sum() > 3 * wm.ngram[has].sum()  # mass moves to words containing the fragment


def test_policy_prefers_actions_that_reduce_the_heuristic(world):
    wm, d, _ = world
    pm = PolicyModel(d, beta=2.0, depth=0)
    g = wm.index["rose"]
    pol = pm.policy(d.initial_state, g)
    assert abs(sum(pol.values()) - 1) < 1e-9
    best = min(d.successors(d.initial_state), key=lambda at: pm.h(at[1], g))[0]
    assert pol[best] == max(pol.values())


def test_exact_posterior_rises_on_the_true_word(world):
    wm, d, traj = world
    post = inf.exact(wm, PolicyModel(d, 2.0, 0), traj)
    g = wm.index["rose"]
    assert post[-1, g] > 10 * post[0, g] and np.allclose(post.sum(1), 1)


def test_open_ended_sips_converges_to_exact(world):
    wm, d, traj = world
    ex = inf.exact(wm, PolicyModel(d, 2.0, 0), traj)

    def err(n, seed):
        return 0.5 * np.abs(inf.OpenEndedSIPS(wm, PolicyModel(d, 2.0, 0), n, np.random.default_rng(seed)).run(traj) - ex).sum(1).mean()
    small = np.mean([err(10, s) for s in range(3)])
    large = np.mean([err(1000, s) for s in range(3)])
    assert large < 0.12 and large < small / 3, (small, large)


def test_weights_estimate_the_marginal_likelihood(world):
    """The mean pool weight must estimate P(observations). This is what catches a wrong weight scale
    between surviving particles and fresh proposals, which the posterior alone can hide."""
    wm, d, traj = world
    pm = PolicyModel(d, 2.0, 0)
    loglik = np.zeros(len(wm.words))
    for s, s2 in zip(traj[:-1], traj[1:]):
        a = d.action_between(s, s2)
        loglik += np.array([pm.logpi(s, a, g) for g in range(len(wm.words))])
    true_log_z = float(np.logaddexp.reduce(np.log(wm.prior) + loglik))
    est = []
    for seed in range(8):
        f = inf.OpenEndedSIPS(wm, PolicyModel(d, 2.0, 0), 200, np.random.default_rng(seed))
        f.run(traj)
        est.append(f.log_evidence[-1])
    assert abs(np.log(np.mean(np.exp(np.array(est) - true_log_z)))) < 0.3, (est, true_log_z)
