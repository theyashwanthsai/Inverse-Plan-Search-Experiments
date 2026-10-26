"""Run the full model, its lesions and the Boltzmann baseline on stimuli."""
from dataclasses import replace

import numpy as np

from birl.birl import BIRL, exact_vi
from sips.smc import SIPS, SIPSConfig
from .agent import MistakeParams, lesion
from .smc import MistakeSIPS

BOLTZMANN_ALPHA = 50.0  # as in the Boltzmann-RL agents of Exp 1.3; not fitted (no human data)


def sips_model(mp: MistakeParams, per_goal: int):
    def run(stim, seed):
        cfg = SIPSConfig(n_particles=per_goal * mp.n_candidates(stim.domain), resample_threshold=0.25)
        return MistakeSIPS(stim.domain, mp, cfg, np.random.default_rng(seed)).run(stim.states)
    return run


def boltzmann_model(plain_domain_of, alpha=BOLTZMANN_ALPHA):
    """plain_domain_of(stim) -> the same world with only the candidate goals (cached value tables)."""
    cache = {}

    def run(stim, seed):
        d = plain_domain_of(stim)
        if d.name not in cache:
            cache[d.name] = exact_vi(d)
        return BIRL(cache[d.name], alpha).run(stim.states)
    return run


def model_zoo(mp: MistakeParams, per_goal: int, plain_domain_of, levels: str):
    """levels: letters of the mistake levels the full model has ('GPA' or 'PA'); each is lesioned in turn."""
    zoo = {"full": sips_model(mp, per_goal)}
    for L in levels:
        zoo[f"{L}-lesioned"] = sips_model(lesion(mp, L), per_goal)
    zoo["Boltzmann"] = boltzmann_model(plain_domain_of)
    return zoo


def evaluate(stimuli, zoo, seed=0, runs=1):
    """posteriors[model][i] = list over runs of [T+1, G] arrays."""
    out = {m: [] for m in zoo}
    for i, stim in enumerate(stimuli):
        for m, f in zoo.items():
            out[m].append([f(stim, [seed, i, r]) for r in range(runs)])
    return out


def p_true(stimuli, posteriors) -> dict:
    """Per model, per stimulus: P(true goal) averaged over time steps 1..T and over runs."""
    return {m: [float(np.mean([post[1:, s.g].mean() for post in runs])) for s, runs in zip(stimuli, rs)]
            for m, rs in posteriors.items()}
