"""One place to create random generators, so every run is reproducible."""
import numpy as np

DEFAULT_SEED = 0


def make_rng(seed: int | None = None) -> np.random.Generator:
    return np.random.default_rng(DEFAULT_SEED if seed is None else seed)
