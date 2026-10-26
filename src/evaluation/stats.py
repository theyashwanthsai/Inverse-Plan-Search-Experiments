import numpy as np


def mean_se(x) -> tuple[float, float]:
    """Mean and standard error of the mean."""
    x = np.asarray(x, dtype=float)
    return float(x.mean()), float(x.std(ddof=1) / np.sqrt(len(x))) if len(x) > 1 else 0.0
