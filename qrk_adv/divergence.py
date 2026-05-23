"""Bernoulli KL divergence."""

# Module: KL divergence utilities for Bernoulli parameters.

import numpy as np


def DKL(q: float, p: float) -> float:
    """Kullback-Leibler divergence D_KL(q || p) for Bernoulli distributions.

        D_KL(q || p) = q * log(q/p) + (1-q) * log((1-q)/(1-p))

    Parameters
    ----------
    q : float
        First Bernoulli parameter.
    p : float
        Second Bernoulli parameter.

    Returns
    -------
    float
        KL divergence value (>= 0).
    """
    # Assumes q, p are strictly within (0, 1).
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
