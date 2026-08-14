"""
Information-theoretic divergence utilities.
"""

import numpy as np


def DKL(q: float, p: float) -> float:
    """Kullback-Leibler divergence D_KL(q || p) for Bernoulli distributions.

    D_KL(q || p) = q * log(q/p) + (1-q) * log((1-q)/(1-p))

    Parameters
    ----------
    q : float
        First Bernoulli parameter (the "true" distribution).
    p : float
        Second Bernoulli parameter (the "reference" distribution).

    Returns
    -------
    float
        KL divergence value (always >= 0).
    """
    return q * np.log(q / p) + (1 - q) * np.log((1 - q) / (1 - p))
