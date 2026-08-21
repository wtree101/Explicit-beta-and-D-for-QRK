"""
Upper-bound (Chernoff) concentration probabilities for the quantile estimator.

These functions compute the Chernoff-type tail probabilities that bound the
chance of the empirical quantile Q_{q,k+1} falling too low (p_l) or too
high (p_u) relative to the true quantile q.
"""

import numpy as np
from ..core.divergence import DKL


def check_quantile_concentration(
    D: float,
    beta: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
) -> dict:
    """Compute Chernoff concentration probabilities for the quantile subsample.

    Parameters
    ----------
    D : float
        Subsample size.
    beta : float
        Corruption fraction.
    q : float
        Quantile level.
    alpha_0 : float
        Lower slack: P(Q_{q,k+1} < q - alpha_0) is bounded by p_l.
    alpha_prime : float
        Upper slack: P(Q_{q,k+1} > 1 - alpha_prime) is bounded by p_u.

    Returns
    -------
    dict
        p_l    – P(S*^c), probability of the lower concentration event failing.
        p_l_c  – P(S*)  = 1 - p_l.
        p_u    – P(upper quantile event), Chernoff upper bound.
    """
    p_l = np.exp(-DKL(q, beta + alpha_0) * D)
    p_l_c = 1.0 - p_l
    p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    return {"p_l": p_l, "p_l_c": p_l_c, "p_u": p_u}
