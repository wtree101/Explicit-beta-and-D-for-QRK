"""
Core quantile and integration utilities for the QRK analysis.

These functions underpin all subsequent feasibility checks and bound computations.
"""

import numpy as np
from functools import lru_cache
from scipy.stats import norm
from scipy.integrate import quad


@lru_cache(maxsize=4096)
def half_normal_quantile(q: float) -> float:
    """Return the q-quantile of the standard half-normal distribution.

    The half-normal distribution is |Z| where Z ~ N(0,1).  Its CDF satisfies
    P(|Z| <= x) = 2*Phi(x) - 1, so the q-quantile is Phi^{-1}((q+1)/2).

    Parameters
    ----------
    q : float
        Probability level in (0, 1).

    Returns
    -------
    float
        Half-normal q-quantile.
    """
    return norm.ppf((q + 1) / 2)


def Phi(q: float) -> float:
    """Alias for :func:`half_normal_quantile`."""
    return half_normal_quantile(q)


@lru_cache(maxsize=4096)
def conditional_expectation(alpha: float) -> float:
    """Compute E[Z^2 | |Z| <= Phi(alpha)] for Z ~ N(0,1).

    This quantity appears in the lower bound on sigma_min^2 when the
    uncorrupted update falls within the acceptance region [0, Phi(alpha)].

    Parameters
    ----------
    alpha : float
        Quantile level; determines the acceptance threshold Phi(alpha).

    Returns
    -------
    float
        The conditional second moment E[Z^2 | |Z| <= Phi(alpha)].
    """
    if alpha <= 0:
        return 0.0
    phi_alpha = Phi(alpha)
    # Closed form: E[Z^2 | |Z| <= a] = 1 - (2 a phi(a)) / P(|Z|<=a)
    p = 2 * norm.cdf(phi_alpha) - 1
    return 1.0 - (2.0 * phi_alpha * norm.pdf(phi_alpha)) / p


def integrate_gaussian(f, a: float, b: float) -> float:
    """Integrate f(z) * phi(z) over [a, b], where phi is the standard normal PDF.

    Parameters
    ----------
    f : callable
        Function of a single float argument.
    a, b : float
        Integration limits (may be -inf or +inf).

    Returns
    -------
    float
        Value of the integral.
    """
    result, _ = quad(lambda z: f(z) * norm.pdf(z), a, b)
    return result


@lru_cache(maxsize=4096)
def sigma_min_alpha0_square(alpha_0: float, q: float) -> float:
    """Lower bound on sigma_min^2 given the acceptance threshold alpha_0.

    Uses conditional_expectation(alpha_0) * alpha_0 as a conservative estimate.
    This corresponds to the fraction alpha_0 of rows whose inner product with
    the error vector falls inside the quantile acceptance region.

    Parameters
    ----------
    alpha_0 : float
        Lower threshold for Q_{q,k+1}.
    q : float
        Quantile level (kept for API consistency).

    Returns
    -------
    float
        Approximate sigma_min^2 value.
    """
    return conditional_expectation(alpha_0) * alpha_0
