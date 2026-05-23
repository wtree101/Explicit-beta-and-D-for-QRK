"""Quantile and integration utilities for the QRK analysis."""

# Module: quantile and conditional expectation helpers.

import numpy as np
from scipy.stats import norm


def half_normal_quantile(q: float) -> float:
    """q-quantile of the standard half-normal |Z|, Z ~ N(0,1).

    P(|Z| <= x) = 2*Phi(x) - 1  =>  x = Phi^{-1}((q+1)/2).
    """
    return norm.ppf((q + 1) / 2)


def conditional_expectation(alpha: float) -> float:
    """E[Z^2 | |Z| <= Phi(alpha)] for Z ~ N(0,1).

    Closed-form derivation via integration by parts:
        2 * integral_0^x  z^2 * phi(z) dz  =  2 * (-x*phi(x) + alpha/2)
    so
        E[Z^2 | |Z| <= x]  =  1 - 2 * x * phi(x) / alpha,
    where x = half_normal_quantile(alpha) and phi is the standard normal PDF.
    """
    if alpha <= 0.0:
        return 0.0
    if alpha >= 1.0:
        return 1.0
    # Closed form for E[Z^2 | |Z| <= x], with x = Phi^{-1}((alpha+1)/2).
    x = half_normal_quantile(alpha)
    return 1.0 - 2.0 * x * norm.pdf(x) / alpha


def sigma_min_alpha0_square(alpha_0: float, q: float) -> float:
    """Lower bound on sigma_min^2 at threshold alpha_0.

    Returns conditional_expectation(alpha_0) * alpha_0.
    The argument q is kept for API consistency but is not used here.
    """
    return conditional_expectation(alpha_0) * alpha_0
