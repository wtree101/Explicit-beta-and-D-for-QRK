"""
Error-increase functions for fixed (deterministic) noise C.

When the corruption has a fixed magnitude C the error increase can be
computed analytically by integrating over the acceptance window
[-qq + C, qq + C] (the region where the corrupted update is accepted).
"""

import numpy as np
from functools import lru_cache
from scipy.stats import norm
from ..core.quantile import integrate_gaussian


def error_increased_C_2(qq: float, C: float) -> float:
    """Error increase for fixed noise C using the simple C^2 - Z^2 integrand.

    Integrates (C^2 - Z^2) * phi(Z) over the acceptance region
    [-qq + C, qq + C].  This is an upper bound on the net error increase
    when the corrupted update is accepted by the quantile filter.

    Parameters
    ----------
    qq : float
        Quantile threshold (half-normal quantile at 1 - alpha').
    C : float
        Fixed (deterministic) noise magnitude (must be >= 0).

    Returns
    -------
    float
        Net error increase contribution from the corrupted update.
    """
    assert C >= 0, "C must be non-negative"
    return (
        integrate_gaussian(lambda z: C ** 2, -qq + C, qq + C)
        - integrate_gaussian(lambda z: z ** 2, -qq + C, qq + C)
    )


@lru_cache(maxsize=8192)
def error_increased_C_3(qq: float, C: float) -> float:
    """Error increase for fixed noise C using a tighter integration region.

    Integrates (C^2 - Z^2) * phi(Z) over the intersection of the acceptance
    window and the region where C^2 > Z^2, i.e. [max(-qq+C, -|C|), min(qq+C, |C|)].
    This avoids over-counting cases where the noise does not actually increase
    the error.

    Parameters
    ----------
    qq : float
        Quantile threshold.
    C : float
        Fixed noise value (may be negative; uses |C| internally).

    Returns
    -------
    float
        Net error increase (clipped to the true-increase region).
    """
    lower = max(-qq + C, -abs(C))
    upper = min(qq + C, abs(C))
    
    if upper <= lower:
        return 0.0
        
    # \int \phi(z) dz = \Phi(z)
    c2_term = (C ** 2) * (norm.cdf(upper) - norm.cdf(lower))
    
    # \int z^2 \phi(z) dz = \Phi(z) - z * \phi(z)
    z2_term = (norm.cdf(upper) - upper * norm.pdf(upper)) - (norm.cdf(lower) - lower * norm.pdf(lower))
    
    # Total integral is \int C^2 \phi(z) dz - \int z^2 \phi(z) dz
    return c2_term - z2_term


def find_C_with_largest_error_increase_fast(
    qq: float,
    C_min: float = 0,
    C_max: float = 10,
    num_points: int = 20,
) -> tuple:
    """Grid-search for the noise magnitude C that maximises error_increased_C_3.

    Parameters
    ----------
    qq : float
        Quantile threshold.
    C_min, C_max : float
        Search range for C.
    num_points : int
        Number of grid points.

    Returns
    -------
    tuple[float, float]
        (C_largest_error, max_error) – the maximising C and the corresponding
        error increase value.
    """
    C_grid = np.linspace(C_min, C_max, num_points)
    absolute_C = np.abs(C_grid)
    lower = np.maximum(-qq + C_grid, -absolute_C)
    upper = np.minimum(qq + C_grid, absolute_C)
    valid = upper > lower

    probabilities = norm.cdf(upper) - norm.cdf(lower)
    second_moments = (
        norm.cdf(upper)
        - upper * norm.pdf(upper)
        - norm.cdf(lower)
        + lower * norm.pdf(lower)
    )
    errors = np.where(valid, C_grid**2 * probabilities - second_moments, 0.0)
    max_idx = int(np.argmax(errors))
    C_largest = C_grid[max_idx]
    max_error = float(errors[max_idx])
    # print(f"Largest error increased for q={qq}: C={C_largest:.4f}, error={max_error:.4f}")
    return C_largest, max_error


def error_decreased_C_qq(C: float, q: float, beta: float) -> float:
    """Effective acceptance probability for uncorrupted update given noise C.

    Given fixed noise magnitude C > 0, derives the effective quantile level
    qq = 2*Phi(C) - 1 and clamps it to the valid range
    [(q - beta) / (1 - beta), q / (1 - beta)].

    Parameters
    ----------
    C : float
        Fixed noise magnitude.
    q : float
        Quantile level.
    beta : float
        Corruption fraction.

    Returns
    -------
    float
        Clamped effective acceptance probability for the uncorrupted update.
    """
    qq = 2 * norm.cdf(C) - 1
    if qq > q / (1 - beta):
        return q / (1 - beta)
    if qq < (q - beta) / (1 - beta):
        return (q - beta) / (1 - beta)
    return qq
