"""
Error-increase functions for oblivious (random) Massart noise.

In the oblivious setting the corruption epsilon_{k+1,0} is drawn
independently of the row a_{r_{k+1}} and the quantile subsample.
For Gaussian noise C ~ N(0, sigma^2) the error increase is obtained by
averaging error_increased_C_3 over the noise distribution.
"""

import numpy as np
from scipy.integrate import quad, quad_vec
from scipy.special import ndtr

from .fixed import error_increased_C_3


def error_increased_Gaussian_noise(qq: float, sigma: float) -> float:
    """Expected error increase when noise C ~ N(0, sigma^2).

    Averages the pointwise error increase error_increased_C_3(qq, C) over
    C ~ N(0, sigma^2).  This exploits the independence of the noise from
    the row selection to evaluate the increase at the same quantile value
    used for the uncorrupted decrease (joint oblivious analysis).

    Parameters
    ----------
    qq : float
        Quantile threshold (half-normal quantile at 1 - Qq_conditioned).
    sigma : float
        Standard deviation of the Gaussian noise.

    Returns
    -------
    float
        E_C[error_increased_C_3(qq, C)] where C ~ N(0, sigma^2).
    """
    def integrand(x: float) -> float:
        normal_density = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -(x ** 2) / (2 * sigma ** 2)
        )
        return error_increased_C_3(qq, x) * normal_density

    result, _ = quad(integrand, -np.inf, np.inf)
    return result


def error_increased_Gaussian_noise_batch(
    qq: float,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Evaluate the Gaussian expectation for several standard deviations.

    ``quad_vec`` shares the adaptive integration grid across all values of
    ``sigma``, which is substantially faster than calling the scalar routine
    once per grid point.
    """
    return error_increased_Gaussian_noise_grid(
        np.asarray([qq], dtype=float),
        sigmas,
    )[0]


def error_increased_Gaussian_noise_grid(
    quantile_thresholds: np.ndarray,
    sigmas: np.ndarray,
) -> np.ndarray:
    """Evaluate expectations on a threshold-by-sigma parameter grid."""
    threshold_values = np.asarray(quantile_thresholds, dtype=float)
    if threshold_values.ndim != 1:
        raise ValueError("quantile_thresholds must be a one-dimensional array")
    if np.any(threshold_values < 0):
        raise ValueError("quantile thresholds must be nonnegative")

    sigma_values = np.asarray(sigmas, dtype=float)
    if sigma_values.ndim != 1:
        raise ValueError("sigmas must be a one-dimensional array")
    if np.any(sigma_values <= 0):
        raise ValueError("all sigma values must be positive")

    normalizers = np.sqrt(2.0 * np.pi) * sigma_values

    def integrand(x: float) -> np.ndarray:
        absolute_x = abs(x)
        lower = np.maximum(-threshold_values + x, -absolute_x)
        upper = np.minimum(threshold_values + x, absolute_x)
        valid = upper > lower
        probabilities = ndtr(upper) - ndtr(lower)
        inverse_sqrt_two_pi = 1.0 / np.sqrt(2.0 * np.pi)
        upper_density = inverse_sqrt_two_pi * np.exp(-(upper**2) / 2.0)
        lower_density = inverse_sqrt_two_pi * np.exp(-(lower**2) / 2.0)
        second_moments = (
            probabilities - upper * upper_density + lower * lower_density
        )
        pointwise_errors = np.where(
            valid,
            x**2 * probabilities - second_moments,
            0.0,
        )
        densities = np.exp(-(x**2) / (2.0 * sigma_values**2)) / normalizers
        return pointwise_errors[:, np.newaxis] * densities[np.newaxis, :]

    result, _ = quad_vec(
        integrand,
        -np.inf,
        np.inf,
        epsabs=1e-8,
        epsrel=1e-7,
    )
    return np.asarray(result, dtype=float)
