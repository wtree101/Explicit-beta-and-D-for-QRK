import numpy as np
from functools import lru_cache
from scipy.integrate import quad
from scipy.stats import norm
from .debug import debug_log

# Module: error-increase utilities for fixed and Gaussian noise.

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
    # Acceptance window intersected with the true-increase region.
    lower = max(-qq + C, -abs(C))
    upper = min(qq + C, abs(C))
    if upper <= lower:
        return 0.0
    # Closed form: \int (C^2 - z^2) phi(z) dz over [lower, upper].
    c2_term = (C ** 2) * (norm.cdf(upper) - norm.cdf(lower))
    z2_term = (norm.cdf(upper) - upper * norm.pdf(upper)) - (norm.cdf(lower) - lower * norm.pdf(lower))
    return c2_term - z2_term


def find_C_with_largest_error_increase_fast(
    qq: float,
    C_min: float = 0,
    C_max: float = 100,
    num_points: int = 1000,
) -> tuple:
    """Grid-search for the noise magnitude C that maximises error_increased_C_2.

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
    errors = [error_increased_C_3(qq, C_val) for C_val in C_grid]
    max_idx = int(np.argmax(errors))
    C_largest = C_grid[max_idx]
    max_error = errors[max_idx]
    debug_log(f"Largest error increased for q={qq}: C={C_largest:.4f}, error={max_error:.4f}")
    return C_largest, max_error



def error_increased_Gaussian_noise(qq: float, sigma: float) -> float:
    """Expected error increase when noise C ~ N(0, sigma^2)."""
    if sigma <= 0:
        return 0.0

    def integrand(x: float) -> float:
        normal_density = (1.0 / (np.sqrt(2 * np.pi) * sigma)) * np.exp(
            -(x ** 2) / (2 * sigma ** 2)
        )
        return error_increased_C_3(qq, x) * normal_density

    # Gaussian expectation over C.
    result, _ = quad(integrand, -np.inf, np.inf)
    return result


def find_sigma_with_largest_error_increase_fast(
    qq: float,
    sigma_min: float = 0.0,
    sigma_max: float = 10.0,
    num_points: int = 200,
) -> tuple:
    """Grid-search for sigma that maximizes error_increased_Gaussian_noise."""
    # Coarse grid search; increase num_points for sharper peaks.
    sigma_grid = np.linspace(sigma_min, sigma_max, num_points)
    errors = [error_increased_Gaussian_noise(qq=qq, sigma=s) for s in sigma_grid]
    max_idx = int(np.argmax(errors))
    debug_log(f"sigma_grid errors: {errors}")
    debug_log(f"Largest error increased for phiq={qq}: sigma={sigma_grid[max_idx]:.4f}, error={errors[max_idx]:.4f}")
    return sigma_grid[max_idx], errors[max_idx]