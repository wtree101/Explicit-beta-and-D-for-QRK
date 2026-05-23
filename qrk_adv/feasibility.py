"""Feasibility condition check for the adversarial (Massart) noise setting.

Preliminary analysis (not revised).  Given (T, beta, D, q, alpha_0,
alpha_prime, delta_f), verifies three conditions required for geometric
convergence of QRK:

  1. p_l_c  – lower Chernoff: quantile subsamples concentrate from below.
  2. c > 0  – net one-step contraction is positive.
  3. failure_prob <= delta_f  – upper Chernoff over T iterations.
"""

# Module: feasibility checks for adversarial and Gaussian-noise variants.

import numpy as np
from .divergence import DKL
from .quantile import half_normal_quantile, sigma_min_alpha0_square
from .noise import (
    error_increased_C_3,
    error_increased_Gaussian_noise,
    find_sigma_with_largest_error_increase_fast,
)

_E_ABS_Z = np.sqrt(2.0 / np.pi)   # E[|Z|] for Z ~ N(0,1)


def check_feasibility(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    c_target: float = 0.0,
) -> dict:
    """Check adversarial Massart feasibility conditions.

    Parameters
    ----------
    T : int
        Number of QRK iterations.
    beta : float
        Corruption fraction (fraction of rows with arbitrary noise).
    D : float
        Subsample size used to compute the quantile.
    q : float
        Quantile level (e.g. 0.75).
    alpha_0 : float
        Lower slack parameter; must satisfy 0 < alpha_0 < q - beta.
    alpha_prime : float
        Upper slack parameter; must satisfy 0 < alpha_prime < 1 - q - beta.
    delta_f : float
        Allowed total failure probability over T iterations.
    c_target : float, optional
        Minimum required net contraction rate.  Default 0.0 (c > 0 suffices).

    Returns
    -------
    dict with keys:
        feasible     : bool
        p_l_c        : float  (lower concentration probability)
        c            : float  (net contraction coefficient)
        p_u          : float  (upper concentration probability per step)
        failure_prob : float  (1 - (1-p_u)^T)
        reason       : str    (only present when infeasible)
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}

    # lower concentration 
    if D == np.inf:
        p_l_c = 1.0
    else:
        p_l_c = 1.0 - np.exp(-DKL(q, beta + alpha_0) * D)

    # --- Condition 1: net contraction ---
    # sigma_min^2 uses alpha_0 / (1 - beta) as the effective threshold.
    sigma_min_sq = sigma_min_alpha0_square(alpha_0 / (1 - beta), q)
    # Adversarial error-increase bound at Phi(1 - alpha'/(1-beta)).
    Phi_val = half_normal_quantile(1 - alpha_prime / (1 - beta))
    c = (
        (1 - beta) * p_l_c * sigma_min_sq
        - beta * (Phi_val ** 2 + 2 * Phi_val * _E_ABS_Z)
    )
    if c < c_target:
        return {"feasible": False, "reason": "c condition violated", "c": c}

    # --- Condition 2: failure probability ---
    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1.0 - (1.0 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
        }

    return {
        "feasible": True,
        "p_l_c": p_l_c,
        "c": c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }





def check_feasibility_conditions_random_sup_revised(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    num_grid: int = 20,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
    num_points: int = 20,
    c_target: float = 0.0,
) -> dict:
    """Revised Gaussian check taking a supremum over sigma on a grid.

    This is the oblivious Gaussian-noise variant. It evaluates the net
    contraction over a grid of conditional quantiles and, for each, takes the
    worst-case (largest) error increase over a sigma grid.

    Parameters
    ----------
    T : int
        Number of QRK iterations.
    beta : float
        Corruption fraction.
    D : float
        Subsample size for the quantile.
    q : float
        Quantile level.
    alpha_0 : float
        Lower slack; must satisfy 0 <= alpha_0 <= q - beta.
    alpha_prime : float
        Upper slack; must satisfy 0 <= alpha_prime <= 1 - q - beta.
    delta_f : float
        Allowed total failure probability over T iterations.
    num_grid : int
        Number of grid points for the conditional quantile sweep. Grid search to find inf over \tilde q in [alpha_0, 1 - alpha_prime]).
    sigma_min, sigma_max : float
        Search range for sigma in the Gaussian noise model.
    num_points : int
        Number of grid points for the sigma sweep. Grid search to find sup over sigma in [sigma_min, sigma_max].
    
    Larger num_grid and num_points and larger regions (sigma_min, sigma_max) yield more accurate results but increase runtime.  Adjust as needed for a speed-accuracy tradeoff. Possible future improvement: replace the grid search with a more efficient optimization method.

    c_target : float
        Minimum required contraction rate.
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}

    if D == np.inf:
        p_l_c = 1.0
        p_l = 0.0
    else:
        p_l_c = 1.0 - np.exp(-DKL(q, beta + alpha_0) * D)
        p_l = np.exp(-DKL(q, beta + alpha_0) * D)
    # Penalty for failing the lower-quantile concentration event.
    S_star_penalty = (1 - beta) * p_l * sigma_min_alpha0_square(alpha_0, q)

    # Sweep conditional quantiles in the admissible interval.
    q_grid = np.linspace(alpha_0, 1.0 - alpha_prime, num_grid)
    c_values = np.full(num_grid, np.nan)
    c_min = np.inf
    worst_Qq = q_grid[0]

    for idx, q_cond in enumerate(q_grid):
        decrease = (1 - beta) * sigma_min_alpha0_square(q_cond, q)
        phi_q = half_normal_quantile(q_cond)
        # Worst-case error increase over the sigma grid.
        sigma_star, max_error = find_sigma_with_largest_error_increase_fast(
            qq=phi_q,
            sigma_min=sigma_min,
            sigma_max=sigma_max,
            num_points=num_points,
        )
        _ = sigma_star
        increase = beta * max_error
        c_val = -S_star_penalty + decrease - increase
        c_values[idx] = c_val
        if c_val < c_min:
            c_min = c_val
            worst_Qq = q_cond
        # Early exit once the contraction falls below target.
        if c_min < c_target:
            break

    # Contraction check.
    if c_min < c_target:
        return {
            "feasible": False,
            "reason": "c condition violated",
            "c_min": c_min,
            "worst_Qq": worst_Qq,
            "c_values": c_values,
            "Qq_grid": q_grid,
            "p_l": p_l,
            "p_l_c": p_l_c,
        }

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    # Upper tail over T iterations.
    failure_prob = 1.0 - (1.0 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "c_min": c_min,
        }

    return {
        "feasible": True,
        "c_min": c_min,
        "worst_Qq": worst_Qq,
        "c_values": c_values,
        "Qq_grid": q_grid,
        "p_l": p_l,
        "p_l_c": p_l_c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }
