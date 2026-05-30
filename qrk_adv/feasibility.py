"""Feasibility condition checks for QRK convergence analysis.

Variants in this module:

  check_feasibility
      Preliminary adversarial (Massart) noise.
  check_feasibility_conditions_random_sup_revised
      Oblivious Gaussian noise; supremum over sigma on a grid.
  check_feasibility_conditions_C_sup_revised
      Fixed noise magnitude; supremum over C on a grid (pointwise over Q).
"""

# Module: feasibility checks for adversarial, fixed-C, and Gaussian-noise variants.

import numpy as np
from .divergence import DKL
from .quantile import half_normal_quantile, sigma_min_alpha0_square
from .noise import (
    error_increased_C_3,
    error_increased_Gaussian_noise,
    find_sigma_with_largest_error_increase_fast,
    find_C_with_largest_error_increase_fast,
)
from .debug import debug_log

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
    num_grid_Q: int = 2, #c seems to be increasing over q, so num_grid_Q=1 also works
    sigma_min: float = 0.01,
    sigma_max: float = 10.0, #safe for alpha' > 0.00001; may need to set larger if return smaller alpha'. 
    num_points_C: int = 20, #seems to stable after num_points_C>10
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
    num_grid_Q : int
        Number of grid points for the conditional quantile sweep. Grid search to find inf over \tilde q in [alpha_0, 1 - alpha_prime]).
    sigma_min, sigma_max : float
        Search range for sigma in the Gaussian noise model.
    num_points_C : int
        Number of grid points for the sigma sweep. Grid search to find sup over sigma in [sigma_min, sigma_max].
    
    Larger num_grid_Q and num_points_C and larger regions (sigma_min, sigma_max) yield more accurate results but increase runtime.  Adjust as needed for a speed-accuracy tradeoff. Possible future improvement: replace the grid search with a more efficient optimization method.

    c_target : float
        Minimum required contraction rate.
    """
    debug_log(f"num_grid_Q={num_grid_Q}, num_points_C={num_points_C}")
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
    S_star_penalty = (1 - beta) * p_l * sigma_min_alpha0_square(alpha_0/(1.0 - beta), q)

    # Sweep conditional quantiles in the admissible interval.
    q_grid = np.linspace(alpha_0/(1.0 - beta), 1.0 - alpha_prime/(1.0 - beta), num_grid_Q)
    debug_log(f"q_grid: {q_grid}")
    c_values = np.full(num_grid_Q, np.nan)
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
            num_points=num_points_C,
        )
        _ = sigma_star
        # print(max_error, beta)
        increase = beta * max_error

        c_val = -S_star_penalty + decrease - increase
        # print(f"c_val: {c_val}, c_min: {c_min}")
        debug_log(f"S_star_penalty: {S_star_penalty}, decrease: {decrease}, increase: {increase}")
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


def check_feasibility_conditions_C_sup_revised(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    num_grid_Q: int = 2,
    C_min: float = 0.0,
    C_max: float = 20.0,
    num_points_C: int = 20,
    c_target: float = 0.0,
) -> dict:
    """Revised fixed-noise check: supremum over C on a grid at each Q_{q,k+1}.

    Mirrors :func:`check_feasibility_conditions_random_sup_revised`, but the
    increase term uses :func:`~qrk_adv.noise.find_C_with_largest_error_increase_fast`
    to approximate sup_{C in [C_min, C_max]} error_increased_C_3(Phi(Qq), C).

    Pointwise net contraction at Qq:
        c(Qq) = - S*_failure_penalty
                + (1 - beta) * sigma_min_alpha0_square(Qq, q)
                - beta * sup_C error_increased_C_3(Phi(Qq), C)

    Feasibility requires min_{Qq} c(Qq) >= c_target.

    Parameters
    ----------
    T : int
        Number of QRK iterations.
    beta : float
        Corruption fraction.
    D : float
        Subsample size for the quantile (use ``np.inf`` if D is inactive).
    q : float
        Quantile level.
    alpha_0 : float
        Lower slack; must satisfy 0 <= alpha_0 <= q - beta.
    alpha_prime : float
        Upper slack; must satisfy 0 <= alpha_prime <= 1 - q - beta.
    delta_f : float
        Allowed total failure probability over T iterations.
    num_grid_Q : int
        Grid points for the conditional quantile sweep over
        [alpha_0/(1-beta), 1 - alpha_prime/(1-beta)].
    C_min, C_max : float
        Search range for the fixed-noise magnitude C.
    num_points_C : int
        Grid size inside :func:`~qrk_adv.noise.find_C_with_largest_error_increase_fast`.
    c_target : float
        Minimum required pointwise contraction (applied to c_min).

    Returns
    -------
    dict
        feasible, c_min, worst_Qq, worst_Qq_C_star, c_values, Qq_grid,
        p_l, p_l_c, p_u, failure_prob; ``reason`` when infeasible.
    """
    debug_log(f"num_grid_Q={num_grid_Q}, num_points_C={num_points_C}")
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
    S_star_penalty = (1 - beta) * p_l * sigma_min_alpha0_square(alpha_0 / (1.0 - beta), q)

    q_grid = np.linspace(alpha_0 / (1.0 - beta), 1.0 - alpha_prime / (1.0 - beta), num_grid_Q)
    debug_log(f"q_grid: {q_grid}")
    c_values = np.full(num_grid_Q, np.nan)
    c_min = np.inf
    worst_Qq = q_grid[0]
    worst_Qq_C_star = np.nan

    for idx, q_cond in enumerate(q_grid):
        decrease = (1 - beta) * sigma_min_alpha0_square(q_cond, q)
        phi_q = half_normal_quantile(q_cond)
        C_star, max_error = find_C_with_largest_error_increase_fast(
            qq=phi_q,
            C_min=C_min,
            C_max=C_max,
            num_points=num_points_C,
        )
        increase = beta * max_error
        c_val = -S_star_penalty + decrease - increase
        debug_log(f"S_star_penalty: {S_star_penalty}, decrease: {decrease}, increase: {increase}")
        c_values[idx] = c_val
        if c_val < c_min:
            c_min = c_val
            worst_Qq = q_cond
            worst_Qq_C_star = C_star
        if c_min < c_target:
            break

    if c_min < c_target:
        return {
            "feasible": False,
            "reason": "c condition violated",
            "c_min": c_min,
            "worst_Qq": worst_Qq,
            "worst_Qq_C_star": worst_Qq_C_star,
            "c_values": c_values,
            "Qq_grid": q_grid,
            "p_l": p_l,
            "p_l_c": p_l_c,
        }

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
            "c_min": c_min,
            "worst_Qq": worst_Qq,
            "worst_Qq_C_star": worst_Qq_C_star,
        }

    return {
        "feasible": True,
        "c_min": c_min,
        "worst_Qq": worst_Qq,
        "worst_Qq_C_star": worst_Qq_C_star,
        "c_values": c_values,
        "Qq_grid": q_grid,
        "p_l": p_l,
        "p_l_c": p_l_c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }
