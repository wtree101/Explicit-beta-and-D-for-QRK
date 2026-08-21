"""Search for feasible slack parameters at a fixed subsample size."""

from __future__ import annotations

from collections.abc import Callable

import numpy as np

from ..core.divergence import DKL
from ..debug import debug_log
from .check import check_feasibility_conditions

FeasibilityCheck = Callable[..., dict]


def _contraction(result: dict) -> float | None:
    value = result.get("c", result.get("c_min"))
    return None if value is None else float(value)


def _validate_search_inputs(beta: float, q: float, num_grid: int) -> None:
    if not 0.0 <= beta < min(q, 1.0 - q):
        raise ValueError("beta must satisfy 0 <= beta < min(q, 1-q)")
    if num_grid < 2:
        raise ValueError("num_grid must be at least 2")


def _largest_failure_feasible_alpha_prime(
    *,
    T: int,
    beta: float,
    D: float,
    q: float,
    delta_f: float,
    tolerance: float,
) -> float | None:
    """Return the largest strictly interior alpha' meeting the failure budget."""
    upper_bound = 1.0 - q - beta
    interior_offset = min(tolerance, upper_bound / 4.0)
    lower = interior_offset
    upper = upper_bound - interior_offset
    per_step_budget = (
        -np.expm1(np.log1p(-delta_f) / T) if T > 0 else delta_f
    )

    def is_feasible(alpha_prime: float) -> bool:
        if D == np.inf or beta == 0.0:
            return True
        p_u = beta * np.exp(-DKL(1.0 - q, beta + alpha_prime) * D)
        return bool(p_u <= per_step_budget)

    if not is_feasible(lower):
        return None
    if is_feasible(upper):
        return upper

    while upper - lower > tolerance:
        midpoint = (lower + upper) / 2.0
        if is_feasible(midpoint):
            lower = midpoint
        else:
            upper = midpoint
    return lower


def find_alpha_pair(
    T: int,
    beta: float,
    D: float,
    q: float,
    delta_f: float,
    num_grid: int = 50,
    c_target: float = 0.0,
    feasibility_check: FeasibilityCheck | None = None,
    maximize_c: bool = False,
    return_best_c_even_if_infeasible: bool = False,
    alpha_prime_tolerance: float = 1e-10,
) -> tuple[tuple[float, float] | None, dict | None]:
    """Find a strictly interior feasible ``(alpha_0, alpha_prime)`` pair."""
    _validate_search_inputs(beta, q, num_grid)
    if alpha_prime_tolerance <= 0.0:
        raise ValueError("alpha_prime_tolerance must be positive")
    if feasibility_check is None:
        feasibility_check = check_feasibility_conditions

    alpha_prime = _largest_failure_feasible_alpha_prime(
        T=T,
        beta=beta,
        D=D,
        q=q,
        delta_f=delta_f,
        tolerance=alpha_prime_tolerance,
    )
    failure_constraint_feasible = alpha_prime is not None
    if alpha_prime is None:
        if not return_best_c_even_if_infeasible:
            return None, None
        alpha_prime = min(alpha_prime_tolerance, (1.0 - q - beta) / 4.0)

    alpha_upper = q - beta
    alpha_offset = min(alpha_prime_tolerance, alpha_upper / 4.0)
    alpha_grid = np.linspace(alpha_offset, alpha_upper - alpha_offset, num_grid)
    best_pair: tuple[float, float] | None = None
    best_result: dict | None = None
    best_c = -np.inf

    for alpha_0 in alpha_grid:
        evaluation_target = -np.inf if return_best_c_even_if_infeasible else c_target
        result = feasibility_check(
            T,
            beta,
            D,
            q,
            float(alpha_0),
            alpha_prime,
            delta_f,
            c_target=evaluation_target,
        )
        c_value = _contraction(result)
        debug_log(
            f"D={D}, alpha_0={alpha_0:.8g}, alpha_prime={alpha_prime:.8g}, "
            f"c={c_value}, feasible={result.get('feasible')}"
        )

        if return_best_c_even_if_infeasible:
            if c_value is None or not np.isfinite(c_value) or c_value <= best_c:
                continue
            best_c = c_value
            best_pair = (float(alpha_0), alpha_prime)
            failure_satisfied = result.get("failure_prob", np.inf) <= delta_f
            best_result = {
                **result,
                "constraint_feasible": bool(
                    failure_constraint_feasible
                    and failure_satisfied
                    and c_value >= c_target
                ),
                "failure_constraint_feasible": failure_constraint_feasible,
                "meets_c_target": bool(c_value >= c_target),
                "requested_c_target": c_target,
            }
            continue

        if not result.get("feasible"):
            continue
        if not maximize_c:
            return (float(alpha_0), alpha_prime), result
        if c_value is not None and c_value > best_c:
            best_c = c_value
            best_pair = (float(alpha_0), alpha_prime)
            best_result = result

    return best_pair, best_result


def find_max_c_without_failure_constraint(
    T: int,
    beta: float,
    D: float,
    q: float,
    delta_f: float,
    num_grid: int = 50,
    feasibility_check: FeasibilityCheck | None = None,
) -> tuple[tuple[float, float] | None, dict | None]:
    """Maximize contraction on the asymptotic alpha' boundary."""
    _validate_search_inputs(beta, q, num_grid)
    if feasibility_check is None:
        feasibility_check = check_feasibility_conditions

    alpha_prime = 1.0 - q - beta
    best_pair: tuple[float, float] | None = None
    best_result: dict | None = None
    best_c = -np.inf
    for alpha_0 in np.linspace(0.0, q - beta, num_grid)[1:]:
        result = feasibility_check(
            T,
            beta,
            D,
            q,
            float(alpha_0),
            alpha_prime,
            delta_f,
            c_target=-np.inf,
            enforce_failure_probability=False,
        )
        c_value = _contraction(result)
        if c_value is None or not np.isfinite(c_value) or c_value <= best_c:
            continue
        best_c = c_value
        best_pair = (float(alpha_0), alpha_prime)
        best_result = {
            **result,
            "c_max": best_c,
            "alpha_0": best_pair[0],
            "alpha_prime": best_pair[1],
            "failure_constraint_enforced": False,
        }
    return best_pair, best_result


# Compatibility name used by the original qrk_analysis programs.
find_optimal_alpha_pair_2 = find_alpha_pair
