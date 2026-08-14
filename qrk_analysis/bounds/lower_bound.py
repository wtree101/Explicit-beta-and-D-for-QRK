"""
Lower bound on D: the subsample size required for the quantile estimator.

The lower bound comes from a Chernoff-type argument requiring that the
probability of a "bad" quantile event (i.e. Q_{q,k+1} deviating from the
true quantile q) is small enough to ensure overall convergence.

The key inequality is:
    ln(β/2) + ln(1/√(2D)) - D · D_KL(1-q+1/D || β/2) ≥ ln(1 - δ_s^{1/T})

The largest D satisfying this inequality is the lower bound we seek.
"""

import numpy as np
from ..core.divergence import DKL


def lower_bound_rough(delta_s: float, T: int, beta: float, q: float) -> float:
    """Rough upper bound on D from the simpler RHS formula.

    Computes (-ln(1 - δ_s^{1/T}) + ln(β/2)) / D_KL(1-q || β/2).
    This ignores the ln(1/√(2D)) correction term and serves as a quick
    first estimate before running the full search.

    Parameters
    ----------
    delta_s : float
        Allowed single-step failure probability (0 < delta_s < 1).
    T : int
        Number of iterations.
    beta : float
        Corruption fraction.
    q : float
        Quantile level.

    Returns
    -------
    float
        Rough upper bound on the feasible range of D.
    """
    numerator = -np.log(1 - delta_s ** (1 / T)) + np.log(beta / 2)
    denominator = DKL(1 - q, beta / 2)
    return numerator / denominator


def inequality_lhs(D: float, beta: float, q: float, delta_s: float, T: float) -> float:
    """Left-hand side of the lower-bound inequality evaluated at D.

    LHS(D) = ln(β/2) + ln(1/√(2D)) - D_KL(1-q+1/D || β/2) · D

    Parameters
    ----------
    D : float
        Candidate subsample size.
    beta, q, delta_s, T : float
        Problem parameters (delta_s and T are unused here but kept for
        a consistent call signature alongside Lower_bound).

    Returns
    -------
    float
        LHS value.
    """
    term1 = np.log(beta / 2)
    term2 = np.log(1 / np.sqrt(2 * D))
    p_arg = 1 - q + 1 / D
    term3 = -DKL(p_arg, beta / 2) * D
    return term1 + term2 + term3


def Lower_bound(
    beta: float,
    q: float,
    delta_s: float,
    T: float,
    D_max: int = 100,
    D_precision: float = 0.001,
) -> dict:
    """Find the largest integer D such that LHS(D) >= ln(1 - δ_s^{1/T}).

    Performs a linear scan from D_max down to D_min = ceil(1/q), stopping at
    the first D where the inequality holds.

    Parameters
    ----------
    beta : float
        Corruption fraction.
    q : float
        Quantile level.
    delta_s : float
        Single-step failure probability (0 < delta_s < 1).
    T : float
        Number of iterations.
    D_max : int
        Maximum D to consider.
    D_precision : float
        Unused (kept for API compatibility with callers that pass it).

    Returns
    -------
    dict
        Keys: largest_D, feasible, rhs, and diagnostic fields.
    """
    rhs = np.log(1 - delta_s ** (1 / T))
    D_low = int(1 / q)

    lhs_low = inequality_lhs(D_low, beta, q, delta_s, T)
    if lhs_low < rhs:
        return {
            "largest_D": None,
            "feasible": False,
            "reason": "No solution exists – LHS too small even at minimum D",
            "rhs": rhs,
            "lhs_at_min_D": lhs_low,
            "min_D_tested": D_low,
        }

    for D in range(int(D_max), 0, -1):
        p_arg = 1 - q + 1 / D
        if not (0 < p_arg < 1) or not (0 < beta / 2 < 1):
            continue
        lhs = inequality_lhs(D, beta, q, delta_s, T)
        if lhs >= rhs:
            return {
                "largest_D": D,
                "feasible": True,
                "rhs": rhs,
                "lhs_at_solution": lhs,
                "p_arg_at_solution": p_arg,
                "margin": lhs - rhs,
            }

    return {
        "largest_D": None,
        "feasible": False,
        "reason": "No solution found in linear search up to D_max",
        "rhs": rhs,
    }


def analyze_inequality_behavior(
    beta: float,
    q: float,
    delta_s: float,
    T: float,
    D_range=None,
) -> dict:
    """Evaluate LHS(D) across a range of D values for visualisation.

    Parameters
    ----------
    beta, q, delta_s, T : float
        Problem parameters.
    D_range : array-like, optional
        D values to evaluate.  Defaults to np.linspace(1, 100, 100).

    Returns
    -------
    dict
        D_values, lhs_values, rhs, and feasible_D arrays.
    """
    if D_range is None:
        D_range = np.linspace(1, 100, 100)

    rhs = np.log(1 - delta_s ** (1 / T))
    lhs_values = []
    valid_D = []

    for D in D_range:
        p_arg = 1 - q + 1 / D
        if 0 < p_arg < 1 and 0 < beta / 2 < 1:
            try:
                lhs = inequality_lhs(D, beta, q, delta_s, T)
                lhs_values.append(lhs)
                valid_D.append(D)
            except Exception:
                continue

    lhs_arr = np.array(lhs_values)
    valid_arr = np.array(valid_D)
    return {
        "D_values": valid_arr,
        "lhs_values": lhs_arr,
        "rhs": rhs,
        "feasible_D": valid_arr[lhs_arr >= rhs] if len(lhs_arr) > 0 else np.array([]),
    }
