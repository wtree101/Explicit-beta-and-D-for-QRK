"""Search for a feasible (alpha_0, alpha_prime) pair at fixed D.

Strategy
--------
1. Pin alpha_prime to the largest value that satisfies the per-step upper
   Chernoff bound  p_u <= 1 - (1 - delta_f)^{1/T}.
2. Grid-search over alpha_0 and return the first feasible (alpha_0, alpha_prime)
   found by default. Optionally scan all feasible alpha_0 values and return the
   one with the largest c.
"""

# Module: grid/bisection search for feasible slack parameters.

import numpy as np
from .divergence import DKL
from .feasibility import check_feasibility


def find_max_c_without_failure_constraint(
    T: int,
    beta: float,
    D: float,
    q: float,
    delta_f: float,
    num_grid: int = 50,
    feasibility_check=None,
) -> tuple:
    """Maximize contraction over alpha_0 with alpha_prime at its upper bound.

    The failure probability is still computed for diagnostics, but it does not
    constrain the search. This exploratory quantity is not an estimate of the
    experiment's actual contraction rate and must not be used as an empirical
    success criterion or as a high-probability theoretical D bound.
    """
    if not 0.0 <= beta < min(q, 1.0 - q):
        raise ValueError("beta must satisfy 0 <= beta < min(q, 1-q)")
    if num_grid < 2:
        raise ValueError("num_grid must be at least 2")

    if feasibility_check is None:
        feasibility_check = check_feasibility

    alpha_prime = 1.0 - q - beta
    best_pair = None
    best_result = None
    best_c = -np.inf

    for alpha_0 in np.linspace(0.0, q - beta, num_grid):
        if alpha_0 <= 0.0:
            continue
        result = feasibility_check(
            T,
            beta,
            D,
            q,
            alpha_0,
            alpha_prime,
            delta_f,
            c_target=-np.inf,
            enforce_failure_probability=False,
        )
        c_value = result.get("c", result.get("c_min"))
        if c_value is None or not np.isfinite(c_value) or c_value <= best_c:
            continue
        best_c = float(c_value)
        best_pair = (float(alpha_0), float(alpha_prime))
        best_result = {
            **result,
            "c_max": best_c,
            "alpha_0": best_pair[0],
            "alpha_prime": best_pair[1],
            "failure_constraint_enforced": False,
        }

    return best_pair, best_result


def find_alpha_pair(
    T: int,
    beta: float,
    D: float,
    q: float,
    delta_f: float,
    num_grid: int = 50,
    c_target: float = 0.0,
    feasibility_check=None,
    maximize_c: bool = False,
    return_best_c_even_if_infeasible: bool = False,
) -> tuple:
    """Find a feasible (alpha_0, alpha_prime) pair for the adversarial setting.

    Parameters
    ----------
    T : int
        Number of iterations.
    beta : float
        Corruption fraction.
    D : float
        Subsample size.
    q : float
        Quantile level.
    delta_f : float
        Allowed total failure probability.
    num_grid : int
        Number of grid points for alpha_0 search.
    c_target : float, optional
        Minimum required contraction rate.  Default 0.0.
    maximize_c : bool, optional
        If True, scan all feasible alpha_0 values and return the pair with the
        largest contraction c.  If False, return the first feasible pair.
    return_best_c_even_if_infeasible : bool, optional
        If True, return the alpha_0 with the largest computed c even when c is
        below c_target or the failure-probability constraint is not satisfied.
        When the failure constraint has no feasible alpha_prime, alpha_prime=0
        is used. The returned result contains ``constraint_feasible`` and
        ``failure_constraint_feasible`` flags.

    Returns
    -------
    tuple[tuple | None, dict | None]
        ((alpha_0, alpha_prime), result_dict), or (None, None) if infeasible
        unless ``return_best_c_even_if_infeasible`` is enabled.
    """
    #Step 1: pin alpha_prime by the tightest p_u budget
    #old method: grid search
    # p_u_max = 1.0 - (1.0 - delta_f) ** (1.0 / T) if T > 0 else delta_f

    # ap_candidates = np.linspace(0, 1 - q - beta, num_grid)
    # feasible_aps = [
    #     ap for ap in ap_candidates
    #     if beta * np.exp(-DKL(1 - q, beta + ap) * D) <= p_u_max
    # ]
    # if not feasible_aps:
    #     return None, None
    # alpha_prime = max(feasible_aps)
    # print(f"feasible_aps: {feasible_aps}")
    # print(f"alpha_prime: {alpha_prime}")
    # Step 1: binary search for the largest alpha_prime satisfying the p_u budget.
    #p_u = beta * exp(-DKL(1-q, beta+ap) * D) is increasing in ap, so the
    #feasible set is [0, alpha_prime*]; bisect to find alpha_prime*.
   
     # optimize: can use binary search to find the largest alpha_prime that satisfies the p_u budget
    delta_f_safe = delta_f  #or - 1e-7 to avoid numerical instability
    p_u_max = 1.0 - (1.0 - delta_f_safe) ** (1.0 / T) if T > 0 else delta_f_safe

    ap_lo, ap_hi = 0.0, 1.0 - q - beta
    failure_constraint_feasible = (
        beta * np.exp(-DKL(1 - q, beta + ap_lo) * D) <= p_u_max
    )
    if not failure_constraint_feasible and not return_best_c_even_if_infeasible:
        return None, None   # infeasible even at ap = 0

    # p_u increases with alpha_prime, so bisection is valid.
    if failure_constraint_feasible:
        for _ in range(20):
            # 20 bisection steps. Too many steps may cause numerical instability.
            ap_mid = (ap_lo + ap_hi) / 2.0
            if beta * np.exp(-DKL(1 - q, beta + ap_mid) * D) <= p_u_max:
                ap_lo = ap_mid
            else:
                ap_hi = ap_mid
        

    alpha_prime = ap_lo
    # print(f"p_u_max: {p_u_max}, alpha_prime: {alpha_prime}")
    # print(f"alpha_prime: {alpha_prime}")
    # Step 2: grid-search over alpha_0
    if feasibility_check is None:
        feasibility_check = check_feasibility

    best_pair = None
    best_result = None
    best_c = -np.inf

    for alpha_0 in np.linspace(0, q - beta, num_grid):
        if alpha_0 <= 0.0:
            continue
        evaluation_c_target = -np.inf if return_best_c_even_if_infeasible else c_target
        result = feasibility_check(
            T, beta, D, q, alpha_0, alpha_prime, delta_f,
            c_target=evaluation_c_target,
        )
        c_value = result.get("c", result.get("c_min"))

        if return_best_c_even_if_infeasible and c_value is not None:
            if c_value > best_c:
                best_c = c_value
                best_pair = (alpha_0, alpha_prime)
                result_failure_feasible = (
                    result.get("failure_prob", np.inf) <= delta_f
                )
                best_result = {
                    **result,
                    "constraint_feasible": bool(
                        failure_constraint_feasible
                        and result_failure_feasible
                        and c_value >= c_target
                    ),
                    "failure_constraint_feasible": failure_constraint_feasible,
                    "meets_c_target": bool(c_value >= c_target),
                    "requested_c_target": c_target,
                }
            continue

        if result["feasible"]:
            if not maximize_c:
                return (alpha_0, alpha_prime), result

            c_value = result.get("c", result.get("c_min", -np.inf))
            if c_value > best_c:
                best_c = c_value
                best_pair = (alpha_0, alpha_prime)
                best_result = result

    if best_pair is not None:
        return best_pair, best_result

    return None, None
