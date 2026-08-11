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

    Returns
    -------
    tuple[tuple | None, dict | None]
        ((alpha_0, alpha_prime), result_dict) or (None, None) if infeasible.
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
    if beta * np.exp(-DKL(1 - q, beta + ap_lo) * D) > p_u_max:
        return None, None   # infeasible even at ap = 0

    # p_u increases with alpha_prime, so bisection is valid.
    for _ in range(20):    
        # 20 bisection steps. Too many steps (like 48) may cause numerical instability.
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
        result = feasibility_check(
            T, beta, D, q, alpha_0, alpha_prime, delta_f, c_target=c_target
        )
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
