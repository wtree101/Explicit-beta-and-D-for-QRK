"""Compute the upper bound on D (smallest feasible subsample size).

Primary entry point
-------------------
    smallest_D(beta, T, q, delta_f, ...)

It binary-searches for the smallest integer D such that the adversarial
feasibility conditions can be simultaneously satisfied for some (alpha_0,
alpha_prime) pair.
"""

# Module: binary search for smallest feasible subsample size.

from .search import find_alpha_pair


def smallest_D(
    beta: float,
    T: int,
    q: float,
    delta_f: float,
    D_max: int = 500,
    D_precision: float = 1.0,
    c_target: float = 0.0,
    num_grid: int = 50,
    feasibility_check=None,
) -> dict:
    """Binary search for the smallest D satisfying adversarial feasibility.

    Parameters
    ----------
    beta : float
        Corruption fraction (e.g. 0.05).
    T : int
        Number of QRK iterations (e.g. 10_000).
    q : float
        Quantile level (e.g. 0.75).  Must satisfy beta < q < 1 - beta.
    delta_f : float
        Allowed total failure probability (e.g. 0.1).
    D_max : int
        Upper search ceiling on D.  Raise if the result hits this ceiling.
    D_precision : float
        Binary-search stopping tolerance on D.  Result is accurate to within
        +/- D_precision.  Default 1 (integer precision).
    c_target : float, optional
        Minimum required net contraction rate c.  Default 0.0.
    num_grid : int
        Number of grid points used for the (alpha_0, alpha_prime) search.

    Returns
    -------
    dict with keys:
        smallest_D  : float or None (None means infeasible within D_max)
        alpha_0     : float  (optimal lower slack)
        alpha_prime : float  (optimal upper slack)
        c           : float  (net contraction at the optimal parameters)
        failure_prob: float
        hit_ceiling : bool   (True if smallest_D == D_max; try a larger D_max)
    """
    D_low, D_high = 1.0, float(D_max)
    best_D = None
    best_params = None

    # Binary search in D; each step runs a feasibility search.
    while D_high - D_low > D_precision:
        D_mid = (D_low + D_high) / 2.0
        alpha_pair, result = find_alpha_pair(
            T, beta, D_mid, q, delta_f,
            num_grid=num_grid, c_target=c_target,
            feasibility_check=feasibility_check,
        )
        if alpha_pair is not None:
            c_value = result.get("c", result.get("c_min"))
            best_D = D_mid
            best_params = {
                "alpha_0": alpha_pair[0],
                "alpha_prime": alpha_pair[1],
                "c": c_value,
                "p_l_c": result["p_l_c"],
                "p_u": result["p_u"],
                "failure_prob": result["failure_prob"],
            }
            D_high = D_mid
        else:
            D_low = D_mid

    # Infeasible within the D_max ceiling.
    if best_D is None:
        return {"smallest_D": None, "hit_ceiling": False}

    return {
        "smallest_D": best_D,
        "hit_ceiling": abs(best_D - D_max) < 2 * D_precision,
        **best_params,
    }


