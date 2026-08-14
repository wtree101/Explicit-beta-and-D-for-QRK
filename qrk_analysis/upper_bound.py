"""Compute the smallest certified integer quantile-subsample size."""

from __future__ import annotations

from .feasibility.search import FeasibilityCheck, find_alpha_pair


def smallest_D(
    beta: float,
    T: int,
    q: float,
    delta_f: float,
    D_max: int = 500,
    D_precision: float = 0.1,
    c_target: float = 0.0,
    num_grid: int = 50,
    feasibility_check: FeasibilityCheck | None = None,
) -> dict:
    """Return the smallest feasible integer ``D`` up to ``D_max``.

    ``D_precision`` is retained for compatibility with ``qrk_adv``. Integer
    bisection makes a sub-unit continuous tolerance unnecessary.
    """
    if D_max < 1:
        raise ValueError("D_max must be at least 1")
    if D_precision <= 0.0:
        raise ValueError("D_precision must be positive")

    def evaluate(D: int) -> tuple[tuple[float, float] | None, dict | None]:
        return find_alpha_pair(
            T,
            beta,
            float(D),
            q,
            delta_f,
            num_grid=num_grid,
            c_target=c_target,
            feasibility_check=feasibility_check,
        )

    ceiling_pair, ceiling_result = evaluate(D_max)
    if ceiling_pair is None or ceiling_result is None:
        return {
            "smallest_D": None,
            "hit_ceiling": False,
            "search_exhausted": True,
        }

    low = 1
    high = D_max
    best_pair = ceiling_pair
    best_result = ceiling_result
    while low < high:
        midpoint = (low + high) // 2
        pair, result = evaluate(midpoint)
        if pair is None or result is None:
            low = midpoint + 1
        else:
            high = midpoint
            best_pair = pair
            best_result = result

    if low != D_max:
        final_pair, final_result = evaluate(low)
        if final_pair is not None and final_result is not None:
            best_pair = final_pair
            best_result = final_result

    c_value = best_result.get("c", best_result.get("c_min"))
    return {
        "smallest_D": low,
        "hit_ceiling": low == D_max,
        "search_exhausted": False,
        "alpha_0": best_pair[0],
        "alpha_prime": best_pair[1],
        "c": c_value,
        "p_l_c": best_result["p_l_c"],
        "p_u": best_result["p_u"],
        "failure_prob": best_result["failure_prob"],
    }
