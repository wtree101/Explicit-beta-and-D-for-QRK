"""Compatibility adapter for the historical Program 2 API."""

from __future__ import annotations

from collections.abc import Callable

from ..upper_bound import smallest_D


def program2_smallest_D_2(
    beta: float,
    T: int,
    q: float,
    delta_f: float,
    D_max: int = 100,
    D_precision: float = 1.0,
    c_target: float = 0.0,
    feasibility_check: Callable[..., dict] | None = None,
    num_grid: int = 50,
) -> dict:
    """Call canonical integer search and return the historical nested shape."""
    result = smallest_D(
        beta=beta,
        T=T,
        q=q,
        delta_f=delta_f,
        D_max=D_max,
        D_precision=D_precision,
        c_target=c_target,
        feasibility_check=feasibility_check,
        num_grid=num_grid,
    )
    if result["smallest_D"] is None:
        return {"smallest_D": None, "best_params": None}
    best_params = {
        "D": result["smallest_D"],
        "T": T,
        "beta": beta,
        "alpha_0": result["alpha_0"],
        "alpha_prime": result["alpha_prime"],
        "result": {
            "feasible": True,
            "c": result["c"],
            "p_l_c": result["p_l_c"],
            "p_u": result["p_u"],
            "failure_prob": result["failure_prob"],
        },
        "epsu": 1.0 - q - beta - result["alpha_prime"],
        "epsl": q - beta - result["alpha_0"],
    }
    return {"smallest_D": result["smallest_D"], "best_params": best_params}
