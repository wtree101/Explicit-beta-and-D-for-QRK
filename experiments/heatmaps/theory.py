"""Map heatmap corruption models to their theoretical feasibility checks."""

from __future__ import annotations

from functools import partial

from qrk_adv.feasibility import (
    check_feasibility_conditions_C_sup_revised,
    check_feasibility_conditions_random_sup_revised,
)


def make_feasibility_check(
    corruption_type: str,
    feasibility_C_min: float = 0.0,
    feasibility_C_max: float = 100.0,
):
    """Return the theoretical check corresponding to a simulation model."""
    match corruption_type:
        case "adversarial":
            return None
        case "sup_c" | "oblivious_large":
            return partial(
                check_feasibility_conditions_C_sup_revised,
                num_grid_Q=2,
                C_min=feasibility_C_min,
                C_max=feasibility_C_max,
                num_points_C=20,
            )
        case "sup_rand":
            return partial(
                check_feasibility_conditions_random_sup_revised,
                num_grid_Q=20,
                num_points_C=50,
            )
        case _:
            raise ValueError(f"Unknown corruption_type: {corruption_type}")
