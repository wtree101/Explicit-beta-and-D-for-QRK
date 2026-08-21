"""
Grid-sensitivity check for fixed-noise supremum (C and Q grids).

Isolates how ``num_points_C`` and ``num_grid_Q`` inside
:func:`~qrk_analysis.feasibility.check.check_feasibility_conditions_C_sup_revised`
affect c_min at fixed (q, beta).

Run:
    ~/anaconda3/bin/python3 qrk_analysis/programs/demo_grid_sensitivity_C_sup.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from qrk_analysis.core.quantile import half_normal_quantile, sigma_min_alpha0_square
from qrk_analysis.core.divergence import DKL
from qrk_analysis.noise.fixed import find_C_with_largest_error_increase_fast

Q = 0.6
BETA = 0.2
D = np.inf
C_MIN = 0.0
C_MAX = 10.0

NUM_POINTS_C_LIST = [1, 5, 10, 20, 50]
NUM_GRID_Q_LIST = [1, 5, 10, 20]
BASE_NUM_POINTS_C = 20
BASE_NUM_GRID_Q = 10


def compute_c_min(num_grid_Q: int, num_points_C: int) -> tuple[float, float, float]:
    alpha_0 = Q - BETA
    alpha_prime = 1.0 - Q - BETA
    if alpha_0 < 0 or alpha_prime < 0:
        raise ValueError("Invalid (Q, BETA) leading to negative alpha bounds.")

    p_l = 0.0 if D == np.inf else np.exp(-DKL(Q, BETA + alpha_0) * D)
    S_star_penalty = (1 - BETA) * p_l * sigma_min_alpha0_square(alpha_0, Q)

    q_grid = np.linspace(alpha_0, 1.0 - alpha_prime, num_grid_Q)
    c_min = np.inf
    worst_Qq = q_grid[0]
    worst_C = np.nan

    for q_cond in q_grid:
        decrease = (1 - BETA) * sigma_min_alpha0_square(q_cond, Q)
        phi_q = half_normal_quantile(q_cond)
        C_star, max_error = find_C_with_largest_error_increase_fast(
            qq=phi_q, C_min=C_MIN, C_max=C_MAX, num_points=num_points_C
        )
        c_val = -S_star_penalty + decrease - BETA * max_error
        if c_val < c_min:
            c_min = c_val
            worst_Qq = q_cond
            worst_C = C_star

    return c_min, worst_Qq, worst_C


if __name__ == "__main__":
    print("=== Grid sensitivity: vary C grid (num_points_C) ===")
    for num_points_C in NUM_POINTS_C_LIST:
        c_min, worst_Qq, worst_C = compute_c_min(BASE_NUM_GRID_Q, num_points_C)
        print(
            f"num_points_C={num_points_C:>3d} -> c_min={c_min:+.6g}, "
            f"worst_Qq={worst_Qq:.4f}, worst_C={worst_C:.4f}"
        )

    print("\n=== Grid sensitivity: vary Q grid (num_grid_Q) ===")
    for num_grid_Q in NUM_GRID_Q_LIST:
        c_min, worst_Qq, worst_C = compute_c_min(num_grid_Q, BASE_NUM_POINTS_C)
        print(
            f"num_grid_Q={num_grid_Q:>3d} -> c_min={c_min:+.6g}, "
            f"worst_Qq={worst_Qq:.4f}, worst_C={worst_C:.4f}"
        )
