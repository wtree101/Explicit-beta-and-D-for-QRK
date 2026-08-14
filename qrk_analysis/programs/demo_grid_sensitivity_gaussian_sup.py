"""
Simple grid-sensitivity check for Gaussian supremum (sigma and Q grids).

This script isolates how the grid sizes used inside
check_feasibility_conditions_random_sup_revised affect the worst-case
contraction c_min for fixed (q, beta) without invoking the full
program1_largest_beta search.

Run:
    python demo_grid_sensitivity_gaussian_sup.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np

from qrk_analysis.core.quantile import half_normal_quantile, sigma_min_alpha0_square
from qrk_analysis.feasibility.check import find_sigma_with_largest_error_increase_fast

# -- parameters ----------------------------------------------------------------
Q = 0.6
BETA = 0.2
D = np.inf
SIGMA_MIN = 0.01
SIGMA_MAX = 10.0

NUM_POINTS_C_LIST = [1,5, 10, 20, 50, 100]
NUM_GRID_Q_LIST = [1,5, 10, 20, 50, 100]

BASE_NUM_POINTS_C = 20
BASE_NUM_GRID_Q = 20


def compute_c_min(num_grid_Q: int, num_points_C: int) -> tuple[float, float, float]:
    alpha_0 = Q - BETA
    alpha_prime = 1.0 - Q - BETA
    if alpha_0 < 0 or alpha_prime < 0:
        raise ValueError("Invalid (Q, BETA) leading to negative alpha bounds.")

    if D == np.inf:
        p_l = 0.0
    else:
        # Kept for completeness if D is later changed.
        from qrk_analysis.core.divergence import DKL
        p_l = np.exp(-DKL(Q, BETA + alpha_0) * D)

    S_star_penalty = (1 - BETA) * p_l * sigma_min_alpha0_square(alpha_0, Q)

    q_grid = np.linspace(alpha_0/(1 - BETA), 1.0 - alpha_prime/(1 - BETA), num_grid_Q)
    c_min = np.inf
    worst_Qq = q_grid[0]
    worst_sigma = np.nan

    for q_cond in q_grid:
        decrease = (1 - BETA) * sigma_min_alpha0_square(q_cond, Q)
        phi_q = half_normal_quantile(q_cond)
        sigma_star, max_error = find_sigma_with_largest_error_increase_fast(
            qq=phi_q,
            sigma_min=SIGMA_MIN,
            sigma_max=SIGMA_MAX,
            num_points=num_points_C,
        )
        c_val = -S_star_penalty + decrease - BETA * max_error
        if c_val < c_min:
            c_min = c_val
            worst_Qq = q_cond
            worst_sigma = sigma_star

    return c_min, worst_Qq, worst_sigma


if __name__ == "__main__":
    print("=== Grid sensitivity: vary sigma grid (num_points_C) ===")
    for num_points_C in NUM_POINTS_C_LIST:
        c_min, worst_Qq, worst_sigma = compute_c_min(BASE_NUM_GRID_Q, num_points_C)
        print(
            f"num_points_C={num_points_C:>3d} -> c_min={c_min:+.6g}, "
            f"worst_Qq={worst_Qq:.4f}, worst_sigma={worst_sigma:.4f}"
        )

    print("\n=== Grid sensitivity: vary Q grid (num_grid_Q) ===")
    for num_grid_Q in NUM_GRID_Q_LIST:
        c_min, worst_Qq, worst_sigma = compute_c_min(num_grid_Q, BASE_NUM_POINTS_C)
        print(
            f"num_grid_Q={num_grid_Q:>3d} -> c_min={c_min:+.6g}, "
            f"worst_Qq={worst_Qq:.4f}, worst_sigma={worst_sigma:.4f}"
        )
