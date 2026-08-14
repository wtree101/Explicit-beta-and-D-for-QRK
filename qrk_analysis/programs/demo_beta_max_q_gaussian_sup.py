"""
Demo: largest feasible beta and corresponding q (Gaussian worst-sigma).

Runs the revised Gaussian supremum check over sigma and reports the
max beta and its q value.

Run:
    python demo_beta_max_q_gaussian_sup.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qrk_analysis.feasibility.check import check_feasibility_conditions_random_sup_revised
from qrk_analysis.programs.program1 import program1_largest_beta

# -- parameters ----------------------------------------------------------------
Q_SWEEP = np.linspace(0.2, 0.9, 3)  # use denser grid if needed
D = np.inf
T = 1
DELTA_F = 1.0
SIGMA_MIN = 0.01
SIGMA_MAX = 10
SIGMA_GRID = 20
NUM_GRID_Q = 1
SIGMA_GRID_SWEEP = [1, 5, 10, 100]
FIG_SIZE = 6


if __name__ == "__main__":
    print("=== Parameter settings ===")
    print(f"Q_SWEEP={Q_SWEEP}")
    print(f"D={D}, T={T}, DELTA_F={DELTA_F}")
    print(f"SIGMA_MIN={SIGMA_MIN}, SIGMA_MAX={SIGMA_MAX}")
    print(f"NUM_GRID_Q={NUM_GRID_Q}, NUM_POINTS_C={SIGMA_GRID}")
    print("===========================")
    betas_by_grid = {}
    for num_points_C in SIGMA_GRID_SWEEP:
        print(f"\n=== Sweeping q for num_points_C={num_points_C} ===")
        betas = []
        for idx, qv in enumerate(Q_SWEEP, start=1):
            print(f"[{idx}/{len(Q_SWEEP)}] q={qv:.4f} ...")
            beta_val = program1_largest_beta(
                qv,
                D,
                T,
                DELTA_F,
                feasibility_check=check_feasibility_conditions_random_sup_revised,
                check_kwargs={
                    "sigma_min": SIGMA_MIN,
                    "sigma_max": SIGMA_MAX,
                    "num_grid_Q": NUM_GRID_Q,
                    "num_points_C": num_points_C,
                },
            )
            betas.append(beta_val)
            print(f"  -> beta={beta_val}")

        betas_by_grid[num_points_C] = betas

        valid = [(qv, b) for qv, b in zip(Q_SWEEP, betas) if b is not None]
        if not valid:
            print("No feasible beta found in q sweep.")
        else:
            best_q, best_beta = max(valid, key=lambda item: item[1])
            print(f"Largest beta: {best_beta:.6g}")
            print(f"Argmax q: {best_q:.6g}")

    # Plot multiple beta(q) curves for different sigma grid sizes.
    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    for num_points_C, betas in betas_by_grid.items():
        betas_arr = np.array([np.nan if b is None else b for b in betas], dtype=float)
        ax.plot(Q_SWEEP, betas_arr, "o-", lw=1.4, label=f"num_points_C={num_points_C}")
    ax.set_xlabel(r"$q$", fontsize=12)
    ax.set_ylabel(r"Largest feasible $\beta$", fontsize=12)
    ax.set_title("Gaussian worst-sigma: $\\beta^*(q)$", fontsize=12)
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures", "gaussian_noise")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "beta_vs_q_sup_sigma_grid_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")
    plt.close(fig)
