"""
Demo: largest feasible beta vs q – fixed noise with supremum over C.

Runs :func:`~qrk_analysis.feasibility.check.check_feasibility_conditions_C_sup_revised`
inside :func:`~qrk_analysis.programs.program1.program1_largest_beta`.

Run:
    ~/anaconda3/bin/python3 qrk_analysis/programs/demo_beta_max_q_C_sup.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qrk_analysis.feasibility.check import check_feasibility_conditions_C_sup_revised
from qrk_analysis.programs.program1 import program1_largest_beta

Q_SWEEP = np.linspace(0.1, 0.95, 18)
D = np.inf
T = 1
DELTA_F = 1.0
C_MIN = 0.0
C_MAX = 20.0
NUM_GRID_Q = 10
C_GRID_SWEEP = [200]
FIG_SIZE = 6

CHECK_KWARGS_BASE = {
    "num_grid_Q": NUM_GRID_Q,
    "C_min": C_MIN,
    "C_max": C_MAX,
}


if __name__ == "__main__":
    print("=== Parameter settings ===")
    print(f"Q_SWEEP len={len(Q_SWEEP)}, D={D}, T={T}, DELTA_F={DELTA_F}")
    print(f"C_MIN={C_MIN}, C_MAX={C_MAX}, NUM_GRID_Q={NUM_GRID_Q}")
    print("===========================")

    betas_by_grid = {}
    for num_points_C in C_GRID_SWEEP:
        print(f"\n=== Sweeping q for num_points_C={num_points_C} ===")
        check_kwargs = {**CHECK_KWARGS_BASE, "num_points_C": num_points_C}
        betas = []
        for idx, qv in enumerate(Q_SWEEP, start=1):
            print(f"[{idx}/{len(Q_SWEEP)}] q={qv:.4f} ...")
            beta_val = program1_largest_beta(
                qv,
                D,
                T,
                DELTA_F,
                feasibility_check=check_feasibility_conditions_C_sup_revised,
                check_kwargs=check_kwargs,
            )
            betas.append(beta_val)
            print(f"  -> beta={beta_val}")
        betas_by_grid[num_points_C] = betas

        valid = [(qv, b) for qv, b in zip(Q_SWEEP, betas) if b is not None]
        if valid:
            best_q, best_beta = max(valid, key=lambda item: item[1])
            print(f"Largest beta: {best_beta:.6g} at q={best_q:.6g}")

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    for num_points_C, betas in betas_by_grid.items():
        betas_arr = np.array([np.nan if b is None else b for b in betas], dtype=float)
        ax.plot(Q_SWEEP, betas_arr, "o-", lw=1.4, label=f"num_points_C={num_points_C}")
    ax.set_xlabel(r"$q$", fontsize=12)
    ax.set_ylabel(r"Largest feasible $\beta$", fontsize=12)
    ax.set_title(
        fr"Worst-case fixed noise: $\beta^*(q)$, $C\in[{C_MIN},{C_MAX}]$",
        fontsize=12,
    )
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)
    plt.tight_layout()
    out_dir = os.path.join(os.path.dirname(__file__), "..", "figures", "fixed_noise")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "beta_vs_q_C_sup_grid_sweep.png")
    fig.savefig(out_path, dpi=150)
    print(f"Figure saved to {out_path}")
    plt.close(fig)
