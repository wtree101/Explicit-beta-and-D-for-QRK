"""
Demo: Largest feasible beta vs fixed noise C  (max over q)

Reproduces the notebook cell:
    for C_val in C_values:
        for q_val in q_values:  # binary search for largest beta
            res = check_feasibility_conditions_sign_id_C(1, beta_mid, 1_000_000, q_val, ..., C_val)
        max_beta = max(largest_betas)

Two variants are run side-by-side:
  • original – worst-case Phi(1-alpha') threshold  (check_feasibility_conditions_sign_id_C)
  • revised  – pointwise over Q_{q,k+1}            (check_feasibility_conditions_sign_id_C_revised)

Figures are saved to:
    figures/fixed_noise/beta_vs_C_orig.pdf
    figures/fixed_noise/beta_vs_C_revised.pdf
    figures/fixed_noise/beta_vs_C_comparison.pdf

Click the ▶ button in Cursor, or run directly:
    python demo_beta_vs_C.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
import matplotlib.pyplot as plt
from qrk_analysis.programs.program1 import plot_largest_beta_vs_C
from qrk_analysis.programs.utils import savefig

# ── parameters ────────────────────────────────────────────────────────────────
C_VALUES = np.linspace(0.01, 10, 20)   # use np.linspace(0,10,100) to match notebook exactly
Q_SWEEP  = np.linspace(0.1, 1, 10)
D        = np.inf
T        = 1
DELTA_F  = 1.0
SUBDIR   = "fixed_noise_comparison_2"

# ── run & plot ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    # print("=== Original: check_feasibility_conditions_sign_id_C ===")
    # C_arr_orig, betas_orig = plot_largest_beta_vs_C(
    #     C_values      = C_VALUES,
    #     q_sweep       = Q_SWEEP,
    #     D             = D,
    #     T             = T,
    #     delta_f       = DELTA_F,
    #     use_revised   = False,
    #     title         = "Largest feasible β vs C  (original, worst-case Qq)",
    #     save_filename = "beta_vs_C_orig.pdf",
    #     save_subdir   = SUBDIR,
    # )

    print("\n=== Revised: check_feasibility_conditions_sign_id_C_revised ===")
    C_arr_rev, betas_rev = plot_largest_beta_vs_C(
        C_values      = C_VALUES,
        q_sweep       = Q_SWEEP,
        D             = D,
        T             = T,
        delta_f       = DELTA_F,
        use_revised   = True,
        title         = "Largest feasible β vs C  (revised, pointwise Qq)",
        save_filename = "beta_vs_C_revised.pdf",
        save_subdir   = SUBDIR,
    )

    # # comparison overlay
    # plt.figure(figsize=(9, 5))
    # plt.plot(C_arr_orig, betas_orig, marker="o",  label="Original (worst-case Qq)")
    # plt.plot(C_arr_rev,  betas_rev,  marker="s", linestyle="--", label="Revised (pointwise Qq)")
    # plt.xlabel("C  (fixed noise magnitude)")
    # plt.ylabel("Largest β  (max over q)")
    # plt.title("Original vs Revised: largest feasible β vs C")
    # plt.legend()
    # plt.grid(True, alpha=0.3)
    # plt.tight_layout()
    # savefig("beta_vs_C_comparison.pdf", subdir=SUBDIR)
