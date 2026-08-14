"""
Demo: Largest feasible beta vs Gaussian noise sigma  (max over q)

Reuses plot_largest_beta_vs_sigma in program1.py.

Two variants are run side-by-side:
  - original: worst-case threshold (check_feasibility_conditions_random)
  - revised : pointwise over Q_{q,k+1} (check_feasibility_conditions_random_revised)

Figures are saved to:
    figures/gaussian_noise/beta_vs_sigma_orig.pdf
    figures/gaussian_noise/beta_vs_sigma_revised.pdf
    figures/gaussian_noise/beta_vs_sigma_comparison.pdf

Click the ▶ button in Cursor, or run directly:
    python demo_beta_vs_sigma.py
"""

import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt
import numpy as np

from qrk_analysis.programs.program1 import plot_largest_beta_vs_sigma
from qrk_analysis.programs.utils import savefig

# -- parameters ----------------------------------------------------------------
SIGMA_VALUES = np.linspace(0.1, 10, 10)  # use denser grid if needed
Q_SWEEP      = np.linspace(0.1, 1, 10)
D            = 1_000_000 # not inf -- not that large
T            = 1
DELTA_F      = 1.0
SUBDIR       = "gaussian_noise"

# -- run & plot ----------------------------------------------------------------
if __name__ == "__main__":
    print("=== Original: check_feasibility_conditions_random ===")
    sigma_arr_orig, betas_orig = plot_largest_beta_vs_sigma(
        sigma_values  = SIGMA_VALUES,
        q_sweep       = Q_SWEEP,
        D             = D,
        T             = T,
        delta_f       = DELTA_F,
        use_revised   = False,
        title         = "Largest feasible β vs σ  (original, worst-case Qq)",
        save_filename = "beta_vs_sigma_orig.pdf",
        save_subdir   = SUBDIR,
    )

    print("\n=== Revised: check_feasibility_conditions_random_revised ===")
    sigma_arr_rev, betas_rev = plot_largest_beta_vs_sigma(
        sigma_values  = SIGMA_VALUES,
        q_sweep       = Q_SWEEP,
        D             = D,
        T             = T,
        delta_f       = DELTA_F,
        use_revised   = True,
        title         = "Largest feasible β vs σ  (revised, pointwise Qq)",
        save_filename = "beta_vs_sigma_revised.pdf",
        save_subdir   = SUBDIR,
    )

    # comparison overlay
    plt.figure(figsize=(9, 5))
    plt.plot(sigma_arr_orig, betas_orig, marker="o",  label="Original (worst-case Qq)")
    plt.plot(sigma_arr_rev,  betas_rev,  marker="s", linestyle="--", label="Revised (pointwise Qq)")
    plt.xlabel("sigma  (Gaussian noise std)")
    plt.ylabel("Largest β  (max over q)")
    plt.title("Original vs Revised: largest feasible β vs σ")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("beta_vs_sigma_comparison.pdf", subdir=SUBDIR)
