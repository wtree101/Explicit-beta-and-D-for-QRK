"""
Demo: Largest feasible beta vs q  (adversarial noise)

Two variants are run side-by-side:
  - original  – separate worst-case thresholds for increase / decrease
                (check_feasibility_conditions)
  - revised   – both terms share the same quantile threshold tilde_q,
                pointwise over Q_{q,k+1}
                (check_feasibility_conditions_adversarial_revised)

Figures are saved to:
    figures/adversarial/beta_vs_q_orig.pdf
    figures/adversarial/beta_vs_q_revised.pdf
    figures/adversarial/beta_vs_q_comparison.pdf

Click the ▶ button in Cursor, or run directly:
    python demo_beta_vs_q.py
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import matplotlib.pyplot as plt
import numpy as np
from qrk_analysis.programs.program1 import plot_largest_beta_vs_q
from qrk_analysis.programs.utils import savefig

# ── parameters ────────────────────────────────────────────────────────────────
Q_VALUES = np.linspace(0.1, 1, 30)
D        = 1_000_000
T        = 1
DELTA_F  = 1.0
SUBDIR   = "adversarial"
FONT_SIZE = 18

# Apply global font settings.
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": max(FONT_SIZE - 2, 8),
    "xtick.labelsize": max(FONT_SIZE - 2, 8),
    "ytick.labelsize": max(FONT_SIZE - 2, 8),
})

# ── run & plot ────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== Original: check_feasibility_conditions ===")
    q_arr_orig, betas_orig = plot_largest_beta_vs_q(
        q_values      = Q_VALUES,
        D             = D,
        T             = T,
        delta_f       = DELTA_F,
        use_revised   = False,
        title         = "Largest feasible β vs q  (adversarial, original)",
        save_filename = "beta_vs_q_orig.pdf",
        save_subdir   = SUBDIR,
    )

    print("\n=== Revised: check_feasibility_conditions_adversarial_revised ===")
    q_arr_rev, betas_rev = plot_largest_beta_vs_q(
        q_values      = Q_VALUES,
        D             = D,
        T             = T,
        delta_f       = DELTA_F,
        use_revised   = True,
        title         = "Largest feasible β vs q  (adversarial, revised)",
        save_filename = "beta_vs_q_revised.pdf",
        save_subdir   = SUBDIR,
    )

    # comparison overlay
    plt.figure(figsize=(9, 5))
    plt.plot(q_arr_orig, betas_orig, marker="o",  label="Original (separate worst-case)")
    plt.plot(q_arr_rev,  betas_rev,  marker="s", linestyle="--", label="Revised (shared threshold)")
    plt.xlabel("q")
    plt.ylabel("Largest β")
    plt.title("Original vs Revised: largest feasible β vs q  (adversarial noise)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    savefig("beta_vs_q_comparison.pdf", subdir=SUBDIR)

    for label, q_arr, betas in [("Original", q_arr_orig, betas_orig),
                                 ("Revised",  q_arr_rev,  betas_rev)]:
        valid = [(q, b) for q, b in zip(q_arr, betas) if b is not None]
        if valid:
            best_q, best_beta = max(valid, key=lambda x: x[1])
            print(f"{label}  peak: q={best_q:.3f}, beta_max={best_beta:.4f}")
