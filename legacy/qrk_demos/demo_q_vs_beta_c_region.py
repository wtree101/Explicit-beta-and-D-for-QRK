"""
Demo: q(beta) maximizing c and feasible q-region (c > 0) vs beta.

Uses the same fixed-alpha heuristic as demo_contraction_heatmap:
  alpha_0 = q - beta - 0.01
  alpha_prime = 1 - q - beta - 0.01

Plots:
  - x-axis: beta in (0, beta_max)
  - y-axis: q
  - shaded band: q range where c >= 0
  - line: q(beta) that maximizes c

Figure saved to figures/adversarial/q_vs_beta_c_region.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit(
        "matplotlib not found. Use Anaconda Python.\n"
        f"Current: {sys.executable}"
    )
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qrk_analysis.feasibility.check import check_feasibility_conditions
from qrk_analysis.programs.program1 import program1_largest_beta
from qrk_analysis.programs.utils import savefig

# ── parameters ─────────────────────────────────────────────────────────────
D = 1_000_000
T = 1
DELTA_F = 1.0
SUBDIR = "adversarial"
FONT_SIZE = 16

N_Q = 200
N_BETA = 200
Q_MIN = 0.001
Q_MAX = 0.999

# Apply global font settings.
plt.rcParams.update({
    "font.size": FONT_SIZE,
    "axes.titlesize": FONT_SIZE + 2,
    "axes.labelsize": FONT_SIZE,
    "legend.fontsize": max(FONT_SIZE - 2, 8),
    "xtick.labelsize": max(FONT_SIZE - 2, 8),
    "ytick.labelsize": max(FONT_SIZE - 2, 8),
})

# ── helper: best c at a single (q, beta) point ─────────────────────────────
FEASIBILITY_CHECK = check_feasibility_conditions
CHECK_KWARGS = {}


def best_c(q: float, beta: float) -> float:
    """Return c using fixed alpha_0, alpha_prime heuristics."""
    alpha_0 = q - beta - 0.01
    alpha_prime = 1 - q - beta - 0.01
    if alpha_0 <= 0 or alpha_prime <= 0:
        return 0.0
    res = FEASIBILITY_CHECK(T, beta, D, q, alpha_0, alpha_prime, DELTA_F, **CHECK_KWARGS)
    return res["c"] if res["feasible"] else 0.0


if __name__ == "__main__":
    q_grid = np.linspace(Q_MIN, Q_MAX, N_Q)

    # beta_max: maximum feasible beta across q, using program1 logic
    beta_candidates = []
    for q in q_grid:
        b = program1_largest_beta(
            q, D, T, DELTA_F,
            feasibility_check=FEASIBILITY_CHECK,
            check_kwargs=CHECK_KWARGS,
        )
        if b is not None:
            beta_candidates.append(b)

    beta_max = max(beta_candidates) if beta_candidates else None
    if beta_max is None:
        sys.exit("Failed to compute beta_max from program1_largest_beta.")

    beta_grid = np.linspace(1e-4, 0.999 * beta_max, N_BETA)

    q_min = np.full_like(beta_grid, np.nan, dtype=float)
    q_max = np.full_like(beta_grid, np.nan, dtype=float)
    q_star = np.full_like(beta_grid, np.nan, dtype=float)

    for i, beta in enumerate(beta_grid):
        c_vals = np.array([best_c(q, beta) for q in q_grid])
        feasible = c_vals > 0
        if np.any(feasible):
            q_valid = q_grid[feasible]
            q_min[i] = np.min(q_valid)
            q_max[i] = np.max(q_valid)
            q_star[i] = q_grid[np.argmax(c_vals)]

    q_at_beta_max = q_star[-1] if np.isfinite(q_star[-1]) else np.nan

    # ── plot ───────────────────────────────────────────────────────────────
    plt.figure(figsize=(9, 6))
    plt.fill_between(beta_grid, q_min, q_max, color="#74b9e7", alpha=0.25,
                     label="q region with c > 0")
    plt.plot(beta_grid, q_star, color="#1a6eb5", linewidth=2.0,
             label=r"$q(\beta)$ maximizing $c$")

    plt.xlabel(r"$\beta$")
    plt.ylabel("q")
    plt.title(r"Feasible q region and $q(\beta)$ maximizing $c$")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="lower left")
    if np.isfinite(q_at_beta_max):
        plt.text(
            0.98, 0.98,
            rf"$q(\beta_{{\max}}) \approx {q_at_beta_max:.3f}$",
            ha="right", va="top",
            transform=plt.gca().transAxes,
        )
    plt.tight_layout()
    savefig("q_vs_beta_c_region.png", subdir=SUBDIR)
    print("Done.")
