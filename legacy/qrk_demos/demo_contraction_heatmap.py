"""
Demo: Optimal contraction rate c  in the (q, beta) plane  –  adversarial noise.

For each grid point (q, beta):
  - If (q, beta) is above the feasibility boundary  =>  c = 0 (infeasible).
  - If (q, beta) is below the boundary              =>  c = best c found by
    grid-searching (alpha_0, alpha_prime).

The result is shown as a heatmap with the boundary curve overlaid.

Figure saved to figures/adversarial/contraction_heatmap.png

Run:
    ~/anaconda3/bin/python3 qrk_analysis/programs/demo_contraction_heatmap.py
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
import matplotlib.colors as mcolors

# from qrk_analysis.feasibility.check import check_feasibility_conditions_sign_id_C
from qrk_analysis.feasibility.check import check_feasibility_conditions
from qrk_analysis.programs.program1 import program1_largest_beta
from qrk_analysis.programs.utils import savefig

# Feasibility model: adversarial noise
# C_FIXED = 4.0
# FEASIBILITY_CHECK = check_feasibility_conditions_sign_id_C
# CHECK_KWARGS = {"C": C_FIXED}
FEASIBILITY_CHECK = check_feasibility_conditions
CHECK_KWARGS = {}


# ── parameters ─────────────────────────────────────────────────────────────
D       = 1_000_000   # large D  =>  D-dependent constraints are inactive
T       = 1
delta_f = 1.0
SUBDIR  = "adversarial"

N_Q    = 100   # grid resolution along q axis
N_BETA = 100   # grid resolution along beta axis

q_grid    = np.linspace(0.001, 0.999, N_Q)
beta_grid = np.linspace(1e-4, 0.06, N_BETA)

# ── helper: best c at a single (q, beta) point ─────────────────────────────
def best_c(q: float, beta: float) -> float:
    """Return the optimal c, achieved at alpha_0 = q-beta-0.01, alpha_prime = 1-q-beta-0.01."""
    alpha_0     = q - beta - 0.01
    alpha_prime = 1 - q - beta - 0.01
    if alpha_0 <= 0 or alpha_prime <= 0:
        return 0.0
    res = FEASIBILITY_CHECK(
        T, beta, D, q, alpha_0, alpha_prime, delta_f, **CHECK_KWARGS
    )
    return res["c"] if res["feasible"] else 0.0


# ── sweep grid ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Grid: {N_Q} x {N_BETA} = {N_Q*N_BETA} points  "
          f"(alpha_0 = q-beta-0.01, alpha_prime = 1-q-beta-0.01)")

    C_matrix = np.zeros((N_BETA, N_Q))
    for j, q in enumerate(q_grid):
        print(f"  q = {q:.3f} ({j+1}/{N_Q})")
        for i, beta in enumerate(beta_grid):
            C_matrix[i, j] = best_c(q, beta)

    # ── boundary curve: beta_max(q) ────────────────────────────────────────
    beta_boundary = []
    for q in q_grid:
        b = program1_largest_beta(
            q, D, T, delta_f,
            feasibility_check=FEASIBILITY_CHECK,
            check_kwargs=CHECK_KWARGS,
        )
        beta_boundary.append(b if b is not None else np.nan)
    beta_boundary = np.array(beta_boundary)

    # ── discrete regions by thresholds ────────────────────────────────────
    # Boundaries:  infeasible | (0, 0.01] | (0.01, 0.05] | (0.05, 0.1] | > 0.1
    THRESHOLDS  = [0.01, 0.05, 0.1]
    REGION_COLORS = [
        "#d0e8f5",   # c in (0,    0.01] – light blue
        "#74b9e7",   # c in (0.01, 0.05] – medium blue
        "#1a6eb5",   # c in (0.05, 0.1]  – dark blue
        "#0a2f6e",   # c > 0.1            – deep navy
    ]
    INFEASIBLE_COLOR = "#e8e8e8"  # grey

    # Build an integer label matrix: 0=infeasible, 1..4=regions
    label_matrix = np.zeros_like(C_matrix, dtype=int)
    for idx, (lo, hi) in enumerate(
        zip([0] + THRESHOLDS, THRESHOLDS + [np.inf]), start=1
    ):
        label_matrix[(C_matrix > lo) & (C_matrix <= hi)] = idx

    # ── plot ───────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 6))

    # Discrete colormap: index 0 = infeasible, 1-4 = regions
    all_colors = [INFEASIBLE_COLOR] + REGION_COLORS
    cmap_disc  = mcolors.ListedColormap(all_colors)
    norm_disc  = mcolors.BoundaryNorm(boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5],
                                      ncolors=5)

    ax.pcolormesh(
        q_grid, beta_grid, label_matrix,
        cmap=cmap_disc, norm=norm_disc,
        shading="auto",
    )

    # Contour lines at the three thresholds
    QQ, BB = np.meshgrid(q_grid, beta_grid)
    contour_levels = THRESHOLDS
    ct = ax.contour(QQ, BB, C_matrix, levels=contour_levels,
                    colors=["#555555", "#222222", "#000000"],
                    linewidths=[1.2, 1.5, 2.0],
                    linestyles=["--", "-.", "-"])
    ax.clabel(ct, fmt={v: f"c={v}" for v in contour_levels}, fontsize=9)

    # Colorbar with discrete ticks
    sm = plt.cm.ScalarMappable(
        cmap=mcolors.ListedColormap(all_colors),
        norm=mcolors.BoundaryNorm([-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5),
    )
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, ticks=[0, 1, 2, 3, 4])
    cbar.ax.set_yticklabels(
        ["infeasible", r"$c \leq 0.01$", r"$0.01 < c \leq 0.05$",
         r"$0.05 < c \leq 0.1$", r"$c > 0.1$"],
        fontsize=9,
    )

    # Feasibility boundary curve
    ax.plot(q_grid, beta_boundary, color="black", linewidth=2.5,
            label=r"$\beta_{\max}(q)$ boundary")

    ax.set_xlabel("Quantile level $q$", fontsize=13)
    ax.set_ylabel(r"Corruption rate $\beta$", fontsize=13)
    ax.set_title(
        "Optimal contraction rate $c$ in $(q,\\,\\beta)$ plane\n"
        fr"(grey = infeasible)",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    savefig("contraction_heatmap_high_res.png", subdir=SUBDIR)
    print("Done.")
