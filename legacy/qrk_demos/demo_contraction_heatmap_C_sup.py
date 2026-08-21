"""
Demo: Optimal contraction rate c_min in the (q, beta) plane – fixed noise sup over C.

Uses :func:`~qrk_analysis.feasibility.check.check_feasibility_conditions_C_sup_revised`
(pointwise over Q_{q,k+1}, supremum over C on a grid).

Figure saved to figures/fixed_noise/contraction_heatmap_C_sup.png

Run:
    ~/anaconda3/bin/python3 qrk_analysis/programs/demo_contraction_heatmap_C_sup.py
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

from qrk_analysis.feasibility.check import check_feasibility_conditions_C_sup_revised
from qrk_analysis.programs.program1 import program1_largest_beta
from qrk_analysis.programs.utils import savefig

# Supremum-over-C grids (inside the feasibility check)
C_MIN = 0.0
C_MAX = 10.0
NUM_GRID_Q = 1
NUM_POINTS_C = 20

FEASIBILITY_CHECK = check_feasibility_conditions_C_sup_revised
CHECK_KWARGS = {
    "num_grid_Q": NUM_GRID_Q,
    "C_min": C_MIN,
    "C_max": C_MAX,
    "num_points_C": NUM_POINTS_C,
}

# ── parameters ─────────────────────────────────────────────────────────────
D       = np.inf
T       = 1
delta_f = 1.0
SUBDIR  = "fixed_noise"

N_Q    = 50
N_BETA = 50

q_grid    = np.linspace(0.05, 0.95, N_Q)
beta_grid = np.linspace(1e-4, 0.4, N_BETA)


def best_c(q: float, beta: float) -> float:
    """c_min at alpha_0 = q-beta-0.01, alpha_prime = 1-q-beta-0.01."""
    alpha_0 = q - beta - 0.01
    alpha_prime = 1 - q - beta - 0.01
    if alpha_0 <= 0 or alpha_prime <= 0:
        return 0.0
    res = FEASIBILITY_CHECK(
        T, beta, D, q, alpha_0, alpha_prime, delta_f, **CHECK_KWARGS
    )
    if not res["feasible"]:
        return 0.0
    return float(res.get("c_min", res.get("c", 0.0)))


if __name__ == "__main__":
    print(f"Grid: {N_Q} x {N_BETA}, CHECK_KWARGS={CHECK_KWARGS}")

    c_matrix = np.zeros((N_BETA, N_Q))
    for j, q in enumerate(q_grid):
        print(f"  q = {q:.3f} ({j+1}/{N_Q})")
        for i, beta in enumerate(beta_grid):
            c_matrix[i, j] = best_c(q, beta)

    beta_boundary = []
    for q in q_grid:
        b = program1_largest_beta(
            q, D, T, delta_f,
            feasibility_check=FEASIBILITY_CHECK,
            check_kwargs=CHECK_KWARGS,
        )
        beta_boundary.append(b if b is not None else np.nan)
    beta_boundary = np.array(beta_boundary)

    THRESHOLDS = [0.01, 0.05, 0.1]
    REGION_COLORS = ["#d0e8f5", "#74b9e7", "#1a6eb5", "#0a2f6e"]
    INFEASIBLE_COLOR = "#e8e8e8"

    label_matrix = np.zeros_like(c_matrix, dtype=int)
    for idx, (lo, hi) in enumerate(
        zip([0] + THRESHOLDS, THRESHOLDS + [np.inf]), start=1
    ):
        label_matrix[(c_matrix > lo) & (c_matrix <= hi)] = idx

    fig, ax = plt.subplots(figsize=(9, 6))
    all_colors = [INFEASIBLE_COLOR] + REGION_COLORS
    cmap_disc = mcolors.ListedColormap(all_colors)
    norm_disc = mcolors.BoundaryNorm(
        boundaries=[-0.5, 0.5, 1.5, 2.5, 3.5, 4.5], ncolors=5
    )

    ax.pcolormesh(q_grid, beta_grid, label_matrix, cmap=cmap_disc, norm=norm_disc, shading="auto")

    QQ, BB = np.meshgrid(q_grid, beta_grid)
    ct = ax.contour(
        QQ, BB, c_matrix, levels=THRESHOLDS,
        colors=["#555555", "#222222", "#000000"],
        linewidths=[1.2, 1.5, 2.0],
        linestyles=["--", "-.", "-"],
    )
    ax.clabel(ct, fmt={v: f"c={v}" for v in THRESHOLDS}, fontsize=9)

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

    ax.plot(q_grid, beta_boundary, color="black", linewidth=2.5,
            label=r"$\beta_{\max}(q)$ boundary")
    ax.set_xlabel("Quantile level $q$", fontsize=13)
    ax.set_ylabel(r"Corruption rate $\beta$", fontsize=13)
    ax.set_title(
        r"Pointwise $c_{\min}$ in $(q,\beta)$ – sup over $C\in$"
        fr"$[{C_MIN},{C_MAX}]$ (num_points_C={NUM_POINTS_C})",
        fontsize=12,
    )
    ax.legend(fontsize=10, loc="upper right")
    plt.tight_layout()
    savefig("contraction_heatmap_C_sup.png", subdir=SUBDIR)
    print("Done.")
