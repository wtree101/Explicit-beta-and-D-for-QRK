"""
Demo: Smallest feasible D vs beta – fixed noise with supremum over C.

Uses :func:`~qrk_analysis.feasibility.check.check_feasibility_conditions_C_sup_revised`
via :func:`~qrk_analysis.programs.program2.program2_smallest_D_2`.

Figure saved to figures/fixed_noise/fixed_D_vs_beta_c_target_C_sup.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
from functools import partial

try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit(
        "matplotlib not found. Use Anaconda Python, e.g.\n"
        "  ~/anaconda3/bin/python3 qrk_analysis/programs/demo_D_vs_beta_C_sup.py\n"
        f"Current: {sys.executable}"
    )
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qrk_analysis.feasibility.check import check_feasibility_conditions_C_sup_revised
from qrk_analysis.programs.program2 import program2_smallest_D_2
from qrk_analysis.programs.utils import savefig

# Supremum-over-C grids (bound into the feasibility check)
C_MIN = 0.0
C_MAX = 10.0
NUM_GRID_Q = 1
NUM_POINTS_C = 20

FEASIBILITY_CHECK = partial(
    check_feasibility_conditions_C_sup_revised,
    num_grid_Q=NUM_GRID_Q,
    C_min=C_MIN,
    C_max=C_MAX,
    num_points_C=NUM_POINTS_C,
)

NUM_POINTS      = 20
BETA_MIN        = 0.001
BETA_MAX        = 0.20
T               = 20_000
Q               = 0.7
DELTA_F         = 0.1
D_MAX           = 1000
D_PRECISION     = 1
C_TARGET_VALUES = [0.0]
SUBDIR          = "fixed_noise"
FIG_SIZE        = 6

COLORS = ["steelblue", "darkorange", "forestgreen", "crimson"]
beta_series = np.linspace(BETA_MIN, BETA_MAX, NUM_POINTS)


def smallest_D_for_beta(beta: float, c_target: float, idx: int, total: int) -> float:
    if beta >= Q or beta >= 1 - Q:
        return np.nan
    print(f"  [{idx}/{total}] beta={beta:.4f}, c_target={c_target}")
    res = program2_smallest_D_2(
        beta=beta, T=T, q=Q, delta_f=DELTA_F,
        D_max=D_MAX, D_precision=D_PRECISION,
        c_target=c_target,
        feasibility_check=FEASIBILITY_CHECK,
    )
    smallest = res["smallest_D"] if res["smallest_D"] is not None else np.nan
    print(f"    -> smallest_D={smallest}")
    return smallest


if __name__ == "__main__":
    print("=== Parameter settings ===")
    print(f"T={T}, Q={Q}, DELTA_F={DELTA_F}, D_MAX={D_MAX}")
    print(f"C_MIN={C_MIN}, C_MAX={C_MAX}, NUM_GRID_Q={NUM_GRID_Q}, NUM_POINTS_C={NUM_POINTS_C}")
    print(f"C_TARGET_VALUES={C_TARGET_VALUES}")
    print("===========================")

    fig, ax = plt.subplots(figsize=(FIG_SIZE, FIG_SIZE))
    for c_target, color in zip(C_TARGET_VALUES, COLORS):
        print(f"-- c_target={c_target} --")
        D_curve = [
            smallest_D_for_beta(b, c_target, idx, len(beta_series))
            for idx, b in enumerate(beta_series, start=1)
        ]
        label = (
            fr"$c_{{\rm target}} = {c_target}$"
            if c_target > 0
            else r"$c_{\rm target} = 0$ (baseline)"
        )
        ax.plot(beta_series, D_curve, ".-", linewidth=1.6, color=color, label=label)

    ax.set_xlabel(r"$\beta$ (corruption rate)", fontsize=12)
    ax.set_ylabel("Smallest feasible $D$", fontsize=12)
    ax.set_title(
        fr"Required $D$ vs $\beta$ – sup over $C\in[{C_MIN},{C_MAX}]$\n"
        fr"$T={T},\; q={Q},\; \delta_f={DELTA_F}$",
        fontsize=12,
    )
    ax.legend(fontsize=9, loc="best")
    ax.grid(True, alpha=0.3)
    # ax.set_yscale("log")
    plt.tight_layout()
    savefig("fixed_D_vs_beta_c_target_C_sup.png", subdir=SUBDIR)
    print("Done.")
