"""
Demo: Smallest feasible D vs beta – Gaussian oblivious noise.

Uses :func:`~qrk_analysis.programs.program2.program2_smallest_D_random`
(scalar contraction c, same alpha search as adversarial).  One subplot per
sigma; each subplot overlays curves for several c_target values.

Figure saved to figures/random_noise/random_D_vs_beta_c_target.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit(
        "matplotlib not found. Use Anaconda Python, e.g.\n"
        "  ~/anaconda3/bin/python3 qrk_analysis/programs/demo_D_vs_beta_random.py\n"
        f"Current: {sys.executable}"
    )
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from functools import partial

from qrk_analysis.programs.program2 import program2_smallest_D_2
from qrk_analysis.programs.utils import savefig
from qrk_analysis.feasibility.check import check_feasibility_conditions_random_sup_revised

# Supremum-over-sigma grids (bound into the feasibility check)
SIGMA_MIN = 0.01
SIGMA_MAX = 10.0
NUM_GRID_Q = 1
NUM_POINTS_C = 20

FEASIBILITY_CHECK = partial(
    check_feasibility_conditions_random_sup_revised,
    num_grid_Q=NUM_GRID_Q,
    sigma_min=SIGMA_MIN,
    sigma_max=SIGMA_MAX,
    num_points_C=NUM_POINTS_C,
)

# ── parameters ────────────────────────────────────────────────────────────────
NUM_POINTS       = 10
BETA_MIN         = 0.001
BETA_MAX         = 0.20
T                = 20_000
Q                = 0.6
DELTA_F          = 0.1
D_MAX            = 1000
D_PRECISION      = 1
SIGMA_VALUES     = [1.0]
C_TARGET_VALUES  = [0.0]
SUBDIR           = "random_noise"
FIG_SIZE         = 6

COLORS = ["steelblue", "darkorange", "forestgreen", "crimson"]

beta_series = np.linspace(BETA_MIN, BETA_MAX, NUM_POINTS)

def smallest_D_for_beta(beta: float, sigma: float, c_target: float, idx: int, total: int) -> float:
    if beta >= Q or beta >= 1 - Q:
        return np.nan
    print(f"  [{idx}/{total}] beta={beta:.4f}, sigma={sigma}, c_target={c_target}")
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
    print(beta_series)
    print("=== Parameter settings ===")
    print(f"NUM_POINTS={NUM_POINTS}")
    print(f"BETA_MIN={BETA_MIN}, BETA_MAX={BETA_MAX}")
    print(f"T={T}, Q={Q}, DELTA_F={DELTA_F}")
    print(f"D_MAX={D_MAX}, D_PRECISION={D_PRECISION}")
    print(f"SIGMA_VALUES={SIGMA_VALUES}, C_TARGET_VALUES={C_TARGET_VALUES}")
    print(f"SUBDIR={SUBDIR}")
    print("===========================")
    nrows = len(SIGMA_VALUES)
    fig, axes = plt.subplots(nrows, 1, figsize=(FIG_SIZE, FIG_SIZE), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, sigma in zip(axes, SIGMA_VALUES):
        print(f"=== sigma={sigma} ===")
        for c_target, color in zip(C_TARGET_VALUES, COLORS):
            print(f"-- c_target={c_target} --")
            D_curve = [
                smallest_D_for_beta(b, sigma, c_target, idx, len(beta_series))
                for idx, b in enumerate(beta_series, start=1)
            ]
            print(D_curve)
            label = (
                fr"$c_{{\rm target}} = {c_target}$"
                if c_target > 0
                else r"$c_{\rm target} = 0$ (baseline)"
            )
            ax.plot(beta_series, D_curve, ".-", linewidth=1.6, color=color, label=label)
        ax.set_ylabel("Smallest feasible $D$", fontsize=11)
        ax.set_title(fr"Gaussian noise $\sigma={sigma}$", fontsize=12)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    axes[-1].set_xlabel(r"$\beta$ (corruption rate)", fontsize=12)
    fig.suptitle(
        fr"Required $D$ vs $\beta$ – Gaussian noise $\sigma$, "
        fr"$T={T},\; q={Q},\; \delta_f={DELTA_F}$",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    savefig("random_D_vs_beta_c_target_sup.png", subdir=SUBDIR)

    print("Done.")
