"""
Demo: Smallest feasible D vs beta – adversarial Massart noise.

Plots one curve per c_target value, showing how the minimum subsample size D
grows with the corruption rate beta for different contraction-rate requirements.

Figure saved to figures/adversarial/adv_D_vs_beta_c_target.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit(
        "matplotlib not found. Use Anaconda Python, e.g.\n"
        "  ~/anaconda3/bin/python3 qrk_analysis/programs/demo_D_vs_beta_adv.py\n"
        f"Current: {sys.executable}"
    )
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qrk_analysis.programs.program2 import program2_smallest_D_2
from qrk_analysis.programs.utils import savefig

# ── parameters ────────────────────────────────────────────────────────────────
NUM_POINTS      = 20
BETA_MIN        = 0.001
BETA_MAX        = 0.02          # hard upper limit; curves end when infeasible
T               = 10_000        # fixed iteration budget
Q               = 0.75
DELTA_F         = 0.1
D_MAX           = 500
D_PRECISION     = 2
# c_target = 0.0 means only c > 0 is required (original condition)
C_TARGET_VALUES = [0.0, 0.01, 0.05, 0.1]
SUBDIR          = "adversarial"

beta_series = np.linspace(BETA_MIN, BETA_MAX, NUM_POINTS)

# ── sweep ─────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    fig, ax = plt.subplots(figsize=(9, 6))

    colors = ["steelblue", "darkorange", "forestgreen", "crimson"]

    for c_target, color in zip(C_TARGET_VALUES, colors):
        D_curve = []
        for b in beta_series:
            # beta must be strictly inside the feasibility region
            if b >= Q - 0.01 or b >= 1 - Q - 0.01:
                D_curve.append(np.nan)
                continue
            res = program2_smallest_D_2(
                beta=b, T=T, q=Q, delta_f=DELTA_F,
                D_max=D_MAX, D_precision=D_PRECISION,
                c_target=c_target,
            )
            D_curve.append(
                res["smallest_D"] if res["smallest_D"] is not None else np.nan
            )

        label = (
            f"$c_{{\\rm target}} = {c_target}$"
            if c_target > 0
            else r"$c_{\rm target} = 0$  (baseline)"
        )
        ax.plot(beta_series, D_curve, marker=".", linewidth=1.8,
                color=color, label=label)

    ax.set_xlabel(r"$\beta$  (corruption rate)", fontsize=12)
    ax.set_ylabel("Smallest feasible $D$", fontsize=12)
    ax.set_title(
        f"Required $D$ vs $\\beta$ – adversarial noise\n"
        f"$T={T},\\; q={Q},\\; \\delta_f={DELTA_F}$",
        fontsize=12,
    )
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3)
    ax.set_yscale("log")
    plt.tight_layout()
    savefig("adv_D_vs_beta_c_target.png", subdir=SUBDIR)

    print("Done.")
