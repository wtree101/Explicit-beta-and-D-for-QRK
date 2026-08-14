"""
Demo: Smallest feasible D vs beta – fixed (deterministic) noise C.

Uses :func:`~qrk_analysis.programs.program2.program2_smallest_D_sign_id_C`
(scalar contraction c at Phi(1-alpha'), same alpha search as adversarial).
One subplot per C; each subplot overlays curves for several c_target values.

Figure saved to figures/fixed_noise/fixed_D_vs_beta_c_target.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit(
        "matplotlib not found. Use Anaconda Python, e.g.\n"
        "  ~/anaconda3/bin/python3 qrk_analysis/programs/demo_D_vs_beta_fixed.py\n"
        f"Current: {sys.executable}"
    )
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qrk_analysis.programs.program2 import program2_smallest_D_sign_id_C
from qrk_analysis.programs.utils import savefig

# ── parameters (match demo_D_vs_beta_adv style) ───────────────────────────────
NUM_POINTS       = 10
BETA_MIN         = 0.001
BETA_MAX         = 0.10
T                = 10_000
Q                = 0.5
DELTA_F          = 0.1
D_MAX            = 50000
D_PRECISION      = 2
C_VALUES         = [4.0]
C_TARGET_VALUES  = [0.0, 0.01, 0.05]
SUBDIR           = "fixed_noise"

COLORS = ["steelblue", "darkorange", "forestgreen", "crimson"]

beta_series = np.linspace(BETA_MIN, BETA_MAX, NUM_POINTS)


def smallest_D_for_beta(beta: float, C: float, c_target: float) -> float:
    if beta >= Q - 0.01 or beta >= 1 - Q - 0.01:
        return np.nan
    res = program2_smallest_D_sign_id_C(
        beta=beta, T=T, q=Q, delta_f=DELTA_F, C=C,
        D_max=D_MAX, D_precision=D_PRECISION,
        c_target=c_target,
    )
    return res["smallest_D"] if res["smallest_D"] is not None else np.nan


if __name__ == "__main__":
    nrows = len(C_VALUES)
    fig, axes = plt.subplots(nrows, 1, figsize=(9, 3.2 * nrows), sharex=True)
    if nrows == 1:
        axes = [axes]

    for ax, C in zip(axes, C_VALUES):
        for c_target, color in zip(C_TARGET_VALUES, COLORS):
            D_curve = [smallest_D_for_beta(b, C, c_target) for b in beta_series]
            label = (
                fr"$c_{{\rm target}} = {c_target}$"
                if c_target > 0
                else r"$c_{\rm target} = 0$ (baseline)"
            )
            ax.plot(beta_series, D_curve, ".-", linewidth=1.6, color=color, label=label)
        ax.set_ylabel("Smallest feasible $D$", fontsize=11)
        ax.set_title(fr"Fixed noise $C={C}$", fontsize=12)
        ax.legend(fontsize=9, loc="best")
        ax.grid(True, alpha=0.3)
        ax.set_yscale("log")

    axes[-1].set_xlabel(r"$\beta$ (corruption rate)", fontsize=12)
    fig.suptitle(
        fr"Required $D$ vs $\beta$ – fixed noise $C$, "
        fr"$T={T},\; q={Q},\; \delta_f={DELTA_F}$",
        fontsize=12,
        y=1.01,
    )
    plt.tight_layout()
    savefig("fixed_D_vs_beta_c_target.png", subdir=SUBDIR)

    print("Done.")
