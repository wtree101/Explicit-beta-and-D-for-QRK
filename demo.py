"""Demo: required subsample size D vs corruption rate beta.

Run
---
    python demo.py

What this script does
---------------------
1. For a single (beta, T, q, delta_f) setting, compute and print the
   smallest feasible D together with the optimal (alpha_0, alpha_prime) pair
   and the resulting contraction coefficient c.

2. Sweep beta over a grid and plot D_upper vs beta.
   Saves the figure to figure/demo_D_vs_beta.png.

Set FEASIBILITY_CHECK to use a different feasibility model, and set VERBOSE
to print per-beta progress during the sweep.

Dependencies: numpy, scipy, matplotlib  (see requirements.txt)
"""

# Module: run demo sweeps and plot D vs beta.

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from functools import partial

from qrk_adv.upper_bound import smallest_D
from qrk_adv.feasibility import check_feasibility
from qrk_adv.feasibility import check_feasibility_conditions_random_sup_revised
from qrk_adv.feasibility import check_feasibility_conditions_C_sup_revised
from qrk_adv import set_debug

#set_debug(True)   # 打开调试输出
set_debug(False)  # 关闭调试输出


# ── shared parameters ─────────────────────────────────────────────────────────
T       = 20_000   # number of QRK iterations
Q       = 0.75     # quantile level
DELTA_F = 0.1      # allowed total failure probability
D_MAX   = 50000      # search ceiling; increase if result hits ceiling
c_target = 0.00
NUM_GRID_Q = 4    # grid size for conditional-quantile sweep (Gaussian check)
NUM_POINTS_C = 40  # grid size for sigma sweep (Gaussian check)
# FEASIBILITY_CHECK = check_feasibility
# FEASIBILITY_CHECK = partial(
#     check_feasibility_conditions_C_sup_revised,
#     num_grid_Q=NUM_GRID_Q,
#     C_min=0.0,
#     C_max=20.0,
#     num_points_C=NUM_POINTS_C,
# )
FEASIBILITY_CHECK = partial(
    check_feasibility_conditions_C_sup_revised,
    num_grid_Q=NUM_GRID_Q,
    num_points_C=NUM_POINTS_C,
)
VERBOSE = True
# ─────────────────────────────────────────────────────────────────────────────


def single_example():
    """Print the smallest D for one (beta, T, q, delta_f) setting."""
    beta = 0.01

    print("=" * 60)
    print(f"  Computing smallest feasible D")
    print(f"  beta={beta},  T={T},  q={Q},  delta_f={DELTA_F}", f"c_target={c_target}")
    print("=" * 60)

    res = smallest_D(
        beta,
        T,
        Q,
        DELTA_F,
        D_max=D_MAX,
        c_target=c_target,
        feasibility_check=FEASIBILITY_CHECK,
    )

    if res["smallest_D"] is None:
        print(f"  Infeasible within D_max={D_MAX}. "
              f"Try a smaller beta or a larger D_max.")
    else:
        print(f"  Smallest D          : {res['smallest_D']:.1f}")
        print(f"  alpha_0             : {res['alpha_0']:.4f}")
        print(f"  alpha_prime         : {res['alpha_prime']:.10f}")
        print(f"  Contraction c       : {res['c']:.6f}")
        print(f"  Failure prob        : {res['failure_prob']:.4f}  "
              f"(budget {DELTA_F})")
        if res["hit_ceiling"]:
            print(f"  WARNING: result hit D_max={D_MAX}; "
                  f"the true value may be larger.")
    print()


def sweep_and_plot():
    """Sweep beta and plot D_upper vs beta."""
    beta_min, beta_max, n_pts = 0.001, 0.02, 25

    # Space betas so 1/log(1/beta) is uniform (denser near 0).
    inv_log = np.linspace(1 / np.log(1 / beta_min),
                          1 / np.log(1 / beta_max), n_pts)
    betas = np.exp(-1 / inv_log)

    D_upper = []

    print(f"Sweeping {n_pts} beta values in [{beta_min:.3f}, {beta_max:.3f}]...")

    for idx, b in enumerate(betas, start=1):
        if VERBOSE:
            print(f"[{idx}/{len(betas)}] beta={b:.6g} ...")
        # This is the expensive step per beta.
        res = smallest_D(
            b,
            T,
            Q,
            DELTA_F,
            D_max=D_MAX,
            c_target=c_target,
            feasibility_check=FEASIBILITY_CHECK,
        )
        D_upper.append(
            res["smallest_D"] if res["smallest_D"] is not None else np.nan
        )
        if VERBOSE:
            print(f"  -> smallest_D={D_upper[-1]}")

    D_upper = np.array(D_upper, dtype=float)

    # ── plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(betas, D_upper, "o-", label="Upper bound on $D$", lw=1.8)

    ax.set_xlabel(r"$\beta$  (corruption fraction)", fontsize=12)
    ax.set_ylabel("Required subsample size $D$",     fontsize=12)
    ax.set_title(
        f"Upper bound on $D$ vs $\\beta$\n"
        f"($T={T},\\; q={Q},\\; \\delta_f={DELTA_F}$,  adversarial noise)",
        fontsize=12,
    )
    # Log scale highlights rapid growth in D.
    ax.set_yscale("log")
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    # Save the figure to the local figure/ directory.
    out = "figure/demo_D_vs_beta.png"
    fig.savefig(out, dpi=150)
    print(f"Figure saved to {out}")
    plt.close(fig)


if __name__ == "__main__":
    single_example()
    # sweep_and_plot()
