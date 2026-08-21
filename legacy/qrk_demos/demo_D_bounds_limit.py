"""
Demo: D/log T upper and lower bounds vs beta (T -> infinity).

Given q, we compute beta_max from F(q, beta, 1 - q - beta) = 0, then sweep
beta in (0, beta_max) to plot:
    - D_upper/log T from the c=0 constraint
    - D_lower/log T from the KL lower-bound limit

Figure saved to figures/adversarial/D_logT_bounds_limit_c0.png
"""

import sys, os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

import numpy as np
try:
    import matplotlib
except ModuleNotFoundError:
    sys.exit(
        "matplotlib not found. Use Anaconda Python, e.g.\n"
        "  ~/anaconda3/bin/python3 qrk_analysis/programs/demo_D_bounds_limit.py\n"
        f"Current: {sys.executable}"
    )
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from qrk_analysis.core.divergence import DKL
from qrk_analysis.core.quantile import half_normal_quantile, sigma_min_alpha0_square
from qrk_analysis.programs.program1 import program1_largest_beta
from qrk_analysis.programs.utils import savefig

# Larger, consistent font sizes for all figures.
plt.rcParams.update({
    "font.size": 14,
    "axes.titlesize": 16,
    "axes.labelsize": 14,
    "legend.fontsize": 16,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
})

E_ABS_Z = np.sqrt(2.0 / np.pi)
EPS = 1e-15


def _f_term(tilde_m: float) -> float:
    return tilde_m ** 2 + 2.0 * tilde_m * E_ABS_Z


def F_q_beta_alpha_prime(q: float, beta: float, alpha_prime: float) -> float | None:
    """JC-thoughts F(q, beta, alpha_prime) for the c=0 constraint."""
    if not (0 <= beta <= 1):
        return None
    alpha_0 = (q - beta)
    if not (0 <= alpha_0 <= 1):
        return None
    if not (0 <= alpha_prime <= 1.0 - q - beta):
        return None

    g_term = sigma_min_alpha0_square(alpha_0/(1-beta), q)
    tilde_q = 1.0 - alpha_prime / (1.0 - beta)
    if not (0 <= tilde_q <= 1):
        return None
    tilde_m = half_normal_quantile(tilde_q)

    return (1.0 - beta) * g_term - beta * _f_term(tilde_m)


def solve_alpha_prime_c0(q: float, beta: float, tol: float = EPS, max_iter: int = 60):
    """Solve F(q, beta, alpha_prime) = 0 for alpha_prime by bisection."""
    ap_low = 0
    ap_high = 1.0 - q - beta
    if ap_high < ap_low:
        return None

    f_low = F_q_beta_alpha_prime(q, beta, ap_low)
    f_high = F_q_beta_alpha_prime(q, beta, ap_high)
    # print(beta, f_low, f_high)
    if f_low is None or f_high is None:
        return None
    if f_low == 0:
        return ap_low
    if f_high == 0:
        return ap_high
    if f_low * f_high > 0:
        return None

    for _ in range(max_iter):
        ap_mid = 0.5 * (ap_low + ap_high)
        f_mid = F_q_beta_alpha_prime(q, beta, ap_mid)
        if f_mid is None:
            return None
        if abs(f_mid) < tol:
            return ap_mid
        if f_low * f_mid <= 0:
            ap_high, f_high = ap_mid, f_mid
        else:
            ap_low, f_low = ap_mid, f_mid

    return 0.5 * (ap_low + ap_high)


def solve_beta_max(q: float) -> float | None:
    """Compute beta_max using program1_largest_beta (feasibility logic)."""
    return program1_largest_beta(q, np.inf, T_BETA_MAX, DELTA_F_BETA_MAX)


def d_over_logT_upper_c0(beta: float, q: float) -> float | None:
    """Compute D/log T upper bound for c=0 using JC-thoughts asymptotics."""
    alpha_prime = solve_alpha_prime_c0(q, beta)
    if alpha_prime is None:
        return None

    denom = DKL(1.0 - q, beta + alpha_prime)
    if denom <= 0:
        return None

    return 1.0 / denom


def d_over_logT_lower(beta: float, q: float) -> float | None:
    """Compute D/log T lower bound as 1 / DKL(1 - q || beta)."""
    denom = DKL(1.0 - q, beta)
    if denom <= 0:
        return None
    return 1.0 / denom


# ── parameters ────────────────────────────────────────────────────────────────
Q = 0.8
NUM_POINTS = 100
SUBDIR = "adversarial"
D_BETA_MAX = 1_000_000
T_BETA_MAX = 0
DELTA_F_BETA_MAX = 1.0


if __name__ == "__main__":
    beta_max = solve_beta_max(Q)
    print(f"beta_max for q={Q}: {beta_max}")
    if beta_max is None:
        sys.exit("Failed to find beta_max for the given q.")

    beta_series = np.linspace(0, 0.05, NUM_POINTS)

    D_over_logT_upper = []
    D_over_logT_lower = []

    for b in beta_series:
        d_up = d_over_logT_upper_c0(b, Q)
        D_over_logT_upper.append(d_up if d_up is not None else np.nan)

        d_low = d_over_logT_lower(b, Q)
        D_over_logT_lower.append(d_low if d_low is not None else np.nan)

    plt.figure(figsize=(9, 6))
    plt.plot(beta_series, D_over_logT_upper, marker="o",
             label=r"$D_{\rm upper}/\log T$ (c=0)")
    plt.plot(beta_series, D_over_logT_lower, marker="x",
             label=r"$D_{\rm lower}/\log T$")
    plt.xlabel(r"$\beta$")
    plt.ylabel(r"$D/\log T$")
    plt.title(
        rf"$D/\log T$ bounds vs $\beta$ (c=0),  q={Q}\n"
        rf"$\beta_{{\max}} \approx {beta_max:.4f}$"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    savefig("D_logT_bounds_limit_c0.png", subdir=SUBDIR)

    ratio = np.array(D_over_logT_upper) / np.array(D_over_logT_lower)
    plt.figure(figsize=(9, 6))
    plt.plot(beta_series, ratio, marker="s",
             label=r"$\frac{D_{\rm upper}/\log T}{D_{\rm lower}/\log T}$")
    plt.xlabel(r"$\beta$")
    plt.ylabel("Ratio")
    plt.title(
        rf"Upper/Lower ratio over $\log T$ vs $\beta$,  q={Q}\n"
        rf"$\beta_{{\max}} \approx {beta_max:.4f}$"
    )
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    savefig("D_logT_ratio_limit_c0.png", subdir=SUBDIR)
    print("Done.")
