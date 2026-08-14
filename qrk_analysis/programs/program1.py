"""
Program 1 – largest feasible beta.

Given (D, q, T, delta_f) find the largest corruption fraction beta for which
the QRK algorithm is provably convergent.  Binary search over beta is used
for each q value, and the result is plotted as a function of q (or C / sigma).

All plot functions accept optional ``save_filename`` / ``save_subdir`` arguments.
When provided the figure is saved to figures/<save_subdir>/<save_filename> before
being displayed; when omitted the figure is only shown on screen.
"""

import numpy as np
import matplotlib.pyplot as plt

from .utils import savefig
from ..feasibility.check import (
    check_feasibility_conditions,       
    check_feasibility_conditions_adversarial_revised,
    check_feasibility_conditions_sign_id_C,
    check_feasibility_conditions_sign_id_C_revised,
    check_feasibility_conditions_random,
    check_feasibility_conditions_random_revised,
    check_feasibility_conditions_C_sup_revised,
    check_feasibility_conditions_random_sup_revised,
)


def program1_largest_beta(
    q: float,
    D: float,
    T: int,
    delta_f: float,
    feasibility_check=None,
    check_kwargs: dict = None,
    beta_tol: float = 1e-10,
) -> float | None:
    """Binary search for the largest beta satisfying feasibility at given q.

    Parameters
    ----------
    q : float
        Quantile level.
    D : float
        Subsample size.
    T : int
        Number of iterations.
    delta_f : float
        Allowed total failure probability.
    feasibility_check : callable, optional
        Function with signature (T, beta, D, q, alpha_0, alpha_prime, delta_f,
        **check_kwargs) -> dict.  Defaults to check_feasibility_conditions.
    check_kwargs : dict, optional
        Extra keyword arguments forwarded to feasibility_check (e.g. C, sigma).
    beta_tol : float
        Convergence tolerance for beta.

    Returns
    -------
    float or None
        Largest feasible beta, or None if none found.
    """
    if feasibility_check is None:
        feasibility_check = check_feasibility_conditions
    if check_kwargs is None:
        check_kwargs = {}

    beta_low = 0.0
    beta_high = min(q, 1.0 - q)
    best_beta = None

    # Seed with beta=0 feasibility so we return 0 when only beta=0 is feasible.
    alpha_0 = q
    alpha_prime = 1 - q
    if alpha_0 >= 0 and alpha_prime >= 0:
        res0 = feasibility_check(T, 0.0, D, q, alpha_0, alpha_prime, delta_f, **check_kwargs)
        if isinstance(res0, dict) and res0.get("feasible"):
            best_beta = 0.0

    while beta_high - beta_low > beta_tol:
        beta_mid = (beta_low + beta_high) / 2
        alpha_0 = q - beta_mid
        alpha_prime = 1 - q - beta_mid
        if alpha_0 < 0 or alpha_prime < 0:
            beta_high = beta_mid
            continue
        res = feasibility_check(T, beta_mid, D, q, alpha_0, alpha_prime, delta_f, **check_kwargs)
        if res["feasible"]:
            best_beta = beta_mid
            beta_low = beta_mid
        else:
            beta_high = beta_mid

    return best_beta


def plot_largest_beta_vs_q(
    q_values=None,
    D: float = np.inf,
    T: int = 1,
    delta_f: float = 1.0,
    feasibility_check=None,
    check_kwargs: dict = None,
    use_revised: bool = False,
    title: str = "Largest feasible beta vs q",
    save_filename: str = None,
    save_subdir: str = "adversarial",
) -> tuple:
    """Plot largest feasible beta as a function of the quantile level q.

    Parameters
    ----------
    q_values : array-like, optional
        Quantile levels to sweep.  Defaults to np.linspace(0.1, 1, 30).
    D, T, delta_f : problem parameters.
    feasibility_check : callable, optional
        Explicit check function; overrides use_revised when provided.
    check_kwargs : dict, optional
        Extra keyword arguments forwarded to feasibility_check.
    use_revised : bool
        If True (and feasibility_check is None), use
        check_feasibility_conditions_adversarial_revised.
    title : str
        Plot title.
    save_filename : str, optional
        If given, save figure to figures/<save_subdir>/<save_filename>.
    save_subdir : str
        Sub-folder inside figures/ (default: "adversarial").

    Returns
    -------
    tuple[np.ndarray, list]
        (q_values_array, largest_betas_list)
    """
    if q_values is None:
        q_values = np.linspace(0.1, 1, 30)

    if feasibility_check is None:
        feasibility_check = (
            check_feasibility_conditions_adversarial_revised
            if use_revised
            else check_feasibility_conditions
        )

    largest_betas = [
        program1_largest_beta(qv, D, T, delta_f, feasibility_check, check_kwargs)
        for qv in q_values
    ]

    plt.figure(figsize=(8, 5))
    plt.plot(q_values, largest_betas, marker="o")
    plt.xlabel("q")
    plt.ylabel("Largest feasible beta")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_filename:
        savefig(save_filename, subdir=save_subdir)
    else:
        plt.close()

    return np.asarray(q_values), largest_betas


def plot_largest_beta_vs_C(
    C_values=None,
    q_sweep=None,
    D: float = 1_000_000,
    T: int = 1,
    delta_f: float = 1.0,
    use_revised: bool = False,
    title: str = "Largest feasible beta vs C",
    save_filename: str = None,
    save_subdir: str = "fixed_noise",
) -> tuple:
    """Plot the maximum-over-q largest beta as a function of fixed noise C.

    Parameters
    ----------
    C_values : array-like, optional
        Noise magnitudes to sweep.
    q_sweep : array-like, optional
        q values to search over for each C.
    use_revised : bool
        If True, use check_feasibility_conditions_sign_id_C_revised.
    save_filename : str, optional
        If given, save figure to figures/<save_subdir>/<save_filename>.
    save_subdir : str
        Sub-folder inside figures/ (default: "fixed_noise").

    Returns
    -------
    tuple[np.ndarray, list]
        (C_values_array, max_beta_per_C)
    """
    if C_values is None:
        C_values = np.linspace(0, 10, 10)
    if q_sweep is None:
        q_sweep = np.linspace(0.1, 1, 10)

    check_fn = (
        check_feasibility_conditions_C_sup_revised
        if use_revised
        else check_feasibility_conditions_C_sup_revised
    )

    max_betas = []
    for C_val in C_values:
        print(f"C = {C_val:.4f}")
        betas = [
            program1_largest_beta(qv, D, T, delta_f, check_fn, {"C": C_val})
            for qv in q_sweep
        ]
        valid = [(qv, b) for qv, b in zip(q_sweep, betas) if b is not None]
        if valid:
            best_q, best_beta = max(valid, key=lambda item: item[1])
            max_betas.append(best_beta)
            print(f"  feasible q count: {len(valid)}; best beta: {best_beta:.6g} at q={best_q:.4f}")
        else:
            max_betas.append(np.nan)
            #print("  feasible q count: 0")
            # Diagnostic: check feasibility at beta=0 to see which condition fails.
            reason_counts = {}
            for qv in q_sweep:
                res = check_fn(T, 0.0, D, qv, qv, 1.0 - qv, delta_f, C=C_val)
                reason = res.get("reason", "unknown") if isinstance(res, dict) else "unknown"
                reason_counts[reason] = reason_counts.get(reason, 0) + 1
            reason_summary = ", ".join(f"{k}:{v}" for k, v in sorted(reason_counts.items()))
            print(f"  beta=0 reasons -> {reason_summary}")

    plt.figure(figsize=(8, 5))
    plt.plot(C_values, max_betas, marker="o")
    plt.xlabel("C")
    plt.ylabel("Largest beta (max over q)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_filename:
        savefig(save_filename, subdir=save_subdir)
    else:
        plt.close()

    return np.asarray(C_values), max_betas


def plot_largest_beta_vs_sigma(
    sigma_values=None,
    q_sweep=None,
    D: float = np.inf,
    T: int = 1,
    delta_f: float = 1.0,
    use_revised: bool = False,
    title: str = "Largest feasible beta vs sigma",
    save_filename: str = None,
    save_subdir: str = "gaussian_noise",
) -> tuple:
    """Plot the maximum-over-q largest beta as a function of Gaussian noise sigma.

    Parameters
    ----------
    sigma_values : array-like, optional
        Gaussian noise standard deviations to sweep.
    q_sweep : array-like, optional
        q values to search over for each sigma.
    use_revised : bool
        If True, use check_feasibility_conditions_random_revised.
    save_filename : str, optional
        If given, save figure to figures/<save_subdir>/<save_filename>.
    save_subdir : str
        Sub-folder inside figures/ (default: "gaussian_noise").

    Returns
    -------
    tuple[np.ndarray, list]
        (sigma_values_array, max_beta_per_sigma)
    """
    if sigma_values is None:
        sigma_values = np.linspace(0.1, 10, 20)
    if q_sweep is None:
        q_sweep = np.linspace(0.1, 1, 30)

    check_fn = (
        check_feasibility_conditions_random_revised
        if use_revised
        else check_feasibility_conditions_random
    )

    max_betas = []
    for sigma_val in sigma_values:
        print(f"sigma = {sigma_val:.4f}")
        betas = [
            program1_largest_beta(qv, D, T, delta_f, check_fn, {"sigma": sigma_val})
            for qv in q_sweep
        ]
        valid = [b for b in betas if b is not None]
        max_betas.append(max(valid) if valid else None)
        print(f"  feasible q count: {len(valid)}; best beta: {max_betas[-1]:.6g}" if valid else "  feasible q count: 0")

    plt.figure(figsize=(8, 5))
    plt.plot(sigma_values, max_betas, marker="o")
    plt.xlabel("sigma  (Gaussian noise std)")
    plt.ylabel("Largest beta (max over q)")
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_filename:
        savefig(save_filename, subdir=save_subdir)
    else:
        plt.close()

    return np.asarray(sigma_values), max_betas
