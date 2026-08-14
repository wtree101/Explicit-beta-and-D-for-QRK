"""
Feasibility condition checks for the QRK convergence analysis.

Each function verifies whether a particular combination of parameters
(T, beta, D, q, alpha_0, alpha_prime, delta_f, noise model) satisfies
all conditions required for geometric convergence of the QRK iterates:

  1. p_l^c condition  – the quantile subsample lower-concentrates.
  2. c condition      – the net one-step contraction c > 0.
  3. p_u condition    – the quantile subsample upper-concentrates.
  4. failure_prob     – overall failure probability <= delta_f.

Noise model variants
--------------------
adversarial      : `check_feasibility_conditions`
                   Worst-case noise; uses sigma_min_alpha0_square and the
                   half-normal quantile bound (Phi_{1-alpha'}).
adv_revised      : `check_feasibility_conditions_adversarial_revised`
                   Revised adversarial: pointwise over Q_{q,k+1}, decrease and
                   increase share the same threshold tilde_q (sharper than
                   separating the two worst cases).
sign_id          : `check_feasibility_conditions_sign_id`
                   As adversarial but replaces the error-increase term with
                   error_increased_C_3 (fixed C, but C is read from an
                   outer-scope variable – legacy, prefer sign_id_C).
sign_id_C        : `check_feasibility_conditions_sign_id_C`
                   Fixed noise C passed explicitly; uses error_increased_C_3.
sign_id_C_revised: `check_feasibility_conditions_sign_id_C_revised`
                   Revised: conditions on Q_{q,k+1} pointwise over a grid,
                   exploiting the independence structure for fixed noise C.
C_sup_revised    : `check_feasibility_conditions_C_sup_revised`
                   Pointwise over Q_{q,k+1} with supremum over C on a grid
                   (oblivious / worst-case fixed magnitude).
random           : `check_feasibility_conditions_random`
                   Gaussian noise C ~ N(0, sigma^2); uses
                   error_increased_Gaussian_noise at the worst-case threshold.
random_revised   : `check_feasibility_conditions_random_revised`
                   Revised: pointwise over Q_{q,k+1}, exploiting independence
                   for Gaussian noise (sharper oblivious analysis).
random_sup_revised: `check_feasibility_conditions_random_sup_revised`
                    Like random_revised but supremum over sigma on a grid.
"""

import numpy as np
from ..core.quantile import half_normal_quantile, sigma_min_alpha0_square
from ..core.divergence import DKL
from ..noise.fixed import error_increased_C_3
from ..noise.fixed import find_C_with_largest_error_increase_fast
from ..noise.oblivious import (
    error_increased_Gaussian_noise,
    error_increased_Gaussian_noise_batch,
    error_increased_Gaussian_noise_grid,
)


def find_sigma_with_largest_error_increase_fast(
    qq: float,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
    num_points: int = 200,
) -> tuple:
    """Grid-search for sigma that maximizes error_increased_Gaussian_noise."""
    sigma_grid = np.linspace(sigma_min, sigma_max, num_points)
    errors = error_increased_Gaussian_noise_batch(qq=qq, sigmas=sigma_grid)
    max_idx = int(np.argmax(errors))
    return sigma_grid[max_idx], errors[max_idx]


# ---------------------------------------------------------------------------
# Adversarial noise
# ---------------------------------------------------------------------------

def check_feasibility_conditions(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    c_target: float = 0.0,
    *,
    enforce_failure_probability: bool = True,
) -> dict:
    """Feasibility check for adversarial Massart noise.

    Parameters
    ----------
    T : int
        Number of iterations.
    beta : float
        Corruption fraction.
    D : float
        Subsample size.
    q : float
        Quantile level.
    alpha_0 : float
        Lower threshold for Q_{q,k+1} (must satisfy 0 < alpha_0 < q - beta).
    alpha_prime : float
        Upper slack (must satisfy 0 < alpha_prime < 1 - q - beta).
    delta_f : float
        Allowed total failure probability.
    c_target : float, optional
        Minimum required contraction rate.  Feasibility is declared only when
        c >= c_target.  Default 0.0 recovers the original behaviour (c > 0).

    Returns
    -------
    dict
        feasible, and diagnostic fields (p_l_c, c, p_u, failure_prob).
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}
    if D == np.inf:
        p_l_c = 1.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)
    # print(D,p_l_c)
    sigma_min_sq = sigma_min_alpha0_square(alpha_0/(1-beta), q)
    Phi_1m_ap = half_normal_quantile(1 - alpha_prime/(1-beta))
    c = (
        (1 - beta) * p_l_c * sigma_min_sq
        - beta * (Phi_1m_ap ** 2 + 2 * Phi_1m_ap * np.sqrt(2.0 / np.pi))
    )
    # print("c:", c)
    if c < c_target:
        return {"feasible": False, "reason": "c condition violated", "c": c}
    
    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1 - (1 - p_u) ** T

    failure_constraint_satisfied = failure_prob <= delta_f
    if enforce_failure_probability and not failure_constraint_satisfied:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "p_l_c": p_l_c,
            "c": c,
            "p_u": p_u,
            "failure_constraint_enforced": True,
            "failure_constraint_satisfied": False,
        }

    return {
        "feasible": True,
        "p_l_c": p_l_c,
        "c": c,
        "p_u": p_u,
        "failure_prob": failure_prob,
        "failure_constraint_enforced": enforce_failure_probability,
        "failure_constraint_satisfied": failure_constraint_satisfied,
    }

def check_feasibility_conditions_adversarial_revised(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    num_grid: int = 20,
) -> dict:
    """Revised feasibility check for adversarial Massart noise.

    Key difference from :func:`check_feasibility_conditions`:
    instead of bounding the error increase at the single worst-case threshold
    Phi(1 - alpha'), we sweep Q_{q,k+1} = tilde_q over [alpha_0, 1 - alpha_prime]
    and verify the net contraction c(tilde_q) >= 0 pointwise.  Both the
    decrease term (sigma_min) and the increase term (adversarial bound) are
    evaluated at the *same* tilde_q, so they share the quantile threshold.

    The pointwise net contraction at tilde_q is:
        c(tilde_q) = - S*_failure_penalty
                     + (1 - beta) * sigma_min_alpha0_square(tilde_q, q)  [decrease]
                     - beta * (Phi(tilde_q)^2 + 2*Phi(tilde_q)*E|Z|)    [adversarial increase]

    Feasibility requires min_{tilde_q} c(tilde_q) > 0.

    Parameters
    ----------
    num_grid : int
        Number of grid points over [alpha_0, 1 - alpha_prime].
    (others as in :func:`check_feasibility_conditions`)

    Returns
    -------
    dict
        feasible, c_min, worst_tilde_q, c_values, tilde_q_grid,
        p_l, p_l_c, p_u, failure_prob.
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}

    if D == np.inf:
        p_l = 0.0
        p_l_c = 1.0
    else:
        p_l = np.exp(-DKL(q, beta + alpha_0) * D)
        p_l_c = 1.0 - p_l
    S_star_penalty = (1 - beta) * p_l * sigma_min_alpha0_square(alpha_0, q)

    tilde_q_grid = np.linspace(alpha_0, 1.0 - alpha_prime, num_grid)
    c_values = np.full(num_grid, np.nan)
    c_min = np.inf
    worst_tilde_q = tilde_q_grid[0]

    for idx, tilde_q in enumerate(tilde_q_grid):
        decrease = (1 - beta) * sigma_min_alpha0_square(tilde_q, q)
        phi_q = half_normal_quantile(tilde_q)           # Phi(tilde_q): adversarial threshold
        increase = beta * (phi_q ** 2 + 2 * phi_q * 0.798)
        c_val = -S_star_penalty + decrease - increase
        c_values[idx] = c_val
        if c_val < c_min:
            c_min = c_val
            worst_tilde_q = tilde_q
        if c_min <= 0:
            break

    if c_min <= 0:
        return {
            "feasible": False,
            "reason": "c condition violated",
            "c_min": c_min,
            "worst_tilde_q": worst_tilde_q,
            "c_values": c_values,
            "tilde_q_grid": tilde_q_grid,
            "p_l": p_l,
            "p_l_c": p_l_c,
        }

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1.0 - (1.0 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "c_min": c_min,
        }

    return {
        "feasible": True,
        "c_min": c_min,
        "worst_tilde_q": worst_tilde_q,
        "c_values": c_values,
        "tilde_q_grid": tilde_q_grid,
        "p_l": p_l,
        "p_l_c": p_l_c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }

# ---------------------------------------------------------------------------
# DEPRECATED – eps/n interface (do not use in new code)
#
# The functions below (_compute_T, check_feasibility_conditions_eps,
# check_feasibility_conditions_sign_id_C_eps,
# check_feasibility_conditions_random_eps) attempt to derive T internally
# from (eps, n) via  T = ceil(ln(delta_f * eps) / ln(1 - c/n)).
#
# This design is problematic:
#   1. The formula conflates the failure-probability budget (delta_f) with
#      the accuracy target (eps) in a single scalar, which is not how the
#      two constraints are separated in the rest of the codebase.
#   2. The contraction c itself depends on (D, alpha_0, alpha_prime), so
#      pinning T before optimising over alpha is circular.
#   3. The resulting T is typically useless for the feasibility check because
#      the failure-probability constraint  1-(1-p_u)^T <= delta_f  is then
#      verified *after* T is fixed – but p_u also depends on D and alpha,
#      so there is no guarantee the two constraints are compatible.
#
# Preferred approach: keep T as an explicit input (use check_feasibility_conditions
# and friends), sweep (D, T) jointly, or derive T separately from the
# convergence guarantee once c is known.
# ---------------------------------------------------------------------------

def _compute_T(eps: float, n: int, c_adv: float, delta_f: float) -> int | None:
    """[DEPRECATED] Compute T = ceil( ln(delta_f * eps) / ln(1 - c_adv/n) ).

    .. deprecated::
        See module-level note above.  Do not use in new code.
    """
    rate = c_adv / n
    if rate <= 0 or rate >= 1:
        return None
    log_arg = 0.1*delta_f * eps
    if log_arg <= 0 or log_arg >= 1:
        return None
    T = int(np.log(log_arg) / np.log(1.0 - rate))
    return max(T, 1)


def check_feasibility_conditions_eps(
    eps: float,
    n: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
) -> dict:
    """[DEPRECATED] Feasibility check for adversarial Massart noise with (eps, n) inputs.

    .. deprecated::
        See module-level note above.  Do not use in new code.
        Use :func:`check_feasibility_conditions` with an explicit T instead.
    """
    if not (0 < alpha_0 < q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds", "T_required": None}
    if not (0 < alpha_prime < 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds", "T_required": None}

    if D == np.inf:
        p_l_c = 1.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)

    sigma_min_sq = sigma_min_alpha0_square(alpha_0, q)
    Phi_1m_ap = half_normal_quantile(1 - alpha_prime)
    c = (
        (1 - beta) * p_l_c * sigma_min_sq
        - beta * (Phi_1m_ap ** 2 + 2 * Phi_1m_ap * 0.798)
    )

    if c <= 0:
        return {"feasible": False, "reason": "c condition violated", "c": c, "T_required": None}

    T = _compute_T(eps, n, c, delta_f)
    if T is None:
        return {"feasible": False, "reason": "c/n out of range for T computation", "c": c, "T_required": None}

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1 - (1 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "T_required": T,
        }

    return {
        "feasible": True,
        "T_required": T,
        "p_l_c": p_l_c,
        "c": c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }


def check_feasibility_conditions_sign_id_C_eps(
    eps: float,
    n: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    C: float,
) -> dict:
    """[DEPRECATED] Feasibility check for fixed noise C with (eps, n) inputs.

    .. deprecated::
        See module-level note above.  Do not use in new code.
        Use :func:`check_feasibility_conditions_sign_id_C` with an explicit T instead.
    """
    if not (0 < alpha_0 < q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds", "T_required": None}
    if not (0 < alpha_prime < 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds", "T_required": None}

    if D == np.inf:
        p_l_c = 1.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)
    sigma_min_sq = sigma_min_alpha0_square(alpha_0, q)
    Phi_1m_ap = half_normal_quantile(1 - alpha_prime)
    c = (1 - beta) * p_l_c * sigma_min_sq - beta * error_increased_C_3(qq=Phi_1m_ap, C=C)

    if c <= 0:
        return {"feasible": False, "reason": "c condition violated", "c": c, "T_required": None}

    T = _compute_T(eps, n, c, delta_f)
    if T is None:
        return {"feasible": False, "reason": "c/n out of range for T computation", "c": c, "T_required": None}

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1 - (1 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "T_required": T,
        }

    return {
        "feasible": True,
        "T_required": T,
        "p_l_c": p_l_c,
        "c": c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }


def check_feasibility_conditions_random_eps(
    eps: float,
    n: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    sigma: float,
) -> dict:
    """[DEPRECATED] Feasibility check for Gaussian noise with (eps, n) inputs.

    .. deprecated::
        See module-level note above.  Do not use in new code.
        Use :func:`check_feasibility_conditions_random` with an explicit T instead.
    """
    if not (0 < alpha_0 < q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds", "T_required": None}
    if not (0 < alpha_prime < 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds", "T_required": None}

    if D == np.inf:
        p_l_c = 1.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)
    sigma_min_sq = sigma_min_alpha0_square(alpha_0, q)
    Phi_1m_ap = half_normal_quantile(1 - alpha_prime)
    c = (
        (1 - beta) * p_l_c * sigma_min_sq
        - beta * error_increased_Gaussian_noise(qq=Phi_1m_ap, sigma=sigma)
    )

    if c <= 0:
        return {"feasible": False, "reason": "c condition violated", "c": c, "T_required": None}

    T = _compute_T(eps, n, c, delta_f)
    if T is None:
        return {"feasible": False, "reason": "c/n out of range for T computation", "c": c, "T_required": None}

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1 - (1 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "T_required": T,
        }

    return {
        "feasible": True,
        "T_required": T,
        "p_l_c": p_l_c,
        "c": c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }

# ---------------------------------------------------------------------------
# Fixed noise C (sign-id variants)
# ---------------------------------------------------------------------------

def check_feasibility_conditions_sign_id(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
) -> dict:
    """Legacy feasibility check using error_increased_C_3 with outer-scope C.

    .. warning::
        This function references a variable ``C`` from the calling scope.
        Prefer :func:`check_feasibility_conditions_sign_id_C` which takes C
        as an explicit argument.
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}
    if D == np.inf:
        p_l_c = 1.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)
    sigma_min_sq = sigma_min_alpha0_square(alpha_0, q)
    Phi_1m_ap = half_normal_quantile(1 - alpha_prime)

    import builtins
    C = builtins.__dict__.get("C", 1.0)  # fallback; caller should set C in scope

    c = (1 - beta) * p_l_c * sigma_min_sq - beta * error_increased_C_3(qq=Phi_1m_ap, C=C)

    if c <= 0:
        return {"feasible": False, "reason": "c condition violated", "c": c}

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1 - (1 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
        }

    return {
        "feasible": True,
        "p_l_c": p_l_c,
        "c": c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }


def check_feasibility_conditions_sign_id_C(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    C: float,
    c_target: float = 0.0,
) -> dict:
    """Feasibility check for fixed (deterministic) noise C.

    Uses error_increased_C_3 at the worst-case threshold Phi(1 - alpha').

    Parameters
    ----------
    C : float
        Fixed noise magnitude.
    c_target : float, optional
        Require c >= c_target (scalar contraction).  Default 0.0.
    (others as in :func:`check_feasibility_conditions`)
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}

    if D == np.inf:
        p_l_c = 1.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)
    sigma_min_sq = sigma_min_alpha0_square(alpha_0, q)
    Phi_1m_ap = half_normal_quantile(1 - alpha_prime)
    c = (1 - beta) * p_l_c * sigma_min_sq - beta * error_increased_C_3(qq=Phi_1m_ap, C=C)

    if c < c_target:
        return {"feasible": False, "reason": "c condition violated", "c": c}

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1 - (1 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
        }

    return {
        "feasible": True,
        "p_l_c": p_l_c,
        "c": c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }


def check_feasibility_conditions_sign_id_C_revised(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    C: float,
    num_grid: int = 100,
    c_target: float = 0.0,
) -> dict:
    """Revised feasibility check for fixed noise C, conditioning on Q_{q,k+1}.

    Key idea: instead of bounding error increase at the worst-case threshold
    Phi(1 - alpha'), we sweep Q_{q,k+1} = Qq over [alpha_0, 1 - alpha_prime]
    and verify c(Qq) >= 0 pointwise.  This exploits the independence of the
    fixed noise C from the quantile subsample.

    The pointwise net contraction at Qq is:
        c(Qq) = - S*_failure_penalty
                + (1 - beta) * sigma_min_alpha0_square(Qq, q)   [decrease]
                - beta * error_increased_C_3(Phi(Qq), C)         [increase]

    Feasibility requires min_{Qq} c(Qq) >= c_target.

    Parameters
    ----------
    num_grid : int
        Number of grid points over [alpha_0, 1 - alpha_prime].
    c_target : float, optional
        Minimum required pointwise contraction (applied to c_min).  Default 0.0.
    (others as in :func:`check_feasibility_conditions_sign_id_C`)

    Returns
    -------
    dict
        feasible, c_min, worst_Qq, c_values, Qq_grid, p_l, p_l_c, p_u,
        failure_prob.
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}

    if D == np.inf:
        p_l_c = 1.0
        p_l = 0.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)
        p_l = np.exp(-DKL(q, beta + alpha_0) * D)
    S_star_penalty = (1 - beta) * p_l * sigma_min_alpha0_square(alpha_0, q)

    q_grid = np.linspace(alpha_0, 1.0 - alpha_prime, num_grid)
    c_values = np.full(num_grid, np.nan)
    c_min = np.inf
    worst_Qq = q_grid[0]

    for idx, q_cond in enumerate(q_grid):
        decrease = (1 - beta) * sigma_min_alpha0_square(q_cond, q)
        phi_q = half_normal_quantile(q_cond)
        increase = beta * error_increased_C_3(qq=phi_q, C=C)
        c_val = -S_star_penalty + decrease - increase
        c_values[idx] = c_val
        if c_val < c_min:
            c_min = c_val
            worst_Qq = q_cond
        if c_min < c_target:
            break

    if c_min < c_target:
        return {
            "feasible": False,
            "reason": "c condition violated",
            "c_min": c_min,
            "worst_Qq": worst_Qq,
            "c_values": c_values,
            "Qq_grid": q_grid,
            "p_l": p_l,
            "p_l_c": p_l_c,
        }

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1.0 - (1.0 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "c_min": c_min,
        }

    return {
        "feasible": True,
        "c_min": c_min,
        "worst_Qq": worst_Qq,
        "c_values": c_values,
        "Qq_grid": q_grid,
        "p_l": p_l,
        "p_l_c": p_l_c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }

def check_feasibility_conditions_C_sup_revised(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    num_grid_Q: int = 10,
    C_min: float = 0.0,
    C_max: float = 20.0,
    num_points_C: int = 200,
    c_target: float = 0.0,
    *,
    enforce_failure_probability: bool = True,
) -> dict:
    """Revised fixed-noise check: supremum over C on a grid at each Q_{q,k+1}.

    Mirrors :func:`check_feasibility_conditions_random_sup_revised`, but the
    increase term uses :func:`~noise.fixed.find_C_with_largest_error_increase_fast`
    to approximate sup_{C in [C_min, C_max]} error_increased_C_3(Phi(Qq), C).

    Pointwise net contraction at Qq:
        c(Qq) = - S*_failure_penalty
                + (1 - beta) * sigma_min_alpha0_square(Qq, q)
                - beta * sup_C error_increased_C_3(Phi(Qq), C)

    Feasibility requires min_{Qq} c(Qq) >= c_target.

    Parameters
    ----------
    num_grid_Q : int
        Grid points over [alpha_0, 1 - alpha_prime] for Q_{q,k+1}.
    C_min, C_max : float
        Range for the supremum search over noise magnitude C.
    num_points_C : int
        Grid size inside :func:`~noise.fixed.find_C_with_largest_error_increase_fast`.
    c_target : float, optional
        Minimum required pointwise contraction (applied to c_min).  Default 0.0.

    Returns
    -------
    dict
        feasible, c_min, worst_Qq, worst_Qq_C_star, c_values, Qq_grid,
        p_l, p_l_c, p_u, failure_prob.
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}

    if D == np.inf:
        p_l_c = 1.0
        p_l = 0.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)
        p_l = np.exp(-DKL(q, beta + alpha_0) * D)
    lower_q = alpha_0 / (1.0 - beta)
    upper_q = 1.0 - alpha_prime / (1.0 - beta)
    S_star_penalty = (
        (1.0 - beta) * p_l * sigma_min_alpha0_square(lower_q, q)
    )

    q_grid = np.linspace(lower_q, upper_q, num_grid_Q)
    c_values = np.full(num_grid_Q, np.nan)
    c_min = np.inf
    worst_Qq = q_grid[0]
    worst_Qq_C_star = np.nan

    for idx, q_cond in enumerate(q_grid):
        decrease = (1 - beta) * sigma_min_alpha0_square(q_cond, q)
        phi_q = half_normal_quantile(q_cond)
        C_star, max_error = find_C_with_largest_error_increase_fast(
            qq=phi_q,
            C_min=C_min,
            C_max=C_max,
            num_points=num_points_C,
        )
        increase = beta * max_error
        c_val = -S_star_penalty + decrease - increase
        c_values[idx] = c_val
        if c_val < c_min:
            c_min = c_val
            worst_Qq = q_cond
            worst_Qq_C_star = C_star

        if c_min < c_target:
            break

    if c_min < c_target:
        return {
            "feasible": False,
            "reason": "c condition violated",
            "c_min": c_min,
            "worst_Qq": worst_Qq,
            "worst_Qq_C_star": worst_Qq_C_star,
            "c_values": c_values,
            "Qq_grid": q_grid,
            "p_l": p_l,
            "p_l_c": p_l_c,
        }

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1.0 - (1.0 - p_u) ** T

    failure_constraint_satisfied = failure_prob <= delta_f
    if enforce_failure_probability and not failure_constraint_satisfied:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "c_min": c_min,
            "worst_Qq": worst_Qq,
            "worst_Qq_C_star": worst_Qq_C_star,
            "p_l": p_l,
            "p_l_c": p_l_c,
            "p_u": p_u,
            "failure_constraint_enforced": True,
            "failure_constraint_satisfied": False,
        }

    return {
        "feasible": True,
        "c_min": c_min,
        "worst_Qq": worst_Qq,
        "worst_Qq_C_star": worst_Qq_C_star,
        "c_values": c_values,
        "Qq_grid": q_grid,
        "p_l": p_l,
        "p_l_c": p_l_c,
        "p_u": p_u,
        "failure_prob": failure_prob,
        "failure_constraint_enforced": enforce_failure_probability,
        "failure_constraint_satisfied": failure_constraint_satisfied,
    }


def check_feasibility_conditions_random_sup_revised(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    num_grid_Q: int = 1,
    sigma_min: float = 0.01,
    sigma_max: float = 10.0,
    num_points_C: int = 10,
    c_target: float = 0.0,
    *,
    enforce_failure_probability: bool = True,
) -> dict:
    """Revised Gaussian check taking supremum over sigma on a grid."""
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}

    if D == np.inf:
        p_l_c = 1.0
        p_l = 0.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)
        p_l = np.exp(-DKL(q, beta + alpha_0) * D)
    lower_q = alpha_0 / (1.0 - beta)
    upper_q = 1.0 - alpha_prime / (1.0 - beta)
    S_star_penalty = (
        (1.0 - beta) * p_l * sigma_min_alpha0_square(lower_q, q)
    )

    q_grid = np.linspace(lower_q, upper_q, num_grid_Q)
    c_values = np.full(num_grid_Q, np.nan)
    c_min = np.inf
    worst_Qq = q_grid[0]
    worst_Qq_sigma_star = np.nan

    sigma_grid = np.linspace(sigma_min, sigma_max, num_points_C)
    phi_grid = np.asarray([half_normal_quantile(q_cond) for q_cond in q_grid])
    error_grid = error_increased_Gaussian_noise_grid(phi_grid, sigma_grid)

    for idx, q_cond in enumerate(q_grid):
        decrease = (1 - beta) * sigma_min_alpha0_square(q_cond, q)
        sigma_index = int(np.argmax(error_grid[idx]))
        sigma_star = sigma_grid[sigma_index]
        max_error = error_grid[idx, sigma_index]
        increase = beta * max_error
        c_val = -S_star_penalty + decrease - increase
        c_values[idx] = c_val
        if c_val < c_min:
            c_min = c_val
            worst_Qq = q_cond
            worst_Qq_sigma_star = sigma_star

        if c_min < c_target:
            break

    if c_min < c_target:
        return {
            "feasible": False,
            "reason": "c condition violated",
            "c_min": c_min,
            "worst_Qq": worst_Qq,
            "worst_Qq_sigma_star": worst_Qq_sigma_star,
            "c_values": c_values,
            "Qq_grid": q_grid,
            "p_l": p_l,
            "p_l_c": p_l_c,
        }

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1.0 - (1.0 - p_u) ** T

    failure_constraint_satisfied = failure_prob <= delta_f
    if enforce_failure_probability and not failure_constraint_satisfied:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "c_min": c_min,
            "worst_Qq": worst_Qq,
            "worst_Qq_sigma_star": worst_Qq_sigma_star,
            "p_l": p_l,
            "p_l_c": p_l_c,
            "p_u": p_u,
            "failure_constraint_enforced": True,
            "failure_constraint_satisfied": False,
        }

    return {
        "feasible": True,
        "c_min": c_min,
        "worst_Qq": worst_Qq,
        "worst_Qq_sigma_star": worst_Qq_sigma_star,
        "c_values": c_values,
        "Qq_grid": q_grid,
        "p_l": p_l,
        "p_l_c": p_l_c,
        "p_u": p_u,
        "failure_prob": failure_prob,
        "failure_constraint_enforced": enforce_failure_probability,
        "failure_constraint_satisfied": failure_constraint_satisfied,
    }


# Concise canonical name; the historical name remains public for A callers.
check_feasibility = check_feasibility_conditions

# ---------------------------------------------------------------------------
# Oblivious / random noise (Gaussian)
# ---------------------------------------------------------------------------

def check_feasibility_conditions_random(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    sigma: float,
    c_target: float = 0.0,
) -> dict:
    """Feasibility check for Gaussian oblivious noise C ~ N(0, sigma^2).

    Uses the worst-case threshold Phi(1 - alpha') and averages error
    increase over the Gaussian noise distribution.

    Parameters
    ----------
    sigma : float
        Noise standard deviation.
    c_target : float, optional
        Require c >= c_target.  Default 0.0.
    (others as in :func:`check_feasibility_conditions`)
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}

    if D == np.inf:
        p_l_c = 1.0
    else:
        p_l_c = 1 - np.exp(-DKL(q, beta + alpha_0) * D)
    sigma_min_sq = sigma_min_alpha0_square(alpha_0, q)
    Phi_1m_ap = half_normal_quantile(1 - alpha_prime)
    c = (1 - beta) * p_l_c * sigma_min_sq - beta * error_increased_Gaussian_noise(qq=Phi_1m_ap, sigma=sigma)

    if c < c_target:
        return {"feasible": False, "reason": "c condition violated", "c": c}

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1 - (1 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
        }

    return {
        "feasible": True,
        "p_l_c": p_l_c,
        "c": c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }


def check_feasibility_conditions_random_revised(
    T: int,
    beta: float,
    D: float,
    q: float,
    alpha_0: float,
    alpha_prime: float,
    delta_f: float,
    sigma: float,
    num_grid: int = 200,
    c_target: float = 0.0,
) -> dict:
    """Revised feasibility check for Gaussian noise, conditioning on Q_{q,k+1}.

    Mirrors :func:`check_feasibility_conditions_sign_id_C_revised` but uses
    error_increased_Gaussian_noise for the increase term.  The independence
    of C ~ N(0, sigma^2) from the quantile subsample makes this joint
    pointwise analysis valid and typically yields larger feasible beta.

    Parameters
    ----------
    sigma : float
        Noise standard deviation.
    num_grid : int
        Number of grid points over [alpha_0, 1 - alpha_prime].
    c_target : float, optional
        Minimum required pointwise contraction (applied to c_min).  Default 0.0.
    (others as in :func:`check_feasibility_conditions`)

    Returns
    -------
    dict
        feasible, c_min, c_values, Qq_grid, p_l, p_l_c, p_u, failure_prob.
    """
    if not (0 <= alpha_0 <= q - beta):
        return {"feasible": False, "reason": "alpha_0 out of bounds"}
    if not (0 <= alpha_prime <= 1 - q - beta):
        return {"feasible": False, "reason": "alpha_prime out of bounds"}

    if D == np.inf:
        p_l = 0.0
        p_l_c = 1.0
    else:
        p_l = np.exp(-DKL(q, beta + alpha_0) * D)
        p_l_c = 1.0 - p_l
    S_star_penalty = (1 - beta) * p_l * sigma_min_alpha0_square(alpha_0, q)

    Qq_grid = np.linspace(alpha_0, 1.0 - alpha_prime, num_grid)
    c_values = np.full(num_grid, np.nan)
    c_min = np.inf
    worst_Qq = Qq_grid[0]

    for idx, Qq in enumerate(Qq_grid):
        decrease = (1 - beta) * sigma_min_alpha0_square(Qq, q)
        phi_Qq = half_normal_quantile(1.0 - Qq)
        increase = beta * error_increased_Gaussian_noise(qq=phi_Qq, sigma=sigma)
        c_val = -S_star_penalty + decrease - increase
        c_values[idx] = c_val
        if c_val < c_min:
            c_min = c_val
            worst_Qq = Qq
        if c_min < c_target:
            break

    if c_min < c_target:
        return {
            "feasible": False,
            "reason": "c condition violated",
            "c_min": c_min,
            "worst_Qq": worst_Qq,
            "c_values": c_values,
            "Qq_grid": Qq_grid,
            "p_l": p_l,
            "p_l_c": p_l_c,
        }

    if D == np.inf:
        p_u = 0.0
    else:
        p_u = beta * np.exp(-DKL(1 - q, beta + alpha_prime) * D)
    failure_prob = 1.0 - (1.0 - p_u) ** T

    if failure_prob > delta_f:
        return {
            "feasible": False,
            "reason": "failure probability too high",
            "failure_prob": failure_prob,
            "delta_f": delta_f,
            "c_min": c_min,
        }

    return {
        "feasible": True,
        "c_min": c_min,
        "c_values": c_values,
        "Qq_grid": Qq_grid,
        "p_l": p_l,
        "p_l_c": p_l_c,
        "p_u": p_u,
        "failure_prob": failure_prob,
    }
