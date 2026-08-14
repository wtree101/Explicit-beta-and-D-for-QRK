"""Streaming QRK simulation kernels used by the heatmap sweeps."""

from __future__ import annotations

import numpy as np


def validate_oblivious_large_noise(
    quantile_noise_min: float,
    quantile_noise_max: float,
    update_noise_min: float,
    update_noise_max: float,
) -> None:
    """Validate finite ordered noise intervals for ``oblivious_large``."""
    values = (
        quantile_noise_min,
        quantile_noise_max,
        update_noise_min,
        update_noise_max,
    )
    if not all(np.isfinite(value) for value in values):
        raise ValueError("oblivious_large noise values must be finite")
    if quantile_noise_min > quantile_noise_max:
        raise ValueError(
            "oblivious_large requires quantile_noise_min <= quantile_noise_max"
        )
    if update_noise_min > update_noise_max:
        raise ValueError(
            "oblivious_large requires update_noise_min <= update_noise_max"
        )


def streaming_subsampled_qRK_step(
    x: np.ndarray,
    xk: np.ndarray,
    q: float,
    beta: float,
    D: int,
    corruption_type: str,
    c_min: float,
    c_max: float,
    s_min: float,
    s_max: float,
    *,
    quantile_noise_min: float = 1e16,
    quantile_noise_max: float = 1e16,
    update_noise_min: float = 1e8,
    update_noise_max: float = 1e8,
) -> tuple[np.ndarray, float]:
    """Run one streaming quantile-RK update."""
    n = len(x)
    rows = np.random.normal(size=(D + 1, n))
    rows /= np.linalg.norm(rows, axis=1)[:, np.newaxis]

    match corruption_type:
        case "sup_c" | "sup_rand":
            if corruption_type == "sup_c":
                epsilon = np.random.uniform(c_min, c_max, size=D + 1)
            else:
                variances = np.random.uniform(s_min, s_max, size=D + 1)
                epsilon = np.array(
                    [np.random.normal(scale=np.sqrt(value)) for value in variances]
                )
            indicators = np.random.binomial(1, beta, size=D + 1)
            residuals = rows @ (x - xk) + indicators * epsilon
            quantile = np.quantile(np.abs(residuals[1:]), q)
            quantile_fraction = float(np.mean(np.abs(residuals[1:]) <= quantile))
            if abs(residuals[0]) <= quantile:
                xk = xk + residuals[0] * rows[0]
        case "oblivious_large":
            validate_oblivious_large_noise(
                quantile_noise_min,
                quantile_noise_max,
                update_noise_min,
                update_noise_max,
            )
            indicators = np.random.binomial(1, beta, size=D + 1)
            clean_residuals = rows @ (x - xk)
            quantile_noise = np.random.uniform(
                low=quantile_noise_min,
                high=quantile_noise_max,
                size=D,
            )
            update_noise = np.random.uniform(
                low=update_noise_min,
                high=update_noise_max,
            )
            quantile_residuals = (
                clean_residuals[1:] + indicators[1:] * quantile_noise
            )
            update_residual = clean_residuals[0] + indicators[0] * update_noise
            quantile = np.quantile(np.abs(quantile_residuals), q)
            quantile_fraction = float(
                np.mean(np.abs(quantile_residuals) <= quantile)
            )
            if abs(update_residual) <= quantile:
                xk = xk + update_residual * rows[0]
        case "adversarial":
            indicators = np.random.binomial(1, beta, size=D + 1)
            if indicators[0] == 1:
                epsilon = np.full(D + 1, 1e16)
                test_residuals = (
                    (rows @ (x - xk)) * (1 - indicators) + epsilon * indicators
                )
                quantile = np.quantile(np.abs(test_residuals[1:]), q)
                xk = xk + np.sign(rows[0] @ (xk - x)) * quantile * rows[0]
            else:
                epsilon = rows @ (xk - x)
                test_residuals = rows @ (x - xk) + epsilon * indicators
                quantile = np.quantile(np.abs(test_residuals[1:]), q)
                residual = rows[0] @ (x - xk)
                if abs(residual) <= quantile:
                    xk = xk + residual * rows[0]
            quantile_fraction = float(
                np.mean(np.abs(test_residuals[1:]) <= quantile)
            )
        case _:
            raise ValueError(f"Unknown corruption_type: {corruption_type}")

    return xk, quantile_fraction


def run_qRK_subsample_D_vs_beta(
    D: int,
    T_max: int,
    x: np.ndarray,
    q: float,
    beta: float,
    n: int,
    c: float,
    corruption_type: str,
    c_min: float,
    c_max: float,
    s_min: float,
    s_max: float,
    quantile_noise_min: float = 1e16,
    quantile_noise_max: float = 1e16,
    update_noise_min: float = 1e8,
    update_noise_max: float = 1e8,
    random_seed: int | None = None,
) -> bool:
    """Return whether one trial meets the fixed-horizon success criterion."""
    if random_seed is not None:
        np.random.seed(random_seed)
    xk = np.zeros_like(x)
    for _ in range(T_max):
        xk, _ = streaming_subsampled_qRK_step(
            x,
            xk,
            q,
            beta,
            D,
            corruption_type,
            c_min,
            c_max,
            s_min,
            s_max,
            quantile_noise_min=quantile_noise_min,
            quantile_noise_max=quantile_noise_max,
            update_noise_min=update_noise_min,
            update_noise_max=update_noise_max,
        )
    relative_error = np.linalg.norm(xk - x) ** 2 / np.linalg.norm(x) ** 2
    return bool(relative_error < (1.0 - c / n) ** T_max)


def run_qRK_subsample_D_vs_T(
    D: int,
    T_max: int,
    T_intervals: int,
    x: np.ndarray,
    q: float,
    beta: float,
    n: int,
    c: float,
    corruption_type: str,
    c_min: float,
    c_max: float,
    s_min: float,
    s_max: float,
    quantile_noise_min: float = 1e16,
    quantile_noise_max: float = 1e16,
    update_noise_min: float = 1e8,
    update_noise_max: float = 1e8,
    random_seed: int | None = None,
) -> tuple[np.ndarray, float]:
    """Return success indicators at each requested iteration checkpoint."""
    if random_seed is not None:
        np.random.seed(random_seed)
    successes = np.zeros(T_max // T_intervals)
    xk = np.zeros_like(x)
    quantile_fraction = float("nan")
    for iteration in range(1, T_max + 1):
        xk, quantile_fraction = streaming_subsampled_qRK_step(
            x,
            xk,
            q,
            beta,
            D,
            corruption_type,
            c_min,
            c_max,
            s_min,
            s_max,
            quantile_noise_min=quantile_noise_min,
            quantile_noise_max=quantile_noise_max,
            update_noise_min=update_noise_min,
            update_noise_max=update_noise_max,
        )
        if iteration % T_intervals != 0:
            continue
        index = iteration // T_intervals - 1
        relative_error = np.linalg.norm(xk - x) ** 2 / np.linalg.norm(x) ** 2
        successes[index] = relative_error <= (1.0 - c / n) ** iteration
    return successes, quantile_fraction
