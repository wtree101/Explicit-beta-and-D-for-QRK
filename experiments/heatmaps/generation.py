"""Multiprocessing sweeps that produce QRK heatmap matrices."""

from __future__ import annotations

import os
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from tqdm import tqdm

from qrk_analysis.upper_bound import smallest_continuous_D

from .io import save_heat_map_matrix
from .simulation import (
    run_qRK_subsample_D_vs_T,
    run_qRK_subsample_D_vs_beta,
    validate_oblivious_large_noise,
)
from .theory import make_feasibility_check


def generate_heat_map_matrix(
    D_vs_TYPE: str,
    D_sample_sizes: np.ndarray,
    num_samples: int,
    T_max: int,
    x: np.ndarray,
    q: float,
    n: int,
    c: float,
    corruption_type: str,
    beta: float = 0.0,
    T_intervals: int = 1,
    beta_samples: np.ndarray | None = None,
    c_min: float = 0.0,
    c_max: float = 1.0,
    s_min: float = 0.0,
    s_max: float = 1.0,
    quantile_noise_min: float = 1e16,
    quantile_noise_max: float = 1e16,
    update_noise_min: float = 1e8,
    update_noise_max: float = 1e8,
    feasibility_C_min: float = 0.0,
    feasibility_C_max: float = 100.0,
    num_workers: int | None = None,
    random_seed: int | None = None,
) -> None:
    """Generate empirical success matrices and theoretical ``D_min`` curves."""
    beta_values = np.zeros(1) if beta_samples is None else np.asarray(beta_samples)
    delta_f = 0.1
    D_max = 500
    if corruption_type == "oblivious_large":
        validate_oblivious_large_noise(
            quantile_noise_min,
            quantile_noise_max,
            update_noise_min,
            update_noise_max,
        )
    feasibility_check = make_feasibility_check(
        corruption_type,
        feasibility_C_min=feasibility_C_min,
        feasibility_C_max=feasibility_C_max,
    )
    if num_samples < 1:
        raise ValueError("num_samples must be at least 1")
    if num_workers is not None and num_workers < 1:
        raise ValueError("num_workers must be at least 1 or None")

    available_workers = os.cpu_count() or 1
    worker_count = min(num_samples, num_workers or available_workers)
    seed_entropy = np.random.SeedSequence(random_seed).entropy

    def sample_seeds(parameter_index: int) -> list[int]:
        return [
            int(
                np.random.SeedSequence(
                    [seed_entropy, parameter_index, sample_index]
                ).generate_state(1)[0]
            )
            for sample_index in range(num_samples)
        ]

    match D_vs_TYPE:
        case "D_vs_T":
            Path("q_e").mkdir(parents=True, exist_ok=True)
            Path("q_e/most_recent_q_e.txt").write_text("")
            mean_success = np.zeros((len(D_sample_sizes), T_max // T_intervals))
            pool = Pool(processes=worker_count)
            try:
                for D_position, D in enumerate(tqdm(D_sample_sizes)):
                    sample_results = pool.starmap(
                        run_qRK_subsample_D_vs_T,
                        [
                            (
                                D,
                                T_max,
                                T_intervals,
                                x,
                                q,
                                beta,
                                n,
                                c,
                                corruption_type,
                                c_min,
                                c_max,
                                s_min,
                                s_max,
                                quantile_noise_min,
                                quantile_noise_max,
                                update_noise_min,
                                update_noise_max,
                                seed,
                            )
                            for seed in sample_seeds(D_position)
                        ],
                    )
                    successes = np.array([result[0] for result in sample_results])
                    fractions = np.array([result[1] for result in sample_results])
                    mean_success[D_position] = np.mean(successes, axis=0)
                    with Path("q_e/most_recent_q_e.txt").open("a") as handle:
                        handle.write(f"(D:{D}) {np.mean(fractions)}\n")
            finally:
                pool.close()
                pool.join()

            common = dict(
                D_vs_TYPE="D_vs_T",
                n=n,
                D_sample_sizes=D_sample_sizes,
                num_samples=num_samples,
                T_max=T_max,
                q=q,
                c=c,
                corruption_type=corruption_type,
                beta=beta,
                T_intervals=T_intervals,
            )
            save_heat_map_matrix(data_type="", mean_success=mean_success, **common)
            D_min_values = np.zeros(T_max // T_intervals)
            for index in tqdm(range(T_max // T_intervals)):
                result = smallest_continuous_D(
                    beta,
                    (index + 1) * T_intervals,
                    q,
                    D_max=D_max,
                    delta_f=delta_f,
                    c_target=c,
                    D_precision=0.05,
                    num_grid=200,
                    feasibility_check=feasibility_check,
                )
                D_min_values[index] = result["smallest_D"]
            save_heat_map_matrix(
                data_type="D_min",
                mean_success=D_min_values[:, np.newaxis],
                **common,
            )
        case "D_vs_beta":
            mean_success = np.zeros((len(D_sample_sizes), len(beta_values)))
            pool = Pool(processes=worker_count)
            try:
                position = 0
                for D_index, D in enumerate(tqdm(D_sample_sizes)):
                    for beta_index, beta_value in enumerate(beta_values):
                        results = pool.starmap(
                            run_qRK_subsample_D_vs_beta,
                            [
                                (
                                    D,
                                    T_max,
                                    x,
                                    q,
                                    beta_value,
                                    n,
                                    c,
                                    corruption_type,
                                    c_min,
                                    c_max,
                                    s_min,
                                    s_max,
                                    quantile_noise_min,
                                    quantile_noise_max,
                                    update_noise_min,
                                    update_noise_max,
                                    seed,
                                )
                                for seed in sample_seeds(position)
                            ],
                        )
                        mean_success[D_index, beta_index] = np.mean(results)
                        position += 1
            finally:
                pool.close()
                pool.join()

            common = dict(
                D_vs_TYPE="D_vs_beta",
                n=n,
                D_sample_sizes=D_sample_sizes,
                num_samples=num_samples,
                T_max=T_max,
                q=q,
                c=c,
                corruption_type=corruption_type,
                beta_samples=beta_values,
            )
            save_heat_map_matrix(data_type="", mean_success=mean_success, **common)
            D_min_values = np.zeros(len(beta_values))
            for index, beta_value in enumerate(tqdm(beta_values)):
                result = smallest_continuous_D(
                    beta_value,
                    T_max,
                    q,
                    D_max=D_max,
                    delta_f=delta_f,
                    c_target=c,
                    D_precision=0.05,
                    num_grid=200,
                    feasibility_check=feasibility_check,
                )
                D_min_values[index] = result["smallest_D"]
            save_heat_map_matrix(
                data_type="D_min", mean_success=D_min_values, **common
            )
            save_heat_map_matrix(
                data_type="D_samples",
                mean_success=np.asarray(D_sample_sizes)[:, np.newaxis],
                **common,
            )
            save_heat_map_matrix(
                data_type="beta_samples",
                mean_success=beta_values[:, np.newaxis],
                **common,
            )
        case _:
            raise ValueError(f"Unknown D_vs_TYPE: {D_vs_TYPE}")
