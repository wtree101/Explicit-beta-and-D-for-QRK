from dataclasses import dataclass
from multiprocessing import Pool
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from heatmap_data_generation.heatmapDataGeneration import (
    streaming_subsampled_qRK_step,
    validate_oblivious_large_noise,
)


@dataclass(frozen=True)
class ExperimentConfig:
    n: int = 100
    q: float = 0.8
    beta: float = 0.005
    T: int = 20_000
    record_every: int = 100
    D_list: tuple[int, ...] = tuple(range(1, 11))
    num_trials: int = 5
    num_workers: int = 5
    corruption_type: str = "oblivious_large"
    fixed_c: float = 0.05
    c_min: float = 0
    c_max: float = 100
    s_min: float = 0.0
    s_max: float = 10.0
    quantile_noise_min: float = -2000
    quantile_noise_max: float = 2000
    update_noise_min: float = -1000
    update_noise_max: float = 1000
    seed: int = 20260812
    lower_percentile: float = 10.0
    upper_percentile: float = 90.0


def run_trial(task: tuple[int, int, np.ndarray, ExperimentConfig]) -> tuple[int, int, np.ndarray]:
    D, trial_index, x, config = task
    np.random.seed(config.seed + D * 100_000 + trial_index)

    xk = np.zeros_like(x)
    errors = np.empty(config.T // config.record_every + 1)
    errors[0] = 1.0
    error_index = 1

    for iteration in range(1, config.T + 1):
        xk, _ = streaming_subsampled_qRK_step(
            x,
            xk,
            config.q,
            config.beta,
            D,
            config.corruption_type,
            config.c_min,
            config.c_max,
            config.s_min,
            config.s_max,
            quantile_noise_min=config.quantile_noise_min,
            quantile_noise_max=config.quantile_noise_max,
            update_noise_min=config.update_noise_min,
            update_noise_max=config.update_noise_max,
        )
        if iteration % config.record_every == 0:
            errors[error_index] = np.linalg.norm(xk - x) ** 2 / np.linalg.norm(x) ** 2
            error_index += 1

    return D, trial_index, errors


def run_experiment(config: ExperimentConfig) -> tuple[np.ndarray, np.ndarray]:
    if config.T % config.record_every != 0:
        raise ValueError("T must be divisible by record_every")
    if config.corruption_type == "oblivious_large":
        validate_oblivious_large_noise(
            config.quantile_noise_min,
            config.quantile_noise_max,
            config.update_noise_min,
            config.update_noise_max,
        )

    rng = np.random.default_rng(config.seed)
    x = rng.normal(size=config.n)
    x /= np.linalg.norm(x)

    tasks = [
        (D, trial_index, x, config)
        for D in config.D_list
        for trial_index in range(config.num_trials)
    ]
    errors = np.empty(
        (len(config.D_list), config.num_trials, config.T // config.record_every + 1)
    )

    if config.num_workers == 1:
        results = map(run_trial, tasks)
        for D, trial_index, trial_errors in results:
            errors[config.D_list.index(D), trial_index] = trial_errors
    else:
        with Pool(processes=config.num_workers) as pool:
            for D, trial_index, trial_errors in pool.imap_unordered(run_trial, tasks):
                errors[config.D_list.index(D), trial_index] = trial_errors

    return x, errors


def plot_results(
    config: ExperimentConfig,
    errors: np.ndarray,
) -> plt.Figure:
    iterations = np.arange(0, config.T + config.record_every, config.record_every)
    colors = plt.colormaps["viridis"](np.linspace(0.05, 0.95, len(config.D_list)))
    figure, error_axis = plt.subplots(figsize=(11, 6))

    for index, (D, color) in enumerate(zip(config.D_list, colors)):
        mean_error = np.mean(errors[index], axis=0)
        lower_error = np.percentile(errors[index], config.lower_percentile, axis=0)
        upper_error = np.percentile(errors[index], config.upper_percentile, axis=0)
        error_axis.plot(iterations, mean_error, color=color, linewidth=2, label=f"D={D}")
        error_axis.fill_between(
            iterations,
            lower_error,
            upper_error,
            color=color,
            alpha=0.12,
            linewidth=0,
        )

    fixed_reference = np.power(1.0 - config.fixed_c / config.n, iterations)
    error_axis.plot(
        iterations,
        fixed_reference,
        color="black",
        linestyle="--",
        linewidth=1.5,
        label=f"fixed-c reference (c={config.fixed_c:g})",
    )

    error_axis.set_yscale("log")
    error_axis.set_xlabel("iteration t")
    error_axis.set_ylabel("squared relative error")
    error_axis.set_title(
        f"Convergence by D ({config.corruption_type}, {config.num_trials} trials)\n"
        "solid: empirical mean, band: percentile interval, dashed: success reference"
    )
    error_axis.grid(True, which="both", alpha=0.25)
    error_axis.legend(ncol=2, fontsize=9)

    figure.tight_layout()
    return figure


def main() -> None:
    config = ExperimentConfig()
    output_dir = Path("figure")
    output_dir.mkdir(exist_ok=True)

    x, errors = run_experiment(config)
    figure = plot_results(config, errors)

    stem = (
        f"convergence_D__noise={config.corruption_type}__T={config.T}"
        f"__trials={config.num_trials}__c={config.fixed_c:1.0e}"
        f"__quantile_noise={config.quantile_noise_min:g}_{config.quantile_noise_max:g}"
        f"__update_noise={config.update_noise_min:g}_{config.update_noise_max:g}"
    )
    figure_path = output_dir / f"{stem}.png"
    data_path = output_dir / f"{stem}.npz"
    figure.savefig(figure_path, dpi=220, bbox_inches="tight")
    np.savez(
        data_path,
        x=x,
        errors=errors,
        D_list=np.asarray(config.D_list),
        iterations=np.arange(0, config.T + config.record_every, config.record_every),
        c_success=config.fixed_c,
        quantile_noise_min=config.quantile_noise_min,
        quantile_noise_max=config.quantile_noise_max,
        update_noise_min=config.update_noise_min,
        update_noise_max=config.update_noise_max,
    )

    print(f"Saved figure: {figure_path}")
    print(f"Saved data:   {data_path}")
    print(f"Fixed success c: {config.fixed_c}")


if __name__ == "__main__":
    main()
