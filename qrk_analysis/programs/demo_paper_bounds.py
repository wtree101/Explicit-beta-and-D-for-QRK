"""Generate paper figures and optional Gaussian-noise exploration figures.

Run from ``codes``. With no figure-group option, the script preserves the
historical behavior and generates only the four paper figures::

    python -m qrk_analysis.programs.demo_paper_bounds --recompute
    python -m qrk_analysis.programs.demo_paper_bounds --paper --recompute
    python -m qrk_analysis.programs.demo_paper_bounds --extra --recompute
    python -m qrk_analysis.programs.demo_paper_bounds --paper --extra
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from functools import partial
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

from qrk_analysis.feasibility.check import (
    check_feasibility_conditions_C_sup_revised,
    check_feasibility_conditions_random_sup_revised,
)
from qrk_analysis.programs.program1 import program1_largest_beta
from qrk_analysis.programs.program2 import program2_smallest_D_2


PROJECT_ROOT = Path(__file__).resolve().parents[2]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
PAPER_DATA_DIR = PROJECT_ROOT / "figure" / "paper_bounds" / "cache"
PAPER_FIGURE_DIR = REPOSITORY_ROOT / "PR_quantile" / "figures"
EXTRA_DIR = PROJECT_ROOT / "figure" / "paper_extra"

T = 20_000
DELTA_F = 0.1
COMPARISON_Q = 0.75
C_MIN = 0.0
C_MAX = 20.0
NUM_C = 200
NUM_QUANTILES = 10
SIGMA_MIN = 0.01
SIGMA_MAX = 10.0
NUM_SIGMAS = 20

MASSART_COLOR = "#176B87"
OBLIVIOUS_COLOR = "#C24D2C"
GAUSSIAN_COLOR = "#577A3A"

OBLIVIOUS_CHECK = partial(
    check_feasibility_conditions_C_sup_revised,
    num_grid_Q=NUM_QUANTILES,
    C_min=C_MIN,
    C_max=C_MAX,
    num_points_C=NUM_C,
)
GAUSSIAN_CHECK = partial(
    check_feasibility_conditions_random_sup_revised,
    num_grid_Q=NUM_QUANTILES,
    sigma_min=SIGMA_MIN,
    sigma_max=SIGMA_MAX,
    num_points_C=NUM_SIGMAS,
)


def configure_plot_style() -> None:
    """Set a compact style that remains legible in a half-width figure."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "lines.linewidth": 2.0,
            "lines.markersize": 4.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def load_or_compute(
    name: str,
    x_values: np.ndarray,
    compute: Callable[[float], float],
    *,
    data_dir: Path,
    recompute: bool,
    progress_label: str | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Load a two-column curve cache or compute and save it."""
    data_dir.mkdir(parents=True, exist_ok=True)
    path = data_dir / f"{name}.csv"
    if path.exists() and not recompute:
        data = np.atleast_2d(np.loadtxt(path, delimiter=",", skiprows=1))
        return data[:, 0], data[:, 1]

    y_values = []
    total = len(x_values)
    for index, value in enumerate(x_values, start=1):
        if progress_label is not None:
            print(f"{progress_label}: {index}/{total} (x={value:.4g})", flush=True)
        y_values.append(compute(float(value)))

    y_array = np.asarray(y_values, dtype=float)
    np.savetxt(
        path,
        np.column_stack((x_values, y_array)),
        delimiter=",",
        header="x,y",
        comments="",
    )
    return x_values, y_array


def largest_massart_beta(q: float) -> float:
    result = program1_largest_beta(q, np.inf, 1, 1.0, beta_tol=1e-8)
    return np.nan if result is None else result


def largest_oblivious_beta(q: float) -> float:
    result = program1_largest_beta(
        q,
        np.inf,
        1,
        1.0,
        feasibility_check=OBLIVIOUS_CHECK,
        beta_tol=1e-5,
    )
    return np.nan if result is None else result


def largest_gaussian_beta(q: float) -> float:
    result = program1_largest_beta(
        q,
        np.inf,
        1,
        1.0,
        feasibility_check=GAUSSIAN_CHECK,
        beta_tol=1e-5,
    )
    return np.nan if result is None else result


def smallest_massart_D(beta: float) -> float:
    result = program2_smallest_D_2(
        beta=beta,
        T=T,
        q=COMPARISON_Q,
        delta_f=DELTA_F,
        D_max=500,
        D_precision=0.05,
        c_target=0.0,
        num_grid=100,
    )["smallest_D"]
    return np.nan if result is None else np.ceil(result)


def smallest_oblivious_D(beta: float) -> float:
    result = program2_smallest_D_2(
        beta=beta,
        T=T,
        q=COMPARISON_Q,
        delta_f=DELTA_F,
        D_max=1_000,
        D_precision=0.1,
        c_target=0.0,
        feasibility_check=OBLIVIOUS_CHECK,
        num_grid=60,
    )["smallest_D"]
    return np.nan if result is None else np.ceil(result)


def smallest_gaussian_D(beta: float) -> float:
    result = program2_smallest_D_2(
        beta=beta,
        T=T,
        q=COMPARISON_Q,
        delta_f=DELTA_F,
        D_max=1_000,
        D_precision=0.1,
        c_target=0.0,
        feasibility_check=GAUSSIAN_CHECK,
        num_grid=60,
    )["smallest_D"]
    return np.nan if result is None else np.ceil(result)


def save_curve(
    x_values: np.ndarray,
    y_values: np.ndarray,
    *,
    output_dir: Path,
    filename: str,
    color: str,
    xlabel: str,
    ylabel: str,
    highlight_x: float | None = None,
    annotation_offset: tuple[int, int] = (7, 7),
) -> None:
    """Save one publication-style curve without a redundant internal title."""
    output_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(4.6, 3.35))
    mark_every = 2 if len(x_values) > 25 else 1
    ax.plot(
        x_values,
        y_values,
        "o-",
        color=color,
        markeredgewidth=0,
        markevery=mark_every,
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.22, linewidth=0.7)
    ax.margins(x=0.025, y=0.08)
    ax.xaxis.set_major_locator(MaxNLocator(nbins=5))

    if highlight_x is not None:
        index = int(np.argmin(np.abs(x_values - highlight_x)))
        x_point = x_values[index]
        y_point = y_values[index]
        ax.scatter([x_point], [y_point], s=34, color=color, edgecolor="white", zorder=3)
        ax.annotate(
            rf"$({x_point:.2f},\,{y_point:.3g})$",
            xy=(x_point, y_point),
            xytext=annotation_offset,
            textcoords="offset points",
            fontsize=10,
        )

    fig.tight_layout(pad=0.4)
    fig.savefig(output_dir / filename, bbox_inches="tight")
    plt.close(fig)


def generate_paper_figures(*, recompute: bool) -> None:
    """Generate the four figures included in the paper."""
    configure_plot_style()

    q_massart, beta_massart = load_or_compute(
        "massart_beta_vs_q",
        np.linspace(0.10, 0.95, 35),
        largest_massart_beta,
        data_dir=PAPER_DATA_DIR,
        recompute=recompute,
    )
    beta_grid_massart, D_massart = load_or_compute(
        "massart_D_vs_beta_q075_comparison",
        np.linspace(0.001, 0.020, 20),
        smallest_massart_D,
        data_dir=PAPER_DATA_DIR,
        recompute=recompute,
    )
    q_oblivious, beta_oblivious = load_or_compute(
        "oblivious_beta_vs_q",
        np.linspace(0.10, 0.95, 18),
        largest_oblivious_beta,
        data_dir=PAPER_DATA_DIR,
        recompute=recompute,
    )
    beta_grid_oblivious, D_oblivious = load_or_compute(
        "oblivious_D_vs_beta_q075_comparison",
        np.linspace(0.001, 0.020, 20),
        smallest_oblivious_D,
        data_dir=PAPER_DATA_DIR,
        recompute=recompute,
    )
    massart_peak_index = int(np.nanargmax(beta_massart))
    massart_peak_q = float(q_massart[massart_peak_index])

    save_curve(
        q_massart,
        beta_massart,
        output_dir=PAPER_FIGURE_DIR,
        filename="massart_beta_vs_q.pdf",
        color=MASSART_COLOR,
        xlabel=r"Quantile parameter $q$",
        ylabel=r"Maximum corruption $\beta^*(q)$",
        highlight_x=massart_peak_q,
        annotation_offset=(-22, -22),
    )
    save_curve(
        beta_grid_massart,
        D_massart,
        output_dir=PAPER_FIGURE_DIR,
        filename="massart_D_vs_beta.pdf",
        color=MASSART_COLOR,
        xlabel=r"Corruption rate $\beta$",
        ylabel=r"Minimum subsample size $D^*$",
        highlight_x=0.01,
    )
    save_curve(
        q_oblivious,
        beta_oblivious,
        output_dir=PAPER_FIGURE_DIR,
        filename="oblivious_beta_vs_q.pdf",
        color=OBLIVIOUS_COLOR,
        xlabel=r"Quantile parameter $q$",
        ylabel=r"Maximum corruption $\beta^*_{\mathrm{obl}}(q)$",
        highlight_x=0.65,
    )
    save_curve(
        beta_grid_oblivious,
        D_oblivious,
        output_dir=PAPER_FIGURE_DIR,
        filename="oblivious_D_vs_beta.pdf",
        color=OBLIVIOUS_COLOR,
        xlabel=r"Corruption rate $\beta$",
        ylabel=r"Minimum subsample size $D^*_{\mathrm{obl}}$",
        highlight_x=0.01,
    )


def generate_extra_figures(*, recompute: bool) -> None:
    """Generate Gaussian worst-sigma figures outside the paper tree."""
    configure_plot_style()

    q_gaussian, beta_gaussian = load_or_compute(
        "gaussian_sup_beta_vs_q_sigma001_10_n20",
        np.linspace(0.10, 0.95, 18),
        largest_gaussian_beta,
        data_dir=EXTRA_DIR,
        recompute=recompute,
        progress_label="Gaussian beta curve",
    )
    beta_grid_gaussian, D_gaussian = load_or_compute(
        "gaussian_sup_D_vs_beta_q075_sigma001_10_n20",
        np.linspace(0.001, 0.020, 20),
        smallest_gaussian_D,
        data_dir=EXTRA_DIR,
        recompute=recompute,
        progress_label="Gaussian D curve",
    )

    finite_indices = np.flatnonzero(np.isfinite(beta_gaussian))
    highlight_q = (
        float(q_gaussian[finite_indices[np.argmax(beta_gaussian[finite_indices])]])
        if finite_indices.size
        else None
    )
    save_curve(
        q_gaussian,
        beta_gaussian,
        output_dir=EXTRA_DIR,
        filename="gaussian_sup_beta_vs_q.pdf",
        color=GAUSSIAN_COLOR,
        xlabel=r"Quantile parameter $q$",
        ylabel=r"Maximum corruption $\beta^*_{\mathrm{Gaussian}}(q)$",
        highlight_x=highlight_q,
        annotation_offset=(-65, 10),
    )
    save_curve(
        beta_grid_gaussian,
        D_gaussian,
        output_dir=EXTRA_DIR,
        filename="gaussian_sup_D_vs_beta.pdf",
        color=GAUSSIAN_COLOR,
        xlabel=r"Corruption rate $\beta$",
        ylabel=r"Minimum subsample size $D^*_{\mathrm{Gaussian}}$",
        highlight_x=0.01,
        annotation_offset=(7, -20),
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Generate the four figures included in the paper.",
    )
    parser.add_argument(
        "--extra",
        action="store_true",
        help="Generate the supplementary Gaussian worst-sigma figures.",
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Recompute selected curves instead of using cached CSV files.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    arguments = build_parser().parse_args(argv)
    generate_paper = arguments.paper or not (arguments.paper or arguments.extra)

    if generate_paper:
        generate_paper_figures(recompute=arguments.recompute)
    if arguments.extra:
        generate_extra_figures(recompute=arguments.recompute)


if __name__ == "__main__":
    main()
