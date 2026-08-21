"""Overlay c=0 theoretical boundaries on the four cached paper heatmaps.

This exploratory program recomputes only theoretical ``D^*`` curves. It reads
the existing Monte Carlo matrices and never writes to the paper figure tree.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np

from heatmap_data_display.plot_heatmaps import create_figure, load_heatmap_data
from heatmap_data_display.profiles import HEATMAP_PROFILES, HeatmapProfile
from qrk_analysis.feasibility.check import check_feasibility_conditions_C_sup_revised
from qrk_analysis.upper_bound import smallest_continuous_D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "figure" / "heatmaps" / "experimental_c0_overlay"
CACHE_DIR = OUTPUT_DIR / "cache"


@dataclass(frozen=True)
class OverlayConfig:
    """Parameters needed to reproduce one c=0 comparison curve."""

    profile_name: str
    model: str
    existing_c: float
    beta: float | None = None
    T: int | None = None
    d_max: float | None = None

    @property
    def profile(self) -> HeatmapProfile:
        return HEATMAP_PROFILES[self.profile_name]

    @property
    def cache_path(self) -> Path:
        return CACHE_DIR / f"{self.profile_name}__c_target=0.csv"

    @property
    def output_path(self) -> Path:
        return OUTPUT_DIR / f"{self.profile_name}__c0_overlay.pdf"


CONFIGS = (
    OverlayConfig("d-vs-t-massart", "massart", 0.01, beta=0.01),
    OverlayConfig("d-vs-t-oblivious", "oblivious", 0.05, beta=0.01),
    OverlayConfig("d-vs-beta-massart", "massart", 0.01, T=20_000, d_max=80.0),
    OverlayConfig("d-vs-beta-oblivious", "oblivious", 0.05, T=20_000, d_max=80.0),
)


def _compute_point(task: tuple[str, float, int, float]) -> float:
    model, beta, T, q = task
    if beta == 0.0:
        return 0

    feasibility_check = None
    if model == "oblivious":
        feasibility_check = partial(
            check_feasibility_conditions_C_sup_revised,
            num_grid_Q=2,
            C_min=0.0,
            C_max=20.0,
            num_points_C=20,
        )
    result = smallest_continuous_D(
        beta=beta,
        T=T,
        q=q,
        delta_f=0.1,
        D_max=500,
        D_precision=0.05,
        c_target=0.0,
        num_grid=200,
        feasibility_check=feasibility_check,
    )
    value = result["smallest_D"]
    if value is None:
        raise RuntimeError(
            f"No c=0 boundary found below D=500 for model={model}, beta={beta}, T={T}."
        )
    return float(value)


def _tasks(config: OverlayConfig, x_values: np.ndarray) -> list[tuple[str, float, int, float]]:
    q = 0.8
    if config.profile.kind == "D_vs_T":
        if config.beta is None:
            raise ValueError(f"{config.profile_name} does not define a fixed beta.")
        return [(config.model, config.beta, int(T), q) for T in x_values]
    if config.T is None:
        raise ValueError(f"{config.profile_name} does not define a fixed T.")
    return [(config.model, float(beta), config.T, q) for beta in x_values]


def _load_cache(config: OverlayConfig, x_values: np.ndarray) -> np.ndarray | None:
    if not config.cache_path.is_file():
        return None
    cached = np.atleast_2d(np.loadtxt(config.cache_path, delimiter=",", skiprows=1))
    if cached.shape != (x_values.size, 2) or not np.allclose(cached[:, 0], x_values):
        raise ValueError(f"Cache grid does not match profile {config.profile_name}: {config.cache_path}")
    if not np.all(np.isfinite(cached[:, 1])):
        raise ValueError(f"Cache contains non-finite D values: {config.cache_path}")
    return cached[:, 1]


def _save_cache(config: OverlayConfig, x_values: np.ndarray, boundary: np.ndarray) -> None:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        config.cache_path,
        np.column_stack((x_values, boundary)),
        delimiter=",",
        header="x,D_star_c_target_0",
        comments="",
        fmt=("%.12g", "%.8f"),
    )


def compute_boundary(
    config: OverlayConfig,
    x_values: np.ndarray,
    *,
    recompute: bool,
    workers: int,
) -> np.ndarray:
    """Load or compute one continuous c=0 theoretical boundary."""
    if not recompute:
        cached = _load_cache(config, x_values)
        if cached is not None:
            print(f"[{config.profile_name}] loaded {config.cache_path}", flush=True)
            return cached

    tasks = _tasks(config, x_values)
    print(f"[{config.profile_name}] computing {len(tasks)} points", flush=True)
    if workers == 1:
        values = []
        for index, task in enumerate(tasks, start=1):
            values.append(_compute_point(task))
            if index == 1 or index % 10 == 0 or index == len(tasks):
                print(f"[{config.profile_name}] {index}/{len(tasks)}", flush=True)
    else:
        with ThreadPoolExecutor(max_workers=workers) as executor:
            values = []
            for index, value in enumerate(executor.map(_compute_point, tasks), start=1):
                values.append(value)
                if index == 1 or index % 10 == 0 or index == len(tasks):
                    print(f"[{config.profile_name}] {index}/{len(tasks)}", flush=True)

    boundary = np.asarray(values, dtype=float)
    _save_cache(config, x_values, boundary)
    return boundary


def render_overlay(config: OverlayConfig, boundary_c0: np.ndarray) -> Path:
    """Render one comparison without touching the stable paper figure."""
    data = load_heatmap_data(config.profile)
    figure = create_figure(config.profile, data, d_max=config.d_max)
    axis = figure.axes[0]
    existing_line = axis.lines[0]
    existing_line.set_label(fr"$D^*$ ($c={config.existing_c:g}$)")
    (comparison_line,) = axis.plot(
        data.x_values,
        boundary_c0,
        color="#ffe066",
        linestyle="--",
        linewidth=2.4,
        label=r"$D^*$ ($c=0$)",
        zorder=4,
    )
    comparison_line.set_path_effects(
        [path_effects.Stroke(linewidth=4.0, foreground="black"), path_effects.Normal()]
    )
    axis.legend(loc="best", framealpha=0.92)
    figure.tight_layout(pad=0.4)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    temporary = config.output_path.with_name(f".{config.output_path.stem}.tmp.pdf")
    try:
        figure.savefig(temporary, format="pdf", bbox_inches="tight")
        os.replace(temporary, config.output_path)
    finally:
        plt.close(figure)
        temporary.unlink(missing_ok=True)
    return config.output_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Overlay c=0 theoretical curves on the four cached heatmaps."
    )
    parser.add_argument(
        "--recompute",
        action="store_true",
        help="Ignore cached c=0 curves and recompute all four boundaries.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=min(8, os.cpu_count() or 1),
        help="Parallel theoretical calculations; defaults to at most 8 workers.",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")

    outputs: list[Path] = []
    for config in CONFIGS:
        data = load_heatmap_data(config.profile)
        boundary = compute_boundary(
            config,
            data.x_values,
            recompute=args.recompute,
            workers=args.workers,
        )
        outputs.append(render_overlay(config, boundary))

    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
