"""Render cached heatmap data as compact preview or paper PDFs.

This module only reads existing text files. It deliberately has no dependency
on the simulation and heatmap-generation package.
"""

from __future__ import annotations

import argparse
import os
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize, PowerNorm, TwoSlopeNorm

from .profiles import HEATMAP_PROFILES, HeatmapProfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
REPOSITORY_ROOT = PROJECT_ROOT.parents[1]
DEFAULT_DATA_DIR = PROJECT_ROOT / "heat_map_raw_data"
PREVIEW_DIR = PROJECT_ROOT / "figure" / "heatmaps"
PAPER_DIR = REPOSITORY_ROOT / "PR_quantile" / "figures" / "heat_maps"
ColorScale = Literal["linear", "threshold", "power"]


@dataclass(frozen=True)
class HeatmapData:
    """Validated arrays for one heatmap panel."""

    x_values: np.ndarray
    d_values: np.ndarray
    success: np.ndarray
    boundary: np.ndarray


@dataclass(frozen=True)
class PathOverrides:
    """Optional replacements for paths declared by a profile."""

    success: Path | None = None
    boundary: Path | None = None
    x_grid: Path | None = None
    d_grid: Path | None = None


@dataclass(frozen=True)
class ColorMapping:
    """Map success probabilities to colors while retaining true colorbar values."""

    scale: ColorScale = "linear"
    center: float = 0.9
    gamma: float = 2.0

    def normalization(self) -> Normalize:
        match self.scale:
            case "linear":
                return Normalize(vmin=0.0, vmax=1.0)
            case "threshold":
                if not 0.0 < self.center < 1.0:
                    raise ValueError("Color center must lie strictly between 0 and 1.")
                return TwoSlopeNorm(vmin=0.0, vcenter=self.center, vmax=1.0)
            case "power":
                if self.gamma <= 0.0:
                    raise ValueError("Color gamma must be positive.")
                return PowerNorm(gamma=self.gamma, vmin=0.0, vmax=1.0)

    def ticks(self) -> list[float] | None:
        if self.scale == "threshold":
            return sorted({0.0, 0.5, 0.8, self.center, 0.95, 1.0})
        return None


def configure_plot_style() -> None:
    """Use the compact typography of the paper's numerical-bound figures."""
    plt.rcParams.update(
        {
            "font.size": 12,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 11,
            "lines.linewidth": 2.0,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
        }
    )


def _load_array(path: Path, *, name: str) -> np.ndarray:
    if not path.is_file():
        raise FileNotFoundError(f"{name} file does not exist: {path}")
    values = np.asarray(np.loadtxt(path), dtype=float)
    if values.size == 0:
        raise ValueError(f"{name} file is empty: {path}")
    if not np.all(np.isfinite(values)):
        raise ValueError(f"{name} contains NaN or infinite values: {path}")
    return values


def _load_vector(path: Path, *, name: str) -> np.ndarray:
    values = np.ravel(_load_array(path, name=name))
    if values.size > 1 and not np.all(np.diff(values) > 0):
        raise ValueError(f"{name} must be strictly increasing: {path}")
    return values


def _regular_grid(
    *, start: float | None, stop: float | None, step: float | None, name: str
) -> np.ndarray:
    if start is None or stop is None or step is None:
        raise ValueError(f"Profile does not define a complete {name} grid.")
    if step <= 0 or stop < start:
        raise ValueError(f"Invalid {name} grid: start={start}, stop={stop}, step={step}.")
    count = round((stop - start) / step) + 1
    values = start + step * np.arange(count, dtype=float)
    if not np.isclose(values[-1], stop):
        raise ValueError(f"{name} grid does not end at {stop}.")
    return values


def _resolve_path(
    profile: HeatmapProfile,
    data_dir: Path,
    override: Path | None,
    profile_filename: str | None,
    *,
    name: str,
) -> Path:
    if override is not None:
        return override.expanduser().resolve()
    if profile_filename is None:
        raise ValueError(f"Profile {profile.name!r} does not define a {name} file.")
    return profile.resolve(data_dir, profile_filename)


def load_heatmap_data(
    profile: HeatmapProfile,
    *,
    data_dir: Path = DEFAULT_DATA_DIR,
    overrides: PathOverrides | None = None,
) -> HeatmapData:
    """Load and validate one explicit cached-data profile."""
    replacements = overrides or PathOverrides()
    success_path = _resolve_path(
        profile,
        data_dir,
        replacements.success,
        profile.success_file,
        name="success matrix",
    )
    boundary_path = _resolve_path(
        profile,
        data_dir,
        replacements.boundary,
        profile.boundary_file,
        name="theoretical boundary",
    )

    if replacements.x_grid is not None or profile.x_grid_file is not None:
        x_path = _resolve_path(
            profile,
            data_dir,
            replacements.x_grid,
            profile.x_grid_file,
            name="horizontal grid",
        )
        x_values = _load_vector(x_path, name="horizontal grid")
    else:
        x_values = _regular_grid(
            start=profile.x_start,
            stop=profile.x_stop,
            step=profile.x_step,
            name="horizontal",
        )

    if replacements.d_grid is not None or profile.d_grid_file is not None:
        d_path = _resolve_path(
            profile,
            data_dir,
            replacements.d_grid,
            profile.d_grid_file,
            name="D grid",
        )
        d_values = _load_vector(d_path, name="D grid")
    else:
        d_values = _regular_grid(
            start=profile.d_start,
            stop=profile.d_stop,
            step=profile.d_step,
            name="D",
        )

    raw_success = _load_array(success_path, name="success matrix")
    success = np.atleast_2d(raw_success)
    expected_shape = (d_values.size, x_values.size)
    if success.shape == expected_shape:
        pass
    elif success.T.shape == expected_shape and success.shape != expected_shape:
        success = success.T
    else:
        raise ValueError(
            f"Success matrix has shape {success.shape}; expected D x horizontal "
            f"grid = {expected_shape}."
        )
    if np.any((success < 0.0) | (success > 1.0)):
        raise ValueError("Success probabilities must lie in [0, 1].")

    boundary = np.ravel(_load_array(boundary_path, name="theoretical boundary"))
    if boundary.size != x_values.size:
        raise ValueError(
            f"Theoretical boundary has length {boundary.size}; expected "
            f"{x_values.size} to match the horizontal grid."
        )

    return HeatmapData(
        x_values=x_values,
        d_values=d_values,
        success=success,
        boundary=boundary,
    )


def cell_edges(values: np.ndarray) -> np.ndarray:
    """Return cell boundaries for a strictly increasing center grid."""
    if values.size == 1:
        return np.array([values[0] - 0.5, values[0] + 0.5])
    midpoints = (values[:-1] + values[1:]) / 2.0
    return np.concatenate(
        ([values[0] - (midpoints[0] - values[0])], midpoints, [
            values[-1] + (values[-1] - midpoints[-1])
        ])
    )


def _d_limits(
    data: HeatmapData, *, d_min: float | None, d_max: float | None
) -> tuple[float, float]:
    data_min, data_max = cell_edges(data.d_values)[[0, -1]]
    lower = float(data_min if d_min is None else d_min)
    upper = float(data_max if d_max is None else d_max)
    if lower >= upper:
        raise ValueError(f"D display range must satisfy d_min < d_max; got {lower}, {upper}.")
    if upper <= data_min or lower >= data_max:
        raise ValueError(
            f"D display range [{lower}, {upper}] does not overlap the data range "
            f"[{data_min}, {data_max}]."
        )
    return lower, upper


def create_figure(
    profile: HeatmapProfile,
    data: HeatmapData,
    *,
    d_min: float | None = None,
    d_max: float | None = None,
    color_mapping: ColorMapping | None = None,
) -> plt.Figure:
    """Create one compact heatmap figure from validated arrays."""
    configure_plot_style()
    mapping = color_mapping or ColorMapping()
    y_limits = _d_limits(data, d_min=d_min, d_max=d_max)
    figure, axis = plt.subplots(figsize=(4.6, 3.35))
    image = axis.pcolormesh(
        cell_edges(data.x_values),
        cell_edges(data.d_values),
        data.success,
        shading="flat",
        norm=mapping.normalization(),
        cmap="jet",
        # Rasterize only the color cells to avoid PDF hairline artifacts.
        # Labels, axes, and the theoretical curve remain vector graphics.
        rasterized=True,
    )
    (boundary_line,) = axis.plot(
        data.x_values,
        data.boundary,
        color="white",
        linewidth=2.3,
        label=r"Theoretical $D^*$",
        zorder=3,
    )
    boundary_line.set_path_effects(
        [path_effects.Stroke(linewidth=3.8, foreground="black"), path_effects.Normal()]
    )

    colorbar = figure.colorbar(
        image,
        ax=axis,
        pad=0.025,
        ticks=mapping.ticks(),
    )
    colorbar.set_label("Empirical success probability")
    axis.set_xlabel(r"$T$" if profile.kind == "D_vs_T" else r"$\beta$")
    axis.set_ylabel(r"$D$")
    axis.set_xlim(cell_edges(data.x_values)[[0, -1]])
    axis.set_ylim(y_limits)
    axis.legend(loc="best", framealpha=0.9)
    figure.tight_layout(pad=0.4)
    return figure


def _staging_path(output_path: Path) -> Path:
    return output_path.with_name(f".{output_path.stem}.tmp{output_path.suffix}")


def render_profiles(
    profiles_and_data: Sequence[tuple[HeatmapProfile, HeatmapData]],
    *,
    output_dir: Path,
    paper: bool,
    output_override: Path | None = None,
    d_min: float | None = None,
    d_max: float | None = None,
    color_mapping: ColorMapping | None = None,
) -> list[Path]:
    """Render all figures before atomically replacing any requested output."""
    if output_override is not None and len(profiles_and_data) != 1:
        raise ValueError("--output can only be used with one profile.")
    output_dir.mkdir(parents=True, exist_ok=True)
    staged: list[tuple[Path, Path]] = []
    try:
        for profile, data in profiles_and_data:
            if output_override is not None:
                destination = output_override.expanduser().resolve()
            else:
                filename = profile.paper_file if paper else profile.preview_file
                destination = output_dir / filename
            if destination.suffix.lower() != ".pdf":
                raise ValueError(f"Heatmap output must be a PDF: {destination}")
            destination.parent.mkdir(parents=True, exist_ok=True)
            temporary = _staging_path(destination)
            figure = create_figure(
                profile,
                data,
                d_min=d_min,
                d_max=d_max,
                color_mapping=color_mapping,
            )
            try:
                figure.savefig(temporary, format="pdf", bbox_inches="tight")
            finally:
                plt.close(figure)
            staged.append((temporary, destination))

        for temporary, destination in staged:
            os.replace(temporary, destination)
    except Exception:
        for temporary, _ in staged:
            temporary.unlink(missing_ok=True)
        raise
    return [destination for _, destination in staged]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot cached heatmap data without rerunning simulations."
    )
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(HEATMAP_PROFILES),
        help="Explicit cached-data profile; repeat to render several panels.",
    )
    parser.add_argument(
        "--list-profiles",
        action="store_true",
        help="List available profiles and exit.",
    )
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=DEFAULT_DATA_DIR,
        help="Directory containing cached text data.",
    )
    parser.add_argument("--success", type=Path, help="Override the success matrix path.")
    parser.add_argument(
        "--boundary", type=Path, help="Override the theoretical boundary path."
    )
    parser.add_argument("--x-grid", type=Path, help="Override the horizontal grid path.")
    parser.add_argument("--d-grid", type=Path, help="Override the D grid path.")
    parser.add_argument(
        "--d-min",
        type=float,
        help="Lower displayed D limit; defaults to the cached data range.",
    )
    parser.add_argument(
        "--d-max",
        type=float,
        help="Upper displayed D limit; defaults to the cached data range.",
    )
    parser.add_argument(
        "--color-scale",
        choices=("linear", "threshold", "power"),
        default="linear",
        help="Probability-to-color mapping; defaults to linear.",
    )
    parser.add_argument(
        "--color-center",
        type=float,
        default=0.9,
        help="Center for threshold color scaling; defaults to 0.9.",
    )
    parser.add_argument(
        "--color-gamma",
        type=float,
        default=2.0,
        help="Exponent for power color scaling; defaults to 2.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Override the preview output path; only valid for one profile.",
    )
    parser.add_argument(
        "--paper",
        action="store_true",
        help="Explicitly publish stable PDFs to the paper figure directory.",
    )
    return parser


def _print_profiles() -> None:
    for name in sorted(HEATMAP_PROFILES):
        profile = HEATMAP_PROFILES[name]
        print(f"{name}: {profile.kind}, {profile.model}")


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.list_profiles:
        _print_profiles()
        return 0
    if not args.profile:
        parser.error("at least one --profile is required (or use --list-profiles)")

    path_overrides = PathOverrides(
        success=args.success,
        boundary=args.boundary,
        x_grid=args.x_grid,
        d_grid=args.d_grid,
    )
    override_values = (
        path_overrides.success,
        path_overrides.boundary,
        path_overrides.x_grid,
        path_overrides.d_grid,
    )
    if any(value is not None for value in override_values) and len(args.profile) != 1:
        parser.error("path overrides can only be used with one --profile")
    if args.paper and args.output is not None:
        parser.error("--output cannot be combined with --paper")

    selected = [HEATMAP_PROFILES[name] for name in args.profile]
    loaded = [
        (
            profile,
            load_heatmap_data(
                profile,
                data_dir=args.data_dir.expanduser().resolve(),
                overrides=path_overrides,
            ),
        )
        for profile in selected
    ]
    output_dir = PAPER_DIR if args.paper else PREVIEW_DIR
    color_mapping = ColorMapping(
        scale=args.color_scale,
        center=args.color_center,
        gamma=args.color_gamma,
    )
    outputs = render_profiles(
        loaded,
        output_dir=output_dir,
        paper=args.paper,
        output_override=args.output,
        d_min=args.d_min,
        d_max=args.d_max,
        color_mapping=color_mapping,
    )
    for output in outputs:
        print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
