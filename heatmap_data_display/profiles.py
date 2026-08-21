"""Explicit input profiles for cached heatmap figures."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Literal

HeatmapKind = Literal["D_vs_T", "D_vs_beta"]


@dataclass(frozen=True)
class HeatmapProfile:
    """Files, coordinates, and output names for one heatmap panel."""

    name: str
    kind: HeatmapKind
    model: str
    success_file: str
    boundary_file: str
    preview_file: str
    paper_file: str
    x_grid_file: str | None = None
    d_grid_file: str | None = None
    x_start: float | None = None
    x_stop: float | None = None
    x_step: float | None = None
    d_start: float | None = None
    d_stop: float | None = None
    d_step: float | None = None

    def resolve(self, data_dir: Path, filename: str) -> Path:
        """Resolve a profile filename relative to the raw-data directory."""
        return data_dir / filename


_D_VS_T_MASSART_SUFFIX = (
    "__n=100__q=80__beta=1__D_min=1__D_max=30__c=1e-02"
    "__num_samples=100__T_intervals=100__T_max=20000"
    "__corruption_type=adversarial.txt"
)
_D_VS_T_OBLIVIOUS_SUFFIX = (
    "__n=100__q=80__beta=1__D_min=1__D_max=30__c=5e-02"
    "__num_samples=100__T_intervals=100__T_max=20000"
    "__corruption_type=sup_c.txt"
)
_D_VS_BETA_MASSART_SUFFIX = (
    "__n=100__q=80__beta_min=0__beta_max=2__D_min=2__D_max=120"
    "__c=1e-02__num_samples=100__T_max=20000"
    "__corruption_type=adversarial.txt"
)
_D_VS_BETA_OBLIVIOUS_SUFFIX = (
    "__n=100__q=80__beta_min=0__beta_max=2__D_min=2__D_max=120"
    "__c=5e-02__num_samples=100__T_max=20000"
    "__corruption_type=sup_c.txt"
)


HEATMAP_PROFILES: dict[str, HeatmapProfile] = {
    "d-vs-t-massart": HeatmapProfile(
        name="d-vs-t-massart",
        kind="D_vs_T",
        model="Massart",
        success_file=f"D_vs_T{_D_VS_T_MASSART_SUFFIX}",
        boundary_file=f"D_vs_T__D_MIN{_D_VS_T_MASSART_SUFFIX}",
        preview_file="D_vs_T_massart.pdf",
        paper_file="D_vs_T_massart.pdf",
        x_start=100.0,
        x_stop=20_000.0,
        x_step=100.0,
        d_start=1.0,
        d_stop=30.0,
        d_step=1.0,
    ),
    "d-vs-t-oblivious": HeatmapProfile(
        name="d-vs-t-oblivious",
        kind="D_vs_T",
        model="Oblivious",
        success_file=f"D_vs_T{_D_VS_T_OBLIVIOUS_SUFFIX}",
        boundary_file=f"D_vs_T__D_MIN{_D_VS_T_OBLIVIOUS_SUFFIX}",
        preview_file="D_vs_T_oblivious.pdf",
        paper_file="D_vs_T_oblivious.pdf",
        x_start=100.0,
        x_stop=20_000.0,
        x_step=100.0,
        d_start=1.0,
        d_stop=30.0,
        d_step=1.0,
    ),
    "d-vs-beta-massart": HeatmapProfile(
        name="d-vs-beta-massart",
        kind="D_vs_beta",
        model="Massart",
        success_file=f"D_vs_beta{_D_VS_BETA_MASSART_SUFFIX}",
        boundary_file=f"D_vs_beta__D_min{_D_VS_BETA_MASSART_SUFFIX}",
        x_grid_file=f"D_vs_beta__beta_samples{_D_VS_BETA_MASSART_SUFFIX}",
        d_grid_file=f"D_vs_beta__D_samples{_D_VS_BETA_MASSART_SUFFIX}",
        preview_file="D_vs_beta_massart.pdf",
        paper_file="D_vs_beta_massart.pdf",
    ),
    "d-vs-beta-oblivious": HeatmapProfile(
        name="d-vs-beta-oblivious",
        kind="D_vs_beta",
        model="Oblivious",
        success_file=f"D_vs_beta{_D_VS_BETA_OBLIVIOUS_SUFFIX}",
        boundary_file=f"D_vs_beta__D_min{_D_VS_BETA_OBLIVIOUS_SUFFIX}",
        x_grid_file=f"D_vs_beta__beta_samples{_D_VS_BETA_OBLIVIOUS_SUFFIX}",
        d_grid_file=f"D_vs_beta__D_samples{_D_VS_BETA_OBLIVIOUS_SUFFIX}",
        preview_file="D_vs_beta_oblivious.pdf",
        paper_file="D_vs_beta_oblivious.pdf",
    ),
}
