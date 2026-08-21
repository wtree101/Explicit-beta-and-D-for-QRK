"""Stable text output for heatmap matrices."""

from __future__ import annotations

from pathlib import Path

import numpy as np


def save_heat_map_matrix(
    D_vs_TYPE: str,
    data_type: str,
    mean_success: np.ndarray,
    n: int,
    D_sample_sizes: np.ndarray,
    num_samples: int,
    T_max: int,
    q: float,
    c_success: float,
    c_theory: float,
    corruption_type: str,
    beta: float = 0.0,
    T_intervals: int = 1,
    beta_samples: np.ndarray | None = None,
) -> None:
    """Write a matrix with explicit success and theory parameters."""
    beta_values = np.zeros(1) if beta_samples is None else np.asarray(beta_samples)
    output_directory = Path("heat_map_raw_data")
    output_directory.mkdir(parents=True, exist_ok=True)

    match D_vs_TYPE:
        case "D_vs_beta":
            suffixes = {
                "D_min": "__D_min",
                "D_samples": "__D_samples",
                "beta_samples": "__beta_samples",
                "": "",
            }
            if data_type not in suffixes:
                raise ValueError(f"Unknown data_type: {data_type}")
            filename = (
                f"D_vs_beta{suffixes[data_type]}__n={n}__q={q*100:2.0f}"
                f"__beta_min={np.min(beta_values)*100:.0f}"
                f"__beta_max={np.max(beta_values)*100:.0f}"
                f"__D_min={np.min(D_sample_sizes)}__D_max={np.max(D_sample_sizes)}"
                f"__c_success={c_success:1.0e}__c_theory={c_theory:1.0e}"
                f"__num_samples={num_samples}__T_max={T_max}"
                f"__corruption_type={corruption_type}.txt"
            )
        case "D_vs_T":
            suffixes = {"D_min": "__D_MIN", "": ""}
            if data_type not in suffixes:
                raise ValueError(f"Unknown data_type: {data_type}")
            filename = (
                f"D_vs_T{suffixes[data_type]}__n={n}__q={q*100:2.0f}"
                f"__beta={beta*100:.0f}__D_min={np.min(D_sample_sizes)}"
                f"__D_max={np.max(D_sample_sizes)}"
                f"__c_success={c_success:1.0e}__c_theory={c_theory:1.0e}"
                f"__num_samples={num_samples}__T_intervals={T_intervals}"
                f"__T_max={T_max}__corruption_type={corruption_type}.txt"
            )
        case _:
            raise ValueError(f"Unknown D_vs_TYPE: {D_vs_TYPE}")

    output_path = output_directory / filename
    matrix = np.atleast_2d(np.asarray(mean_success))
    np.savetxt(output_path, matrix, fmt="%.8f")
    if data_type == "":
        print(f"Filename: ./{output_path}\n")
