"""Compatibility exports for fixed and Gaussian noise calculations."""

from qrk_analysis.feasibility.check import (
    find_sigma_with_largest_error_increase_fast,
)
from qrk_analysis.noise.fixed import (
    error_increased_C_3,
    find_C_with_largest_error_increase_fast,
)
from qrk_analysis.noise.oblivious import error_increased_Gaussian_noise

__all__ = [
    "error_increased_C_3",
    "error_increased_Gaussian_noise",
    "find_C_with_largest_error_increase_fast",
    "find_sigma_with_largest_error_increase_fast",
]
