"""Compatibility exports for canonical feasibility checks."""

from qrk_analysis.feasibility.check import (
    check_feasibility,
    check_feasibility_conditions_C_sup_revised,
    check_feasibility_conditions_random_sup_revised,
)

__all__ = [
    "check_feasibility",
    "check_feasibility_conditions_C_sup_revised",
    "check_feasibility_conditions_random_sup_revised",
]
