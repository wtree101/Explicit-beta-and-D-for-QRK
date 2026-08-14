"""Compatibility exports for canonical slack-parameter searches."""

from qrk_analysis.feasibility.search import (
    find_alpha_pair,
    find_max_c_without_failure_constraint,
)

__all__ = ["find_alpha_pair", "find_max_c_without_failure_constraint"]
