"""Canonical numerical analysis for streaming QRK bounds."""

from .debug import set_debug
from .upper_bound import smallest_D, smallest_continuous_D

__all__ = ["set_debug", "smallest_D", "smallest_continuous_D"]
