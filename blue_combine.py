"""Compatibility layer for BLUE combination."""

from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np

from efficiency import blue_combine as _efficiency_blue_combine

# Provide a local alias so this module can re-export ``blue_combine``.
blue_combine = _efficiency_blue_combine

CovarianceMatrix = np.ndarray

@dataclass
class Measurements:
    values: Sequence[float]
    errors: Sequence[float]
    corr: Optional[np.ndarray] = None


def BLUE(measurements: Measurements, *, allow_negative: bool = True):
    """Return BLUE combination of the given measurements."""
    return blue_combine(
        measurements.values,
        measurements.errors,
        measurements.corr,
        allow_negative=allow_negative,
    )

__all__ = ["blue_combine", "BLUE", "Measurements", "CovarianceMatrix"]
