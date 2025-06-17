"""Compatibility layer for BLUE combination."""

from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np

from efficiency import blue_combine as _blue_combine

CovarianceMatrix = np.ndarray

@dataclass
class Measurements:
    values: Sequence[float]
    errors: Sequence[float]
    corr: Optional[np.ndarray] = None


def BLUE(measurements: Measurements):
    """Return BLUE combination of the given measurements."""
    return _blue_combine(measurements.values, measurements.errors, measurements.corr)

# Expose efficiency.blue_combine under the same name for convenience
blue_combine = _blue_combine

__all__ = ["BLUE", "Measurements", "CovarianceMatrix", "blue_combine"]
