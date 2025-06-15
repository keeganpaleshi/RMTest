"""Compatibility layer for BLUE combination."""

from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

from efficiency import blue_combine

CovarianceMatrix = np.ndarray


@dataclass
class Measurements:
    values: Sequence[float]
    errors: Sequence[float]
    corr: Optional[np.ndarray] = None


def BLUE(measurements: Measurements):
    """Return BLUE combination of the given measurements."""
    return blue_combine(measurements.values, measurements.errors, measurements.corr)


__all__ = ["BLUE", "Measurements", "CovarianceMatrix"]
