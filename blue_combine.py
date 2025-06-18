"""Compatibility layer for BLUE combination."""

from dataclasses import dataclass
from typing import Sequence, Optional
import numpy as np

from efficiency import blue_combine as _efficiency_blue_combine


def blue_combine(
    values: Sequence[float],
    errors: Sequence[float],
    corr: Optional[np.ndarray] = None,
    *,
    allow_negative: bool = False,
):
    """Passthrough to :func:`efficiency.blue_combine` with kwarg support."""
    return _efficiency_blue_combine(
        values,
        errors,
        corr,
        allow_negative=allow_negative,
    )

CovarianceMatrix = np.ndarray

@dataclass
class Measurements:
    values: Sequence[float]
    errors: Sequence[float]
    corr: Optional[np.ndarray] = None


def BLUE(measurements: Measurements, *, allow_negative: bool = False):
    """Return BLUE combination of the given measurements."""
    return blue_combine(
        measurements.values,
        measurements.errors,
        measurements.corr,
        allow_negative=allow_negative,
    )

__all__ = ["blue_combine", "BLUE", "Measurements", "CovarianceMatrix"]
