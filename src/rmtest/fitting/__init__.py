"""Public fitting API for the :mod:`rmtest` package."""

from __future__ import annotations

from fitting import (
    FitParams,
    FitResult,
    fit_decay,
    fit_spectrum,
    fit_time_series,
    make_linear_bkg,
    softplus,
    _TAU_MIN,
)

__all__ = [
    "FitParams",
    "FitResult",
    "fit_decay",
    "fit_spectrum",
    "fit_time_series",
    "make_linear_bkg",
    "softplus",
    "_TAU_MIN",
]

