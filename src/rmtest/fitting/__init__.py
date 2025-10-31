"""Public fitting API exposed through the :mod:`rmtest` namespace."""

from __future__ import annotations

from fitting import FitResult, fit_decay, fit_spectrum, fit_time_series, _TAU_MIN

from rmtest.calibration.emg import (
    EMG_STABLE_MODE,
    emg_left,
    gaussian,
    get_emg_stable_mode,
    set_emg_stable_mode,
)

__all__ = [
    "FitResult",
    "_TAU_MIN",
    "EMG_STABLE_MODE",
    "emg_left",
    "fit_decay",
    "fit_spectrum",
    "fit_time_series",
    "gaussian",
    "get_emg_stable_mode",
    "set_emg_stable_mode",
]
