"""Fitting-related EMG/config helpers.

This module re-exports from rmtest.emg_constants so tests that
import from `rmtest.fitting` see the exact same values as
`rmtest.emg_constants`.

Important
---------
Do NOT import the top-level `fitting.py` from here. That file imports us,
so importing it here creates a circular import during test collection.
"""

from rmtest.emg_constants import (
    EMG_MIN_TAU,
    EMG_STABLE_MODE,
    EMG_METHOD,
    EMG_USE_EMG,
    emg_min_tau_from_config,
    emg_stable_mode_from_config,
    emg_method_from_config,
    emg_use_emg_from_config,
    clamp_tau,
)

# Do NOT assign EMG_STABLE_MODE = True here; just re-export
__all__ = [
    "EMG_MIN_TAU",
    "EMG_STABLE_MODE",
    "EMG_METHOD",
    "EMG_USE_EMG",
    "emg_min_tau_from_config",
    "emg_stable_mode_from_config",
    "emg_method_from_config",
    "emg_use_emg_from_config",
    "clamp_tau",
]


def get_FitResult():
    """Lazy import of FitResult to avoid circular imports.

    Returns
    -------
    type
        The FitResult class from the top-level fitting module.

    Notes
    -----
    This function imports fitting.FitResult lazily at call time rather than
    at module import time to avoid circular import issues.
    """
    from fitting import FitResult
    return FitResult
