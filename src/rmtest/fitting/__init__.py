"""
Thin re-export of EMG config for tests.
Do NOT import the top-level fitting.py here to avoid circular imports.
"""

from rmtest.emg_constants import (
    EMG_MIN_TAU,
    EMG_STABLE_MODE,
    EMG_DEFAULT_METHOD,
    emg_min_tau_from_config,
    emg_stable_mode_from_config,
    emg_method_from_config,
    emg_use_emg_from_config,
    clamp_tau,
)

__all__ = [
    "EMG_MIN_TAU",
    "EMG_STABLE_MODE",
    "EMG_DEFAULT_METHOD",
    "emg_min_tau_from_config",
    "emg_stable_mode_from_config",
    "emg_method_from_config",
    "emg_use_emg_from_config",
    "clamp_tau",
]
