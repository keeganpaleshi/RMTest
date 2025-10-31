"""Calibration helpers exposed via the public :mod:`rmtest` package."""

from .emg import (
    EMG_STABLE_MODE,
    get_emg_stable_mode,
    set_emg_stable_mode,
    emg_left,
    gaussian,
    get_emg_tau_min,
    set_emg_tau_min,
    get_use_stable_emg,
    set_use_stable_emg,
    configure_emg,
)

__all__ = [
    "EMG_STABLE_MODE",
    "configure_emg",
    "emg_left",
    "gaussian",
    "get_emg_stable_mode",
    "get_emg_tau_min",
    "get_use_stable_emg",
    "set_emg_stable_mode",
    "set_emg_tau_min",
    "set_use_stable_emg",
]

