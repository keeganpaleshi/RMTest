"""Calibration helpers exposed via the :mod:`rmtest` namespace."""

from .emg import (
    configure_emg,
    emg_left,
    gaussian,
    get_emg_tau_min,
    get_use_stable_emg,
    set_emg_stable_mode,
    set_emg_tau_min,
    set_use_stable_emg,
)

__all__ = [
    "configure_emg",
    "emg_left",
    "gaussian",
    "get_emg_tau_min",
    "get_use_stable_emg",
    "set_emg_stable_mode",
    "set_emg_tau_min",
    "set_use_stable_emg",
    "EMG_STABLE_MODE",
]


def __getattr__(name):
    if name == "EMG_STABLE_MODE":
        from . import emg as _emg

        return getattr(_emg, name)
    raise AttributeError(name)


def __dir__():
    return sorted(__all__)
