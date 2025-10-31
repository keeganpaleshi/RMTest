"""Public EMG helpers for :mod:`rmtest.calibration`."""

from __future__ import annotations

import calibration as _legacy_calibration
import emg_stable as _emg_module

from .emg_math import emg_left, gaussian


EMG_STABLE_MODE: str = getattr(_emg_module, "EMG_STABLE_MODE", "scipy_safe")
setattr(_emg_module, "EMG_STABLE_MODE", EMG_STABLE_MODE)


def get_emg_stable_mode() -> str:
    """Return the current EMG mode used by :func:`emg_left`."""

    mode = getattr(_emg_module, "EMG_STABLE_MODE", EMG_STABLE_MODE)
    globals()["EMG_STABLE_MODE"] = mode
    return mode


def set_emg_stable_mode(mode: str) -> None:
    """Update the EMG evaluation mode across the legacy and stable helpers."""

    normalized = str(mode)
    setattr(_emg_module, "EMG_STABLE_MODE", normalized)
    globals()["EMG_STABLE_MODE"] = normalized


get_use_stable_emg = _legacy_calibration.get_use_stable_emg
set_use_stable_emg = _legacy_calibration.set_use_stable_emg
get_emg_tau_min = _legacy_calibration.get_emg_tau_min
set_emg_tau_min = _legacy_calibration.set_emg_tau_min
configure_emg = _legacy_calibration.configure_emg


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

