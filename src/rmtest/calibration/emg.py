"""Compatibility layer for EMG calibration helpers in :mod:`rmtest`."""

from __future__ import annotations

import sys
import types

import calibration as _calibration

from . import emg_math as _emg_math

configure_emg = _calibration.configure_emg
emg_left = _emg_math.emg_left
gaussian = _calibration.gaussian
get_emg_tau_min = _calibration.get_emg_tau_min
get_use_stable_emg = _calibration.get_use_stable_emg
set_emg_stable_mode = _calibration.set_emg_stable_mode
set_emg_tau_min = _calibration.set_emg_tau_min
set_use_stable_emg = _calibration.set_use_stable_emg

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


class _EmgProxyModule(types.ModuleType):
    """Module proxy that keeps compatibility globals synced with :mod:`calibration`."""

    def __getattr__(self, name):
        if name == "EMG_STABLE_MODE":
            return getattr(_calibration, name)
        if hasattr(_calibration, name):
            return getattr(_calibration, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == "EMG_STABLE_MODE":
            _calibration.set_emg_stable_mode(value)
            return
        if hasattr(_calibration, name):
            setattr(_calibration, name, value)
            return
        super().__setattr__(name, value)

    def __dir__(self):
        combined = set(super().__dir__())
        combined.update(__all__)
        combined.update(dir(_calibration))
        return sorted(combined)


module = sys.modules[__name__]
module.__class__ = _EmgProxyModule
