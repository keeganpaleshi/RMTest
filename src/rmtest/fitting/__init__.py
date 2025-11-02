"""Public fitting API exposed via the :mod:`rmtest` namespace."""

from __future__ import annotations

import sys
import types

import fitting as _fitting

from .emg_config import (
    DEFAULT_EMG_STABLE_MODE,
    emg_stable_mode_from_config,
    resolve_emg_stable_mode,
    synchronize_emg_stable_mode,
)

FitResult = _fitting.FitResult
_TAU_MIN = _fitting._TAU_MIN
fit_spectrum = _fitting.fit_spectrum
fit_time_series = _fitting.fit_time_series

__all__ = [
    "FitResult",
    "_TAU_MIN",
    "fit_spectrum",
    "fit_time_series",
    "DEFAULT_EMG_STABLE_MODE",
    "emg_stable_mode_from_config",
    "resolve_emg_stable_mode",
    "synchronize_emg_stable_mode",
    "EMG_STABLE_MODE",
]


class _FittingProxyModule(types.ModuleType):
    """Delegate attribute access to the top-level :mod:`fitting` module."""

    def __getattr__(self, name):
        data = object.__getattribute__(self, "__dict__")
        if name in data:
            return data[name]
        if hasattr(_fitting, name):
            return getattr(_fitting, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if hasattr(_fitting, name):
            setattr(_fitting, name, value)
            if name in __all__:
                if name == "EMG_STABLE_MODE":
                    data = object.__getattribute__(self, "__dict__")
                    data.pop(name, None)
                else:
                    super().__setattr__(name, getattr(_fitting, name))
            return
        super().__setattr__(name, value)

    def __dir__(self):
        combined = set(super().__dir__())
        combined.update(__all__)
        combined.update(dir(_fitting))
        return sorted(combined)


module = sys.modules[__name__]
module.__class__ = _FittingProxyModule
