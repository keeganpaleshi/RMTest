"""Public fitting API exposed via the :mod:`rmtest` namespace."""

from __future__ import annotations

import sys
import types

import fitting as _fitting

from .emg_utils import EMGTailSpec, resolve_emg_usage

FitResult = _fitting.FitResult
_TAU_MIN = _fitting._TAU_MIN
fit_spectrum = _fitting.fit_spectrum
fit_time_series = _fitting.fit_time_series
get_emg_stable_mode = _fitting.get_emg_stable_mode

__all__ = [
    "FitResult",
    "_TAU_MIN",
    "fit_spectrum",
    "fit_time_series",
    "EMG_STABLE_MODE",
    "get_emg_stable_mode",
    "EMGTailSpec",
    "resolve_emg_usage",
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
