"""Public fitting API exposed via the :mod:`rmtest` namespace."""

from __future__ import annotations

import sys
import types

import fitting as _fitting

from . import emg_config as _emg_config

FitResult = _fitting.FitResult
_TAU_MIN = _fitting._TAU_MIN
fit_spectrum = _fitting.fit_spectrum
fit_time_series = _fitting.fit_time_series
load_emg_config = _emg_config.load_emg_config

_emg_config.set_default_mode(getattr(_fitting, "EMG_STABLE_MODE", True))

__all__ = [
    "FitResult",
    "_TAU_MIN",
    "fit_spectrum",
    "fit_time_series",
    "EMG_STABLE_MODE",
    "load_emg_config",
]


class _FittingProxyModule(types.ModuleType):
    """Delegate attribute access to the top-level :mod:`fitting` module."""

    def __getattr__(self, name):
        data = object.__getattribute__(self, "__dict__")
        if name in data:
            return data[name]
        if name == "EMG_STABLE_MODE" and not hasattr(_fitting, name):
            return _emg_config.get_emg_stable_mode()
        if hasattr(_fitting, name):
            return getattr(_fitting, name)
        raise AttributeError(name)

    def __setattr__(self, name, value):
        if name == "EMG_STABLE_MODE" and not hasattr(_fitting, name):
            _emg_config.set_mode_override(value)
            data = object.__getattribute__(self, "__dict__")
            data.pop(name, None)
            return
        if hasattr(_fitting, name):
            setattr(_fitting, name, value)
            if name in __all__:
                if name == "EMG_STABLE_MODE":
                    data = object.__getattribute__(self, "__dict__")
                    data.pop(name, None)
                    _emg_config.set_default_mode(getattr(_fitting, name))
                    _emg_config.set_mode_override(None)
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
