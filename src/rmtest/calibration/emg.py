"""Compatibility layer for EMG helpers exposed by :mod:`rmtest`."""

from __future__ import annotations

import sys
import types

from . import emg_math as _emg_math

emg_left = _emg_math.emg_left
"""Evaluate the left-skewed EMG PDF with the active implementation."""

gaussian = _emg_math.gaussian
"""Gaussian PDF helper mirroring :func:`calibration.gaussian`."""

get_emg_stable_mode = _emg_math.get_emg_stable_mode
set_emg_stable_mode = _emg_math.set_emg_stable_mode


class _EmgModule(types.ModuleType):
    """Module proxy that keeps ``EMG_STABLE_MODE`` in sync with :mod:`emg_stable`."""

    @property
    def EMG_STABLE_MODE(self) -> str:  # pragma: no cover - trivial delegation
        return _emg_math.get_emg_stable_mode()

    @EMG_STABLE_MODE.setter
    def EMG_STABLE_MODE(self, value: str) -> None:  # pragma: no cover - delegation
        _emg_math.set_emg_stable_mode(value)


# Rebind the current module instance to the proxy subclass so attribute access goes
# through the descriptors defined above. Existing attributes remain untouched.
sys.modules[__name__].__class__ = _EmgModule

__all__ = [
    "EMG_STABLE_MODE",
    "emg_left",
    "gaussian",
    "get_emg_stable_mode",
    "set_emg_stable_mode",
]
