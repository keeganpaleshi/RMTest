"""Public EMG helpers re-exported through :mod:`rmtest`."""

from __future__ import annotations

import sys
from types import ModuleType

import emg_stable as _emg_module

from .emg_math import emg_left_dispatch, gaussian

_MODE_ALIASES = {
    "scipy": "scipy_safe",
    "scipy_safe": "scipy_safe",
    "stable": "scipy_safe",
    "auto": "scipy_safe",
    "erfcx": "erfcx_exact",
    "erfcx_exact": "erfcx_exact",
    "legacy": "legacy",
    "disabled": "legacy",
    "off": "legacy",
}


def _canonicalize_mode(mode: object) -> str:
    """Return the canonical string representation for ``mode``."""

    if isinstance(mode, bytes):
        mode = mode.decode("ascii", "ignore")
    text = str(mode).strip().lower()
    try:
        return _MODE_ALIASES[text]
    except KeyError as exc:
        raise ValueError(f"Unknown EMG stable mode: {mode!r}") from exc


def _mode_prefers_legacy(mode: str) -> bool:
    return mode == "legacy"


def _mode_prefers_log_scale(mode: str) -> bool:
    return mode == "erfcx_exact"


EMG_STABLE_MODE = _canonicalize_mode(getattr(_emg_module, "EMG_STABLE_MODE", "scipy_safe"))
setattr(_emg_module, "EMG_STABLE_MODE", EMG_STABLE_MODE)


def get_emg_stable_mode() -> str:
    """Return the currently configured EMG evaluation strategy."""

    return EMG_STABLE_MODE


def set_emg_stable_mode(mode: object) -> None:
    """Update :data:`EMG_STABLE_MODE` and keep :mod:`emg_stable` in sync."""

    global EMG_STABLE_MODE
    EMG_STABLE_MODE = _canonicalize_mode(mode)
    setattr(_emg_module, "EMG_STABLE_MODE", EMG_STABLE_MODE)


def emg_left(x, mu: float, sigma: float, tau: float):
    """Evaluate the EMG PDF using the configured strategy."""

    mode = EMG_STABLE_MODE
    prefer_legacy = _mode_prefers_legacy(mode)
    use_log_scale = _mode_prefers_log_scale(mode)
    return emg_left_dispatch(
        x,
        mu,
        sigma,
        tau,
        use_log_scale=use_log_scale,
        prefer_legacy=prefer_legacy,
    )


__all__ = [
    "EMG_STABLE_MODE",
    "emg_left",
    "gaussian",
    "get_emg_stable_mode",
    "set_emg_stable_mode",
]


class _EmgModule(ModuleType):
    """Module proxy that routes attribute assignments through the setter."""

    def __setattr__(self, name, value):  # noqa: D401 - short delegating docstring
        if name == "EMG_STABLE_MODE":
            set_emg_stable_mode(value)
        else:
            super().__setattr__(name, value)


sys.modules[__name__].__class__ = _EmgModule
