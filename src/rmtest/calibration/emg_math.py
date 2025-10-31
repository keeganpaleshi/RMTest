"""Exponentially modified Gaussian helpers used by :mod:`rmtest` calibration."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import exponnorm

import emg_stable as _stable_module
from calibration import gaussian as _legacy_gaussian
from constants import _TAU_MIN as _DEFAULT_TAU_MIN, safe_exp as _safe_exp
from emg_stable import emg_left_stable

# ---------------------------------------------------------------------------
# Legacy compatibility shims
# ---------------------------------------------------------------------------
_DEFAULT_MODE = "scipy_safe"
_LEGACY_MODES: frozenset[str] = frozenset({"legacy", "scipy", "exponnorm"})


def _normalise_mode(value: str) -> str:
    """Return a canonical representation for ``value``.

    ``legacy`` and ``scipy`` are treated as synonyms for the original
    :mod:`scipy.stats.exponnorm` implementation, while any other value delegates
    to the numerically stable helper provided by :mod:`emg_stable`.
    """

    normalised = value.strip().lower()
    if normalised in _LEGACY_MODES:
        return "legacy"
    return normalised or _DEFAULT_MODE


def _sync_mode(value: str) -> str:
    mode = _normalise_mode(value)
    setattr(_stable_module, "EMG_STABLE_MODE", mode)
    return mode


EMG_STABLE_MODE: str = _sync_mode(getattr(_stable_module, "EMG_STABLE_MODE", _DEFAULT_MODE))


def get_emg_stable_mode() -> str:
    """Return the active EMG implementation mode."""

    return EMG_STABLE_MODE


def set_emg_stable_mode(value: str) -> None:
    """Update the EMG implementation mode, keeping :mod:`emg_stable` in sync."""

    global EMG_STABLE_MODE
    EMG_STABLE_MODE = _sync_mode(value)


def _tau_minimum() -> float:
    return float(getattr(_stable_module, "_EMG_TAU_MIN", _DEFAULT_TAU_MIN))


def gaussian(x: Iterable[float] | np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Lightweight Gaussian wrapper that mirrors :func:`calibration.gaussian`."""

    return _legacy_gaussian(x, mu, sigma)


def emg_left(
    x: Iterable[float] | np.ndarray,
    mu: float,
    sigma: float,
    tau: float,
) -> np.ndarray:
    """Evaluate the left-skewed EMG PDF using the configured strategy."""

    tau_min = _tau_minimum()
    if tau <= tau_min:
        return gaussian(x, mu, sigma)

    mode = get_emg_stable_mode()
    if mode == "legacy":
        K = tau / sigma
        log_pdf = exponnorm.logpdf(x, K, loc=mu, scale=sigma)
        return _safe_exp(log_pdf)

    if mode not in getattr(_stable_module, "_EMG_STRATEGIES", {}):
        raise ValueError(f"Unknown EMG stable mode: {mode}")

    return emg_left_stable(x, mu, sigma, tau, amplitude=1.0, use_log_scale=False)


__all__ = [
    "EMG_STABLE_MODE",
    "emg_left",
    "gaussian",
    "get_emg_stable_mode",
    "set_emg_stable_mode",
]
