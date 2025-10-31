"""Numerical helpers for the Exponentially Modified Gaussian (EMG)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import exponnorm

from constants import safe_exp as _safe_exp
import emg_stable as _emg_module
from emg_stable import emg_left_stable


# Modes that should defer to the numerically stable implementation.
_STABLE_MODES = {"scipy_safe", "erfcx_exact"}


def _current_mode() -> str:
    """Return the currently configured EMG mode."""

    return getattr(_emg_module, "EMG_STABLE_MODE", "scipy_safe")


def gaussian(x: Iterable[float] | np.ndarray, mu: float, sigma: float) -> np.ndarray:
    """Standard Gaussian probability density function."""

    x_arr = np.asarray(x, dtype=float)
    expo = -0.5 * ((x_arr - mu) / sigma) ** 2
    return _safe_exp(expo) / (sigma * np.sqrt(2 * np.pi))


def _legacy_emg_left(
    x: Iterable[float] | np.ndarray, mu: float, sigma: float, tau: float
) -> np.ndarray:
    """Evaluate the legacy ``scipy.stats.exponnorm`` implementation."""

    x_arr = np.asarray(x, dtype=float)
    K = tau / sigma
    logpdf = exponnorm.logpdf(x_arr, K, loc=mu, scale=sigma)
    return _safe_exp(logpdf)


def emg_left(
    x: Iterable[float] | np.ndarray, mu: float, sigma: float, tau: float
) -> np.ndarray:
    """Exponentially modified Gaussian (left-skewed) PDF."""

    from calibration import get_emg_tau_min  # Local import to avoid circular deps.

    tau_min = float(get_emg_tau_min())
    if tau <= tau_min:
        return gaussian(x, mu, sigma)

    mode = _current_mode()

    if mode in _STABLE_MODES:
        # ``emg_left_stable`` reads the global ``EMG_STABLE_MODE`` from ``emg_stable``.
        return emg_left_stable(x, mu, sigma, tau, amplitude=1.0, use_log_scale=False)

    if mode == "legacy":
        return _legacy_emg_left(x, mu, sigma, tau)

    raise ValueError(f"Unknown EMG mode: {mode}")


__all__ = ["emg_left", "gaussian"]

