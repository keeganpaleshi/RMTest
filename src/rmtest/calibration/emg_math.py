"""Low-level helpers for the Exponentially Modified Gaussian (EMG)."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy.stats import exponnorm

from constants import _TAU_MIN as _DEFAULT_TAU_MIN, _safe_exp
import emg_stable as _emg_module
from emg_stable import emg_left_stable

ArrayLike = Iterable[float] | np.ndarray


def _get_tau_min() -> float:
    """Return the current minimum ``tau`` permitted for EMG evaluations."""

    tau_min = getattr(_emg_module, "_EMG_TAU_MIN", None)
    if tau_min is None:
        return float(_DEFAULT_TAU_MIN)
    return float(tau_min)


def gaussian(x: ArrayLike, mu: float, sigma: float) -> np.ndarray:
    """Return a unit-area Gaussian PDF."""

    x = np.asarray(x, dtype=float)
    norm = sigma * np.sqrt(2.0 * np.pi)
    expo = -0.5 * ((x - mu) / sigma) ** 2
    return _safe_exp(expo) / norm


def legacy_emg_left(x: ArrayLike, mu: float, sigma: float, tau: float) -> np.ndarray:
    """Evaluate the legacy scipy-based EMG PDF."""

    x = np.asarray(x, dtype=float)
    K = tau / sigma
    logpdf = exponnorm.logpdf(x, K, loc=mu, scale=sigma)
    return _safe_exp(logpdf)


def stable_emg_left(
    x: ArrayLike,
    mu: float,
    sigma: float,
    tau: float,
    *,
    use_log_scale: bool = False,
) -> np.ndarray:
    """Evaluate the numerically stable EMG PDF provided by :mod:`emg_stable`."""

    return emg_left_stable(x, mu, sigma, tau, amplitude=1.0, use_log_scale=use_log_scale)


def emg_left_dispatch(
    x: ArrayLike,
    mu: float,
    sigma: float,
    tau: float,
    *,
    use_log_scale: bool = False,
    prefer_legacy: bool = False,
) -> np.ndarray:
    """Dispatch between the stable and legacy EMG implementations."""

    tau_min = _get_tau_min()
    if tau <= tau_min:
        return gaussian(x, mu, sigma)

    if prefer_legacy:
        return legacy_emg_left(x, mu, sigma, tau)

    try:
        return stable_emg_left(x, mu, sigma, tau, use_log_scale=use_log_scale)
    except ValueError:
        # Unknown strategy requested; fall back to the legacy behaviour to match
        # the historical API contract.
        return legacy_emg_left(x, mu, sigma, tau)


__all__ = [
    "gaussian",
    "legacy_emg_left",
    "stable_emg_left",
    "emg_left_dispatch",
]
