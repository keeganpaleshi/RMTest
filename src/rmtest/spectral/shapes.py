import math
from functools import lru_cache

import numpy as np
from scipy.optimize import brentq
from scipy.special import erfcx
from scipy.stats import exponnorm, norm


_SQRT_TWO = math.sqrt(2.0)
_SQRT_TWO_OVER_PI = math.sqrt(2.0 / math.pi)


@lru_cache(maxsize=2048)
def _right_emg_mode_offset(sigma: float, tau: float) -> float:
    """Return the standard right-EMG mode offset relative to ``loc``."""
    if sigma <= 0.0 or tau <= 0.0:
        return 0.0

    target = _SQRT_TWO_OVER_PI * (tau / sigma)
    if target <= 0.0:
        return 0.0

    lo = -1.0
    while float(erfcx(lo)) < target:
        lo *= 2.0
        if lo <= -512.0:
            break

    hi = 1.0
    while float(erfcx(hi)) > target:
        hi *= 2.0
        if hi >= 512.0:
            break

    def _root_fn(u: float) -> float:
        return float(erfcx(u) - target)

    u = brentq(_root_fn, lo, hi, xtol=1e-12, rtol=1e-12, maxiter=100)
    return sigma * sigma / tau - _SQRT_TWO * sigma * u


def emg_mode_to_loc(mu, sigma, tau):
    """Convert a left-EMG peak position into the underlying mirrored ``loc``."""
    mu_arr = np.asarray(mu, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    tau_val = float(tau)

    if tau_val <= 0.0 or np.any(sigma_arr <= 0.0):
        return mu_arr

    if sigma_arr.ndim == 0:
        return float(mu_arr) + _right_emg_mode_offset(float(sigma_arr), tau_val)

    tau_arr = np.full(sigma_arr.shape, tau_val, dtype=float)
    offsets = np.vectorize(_right_emg_mode_offset, otypes=[float])(sigma_arr, tau_arr)
    return mu_arr + offsets



def emg_loc_to_mode(loc, sigma, tau):
    """Convert a left-EMG mirrored ``loc`` parameter into its visible peak mode."""
    loc_arr = np.asarray(loc, dtype=float)
    sigma_arr = np.asarray(sigma, dtype=float)
    tau_val = float(tau)

    if tau_val <= 0.0 or np.any(sigma_arr <= 0.0):
        return loc_arr

    if sigma_arr.ndim == 0:
        return float(loc_arr) - _right_emg_mode_offset(float(sigma_arr), tau_val)

    tau_arr = np.full(sigma_arr.shape, tau_val, dtype=float)
    offsets = np.vectorize(_right_emg_mode_offset, otypes=[float])(sigma_arr, tau_arr)
    return loc_arr - offsets

def emg_pdf_E(E, mu, sigma, tau):
    """
    Left-skewed exponentially modified Gaussian PDF parameterized by peak mode.

    ``mu`` is the visible peak position in MeV, not the underlying SciPy
    ``exponnorm`` location parameter.
    """
    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0) or tau <= 0:
        return np.zeros_like(E, dtype=float)
    K = tau / sigma
    loc = emg_mode_to_loc(mu, sigma, tau)
    E_mirror = 2.0 * np.asarray(loc, dtype=float) - E
    return exponnorm.pdf(E_mirror, K, loc=loc, scale=sigma)


def emg_cdf_E(E, mu, sigma, tau):
    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0) or tau <= 0:
        return np.zeros_like(E, dtype=float)
    K = tau / sigma
    loc = emg_mode_to_loc(mu, sigma, tau)
    E_mirror = 2.0 * np.asarray(loc, dtype=float) - E
    return 1.0 - exponnorm.cdf(E_mirror, K, loc=loc, scale=sigma)


def gaussian_pdf_E(E, mu, sigma):
    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0):
        return np.zeros_like(E, dtype=float)
    return norm.pdf(E, loc=mu, scale=sigma)


def gaussian_cdf_E(E, mu, sigma):
    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0):
        return np.zeros_like(E, dtype=float)
    return norm.cdf(E, loc=mu, scale=sigma)
