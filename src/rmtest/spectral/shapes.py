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

    # Check bracket validity before calling brentq
    f_lo = _root_fn(lo)
    f_hi = _root_fn(hi)
    if f_lo * f_hi > 0:
        # Bracket doesn't straddle the root — use asymptotic approximation.
        # For large tau/sigma (tau >> sigma), mode ≈ mu (offset ≈ 0).
        # For small tau/sigma (tau << sigma), mode ≈ mu - sigma²/tau + sigma
        # (the Gaussian dominates, mode near the Gaussian peak).
        ratio = tau / sigma
        if ratio > 10.0:
            return 0.0  # EMG is essentially Gaussian
        else:
            # Approximate: mode offset ≈ sigma * (1 - sigma/tau) for moderate tau
            return max(sigma * (1.0 - sigma / tau), 0.0)

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


# ── Low-energy shelf (erfc step) for alpha spectroscopy ─────────────────

def shelf_pdf_E(E, mu, sigma, E_lo, E_hi, shelf_range=None):
    """Low-energy shelf PDF for degraded alpha particles.

    Models the flat continuum of alphas that lose variable amounts of energy
    in the source material or detector dead layer.  The shape is an erfc step
    that transitions at the peak energy ``mu``, multiplied by a one-sided
    Gaussian taper that limits how far below the peak the shelf extends:

        shape(E) ∝ erfc((E − μ) / (σ√2)) × taper(E)

    where taper(E) = 1 for E ≥ μ, and for E < μ:

        taper(E) = exp(−(E − μ)² / (2 × shelf_range²))

    This prevents the shelf from extending unrealistically far below the peak.

    Parameters
    ----------
    E : array-like
        Energy values in MeV.
    mu : float
        Peak centroid (transition midpoint).
    sigma : float
        Width of the step transition (typically the detector resolution).
    E_lo, E_hi : float
        Fit window boundaries for normalization.
    shelf_range : float, optional
        1-sigma range (MeV) of the one-sided Gaussian taper below the peak.
        Controls how far the shelf extends.  Default ``1.0`` MeV.

    Returns
    -------
    ndarray
        Normalized shelf PDF values at each *E*.
    """
    E = np.asarray(E, dtype=float)
    sigma = float(sigma)
    if sigma <= 0:
        return np.zeros_like(E, dtype=float)

    if shelf_range is None:
        shelf_range = 1.0
    shelf_range = max(float(shelf_range), 0.01)

    from scipy.special import erfc

    shape = erfc((E - mu) / (sigma * _SQRT_TWO))

    # One-sided Gaussian taper: only attenuate below the peak
    taper = np.where(
        E >= mu,
        1.0,
        np.exp(-0.5 * ((E - mu) / shelf_range) ** 2),
    )
    shape = shape * taper

    # Normalize over [E_lo, E_hi] via numerical integration
    n_pts = 2048
    grid = np.linspace(E_lo, E_hi, n_pts)
    shape_grid = erfc((grid - mu) / (sigma * _SQRT_TWO))
    taper_grid = np.where(
        grid >= mu,
        1.0,
        np.exp(-0.5 * ((grid - mu) / shelf_range) ** 2),
    )
    shape_grid = shape_grid * taper_grid
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    Z = _trapz(shape_grid, grid)
    Z = max(Z, 1e-300)

    return shape / Z


def shelf_cdf_E(E, mu, sigma, E_lo, E_hi, shelf_range=None):
    """CDF of the shelf component (integral from E_lo to E).

    Used by :func:`normalize_pdf_to_window` for window-normalized shelf
    components.
    """
    E = np.asarray(E, dtype=float)
    sigma = float(sigma)
    if sigma <= 0:
        return np.zeros_like(E, dtype=float)

    if shelf_range is None:
        shelf_range = 1.0
    shelf_range = max(float(shelf_range), 0.01)

    from scipy.special import erfc

    def _shelf_shape(grid):
        s = erfc((grid - mu) / (sigma * _SQRT_TWO))
        taper = np.where(
            grid >= mu,
            1.0,
            np.exp(-0.5 * ((grid - mu) / shelf_range) ** 2),
        )
        return s * taper

    n_pts = 2048
    # Compute the full integral for normalization
    grid_full = np.linspace(E_lo, E_hi, n_pts)
    shape_full = _shelf_shape(grid_full)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    Z = _trapz(shape_full, grid_full)
    Z = max(Z, 1e-300)

    # Compute cumulative integral from E_lo to each E value
    result = np.empty_like(E, dtype=float)
    for i, e_val in enumerate(np.atleast_1d(E)):
        if e_val <= E_lo:
            result.flat[i] = 0.0
        elif e_val >= E_hi:
            result.flat[i] = 1.0
        else:
            grid_e = np.linspace(E_lo, float(e_val), n_pts)
            shape_e = _shelf_shape(grid_e)
            result.flat[i] = _trapz(shape_e, grid_e) / Z

    return result
