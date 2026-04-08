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


def right_emg_pdf_E(E, mu, sigma, tau):
    """Right-skewed exponentially modified Gaussian PDF.

    Standard (non-mirrored) EMG: the exponential tail extends to HIGHER
    energies.  Models slight right-side broadening from electronic effects
    or detector response asymmetry.

    ``mu`` is the visible peak mode, converted internally to the underlying
    ``exponnorm`` location parameter so the peak stays at *mu*.
    """
    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0) or tau <= 0:
        return np.zeros_like(E, dtype=float)
    K = tau / sigma
    # Standard right-EMG: mode is shifted LEFT of loc by the mode offset.
    # We want peak at mu, so loc = mu + offset (shift loc right so mode
    # lands at mu).
    offset = _right_emg_mode_offset(float(sigma.flat[0]) if sigma.ndim else float(sigma), float(tau))
    loc = np.asarray(mu, dtype=float) + offset
    return exponnorm.pdf(E, K, loc=loc, scale=sigma)


def right_emg_cdf_E(E, mu, sigma, tau):
    """CDF of the right-skewed EMG (standard exponnorm)."""
    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    if np.any(sigma <= 0) or tau <= 0:
        return np.zeros_like(E, dtype=float)
    K = tau / sigma
    offset = _right_emg_mode_offset(float(sigma.flat[0]) if sigma.ndim else float(sigma), float(tau))
    loc = np.asarray(mu, dtype=float) + offset
    return exponnorm.cdf(E, K, loc=loc, scale=sigma)


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


# ── Skew-normal (tilted Gaussian) ────────────────────────────────────────

def skewnorm_pdf_E(E, mu, sigma, alpha):
    """Skew-normal PDF: f(x) = 2/σ × φ((x-μ)/σ) × Φ(α(x-μ)/σ).

    Parameters
    ----------
    alpha : float
        Skewness parameter.  α>0 → right-skewed, α<0 → left-skewed, α=0 → Gaussian.
        For PIN diode alpha spectroscopy, expect α<0 (left skew from charge loss).
    """
    E = np.asarray(E, dtype=float)
    if sigma <= 0:
        return np.zeros_like(E, dtype=float)
    z = (E - mu) / sigma
    return (2.0 / sigma) * norm.pdf(z) * norm.cdf(alpha * z)


def skewnorm_cdf_E(E, mu, sigma, alpha):
    """CDF of the skew-normal distribution (numerical integration)."""
    from scipy.stats import skewnorm as _skewnorm
    E = np.asarray(E, dtype=float)
    if sigma <= 0:
        return np.zeros_like(E, dtype=float)
    return _skewnorm.cdf(E, alpha, loc=mu, scale=sigma)


# ── Split (asymmetric) Gaussian ──────────────────────────────────────────

def split_gaussian_pdf_E(E, mu, sigma_left, sigma_right):
    """Asymmetric Gaussian PDF with different widths on each side of mu.

    Continuous at E=mu.  Normalized so the total integral is 1.
    """
    E = np.asarray(E, dtype=float)
    sl = float(sigma_left)
    sr = float(sigma_right)
    if sl <= 0 or sr <= 0:
        return np.zeros_like(E, dtype=float)
    # Normalization: integral = sqrt(2*pi)/2 * (sigma_L + sigma_R)
    norm_const = np.sqrt(2.0 * np.pi) * 0.5 * (sl + sr)
    result = np.where(
        E <= mu,
        np.exp(-0.5 * ((E - mu) / sl) ** 2),
        np.exp(-0.5 * ((E - mu) / sr) ** 2),
    )
    return result / norm_const


def split_gaussian_cdf_E(E, mu, sigma_left, sigma_right):
    """CDF of the split (asymmetric) Gaussian."""
    E = np.asarray(E, dtype=float)
    sl = float(sigma_left)
    sr = float(sigma_right)
    if sl <= 0 or sr <= 0:
        return np.zeros_like(E, dtype=float)
    w_l = sl / (sl + sr)  # weight of left half
    w_r = sr / (sl + sr)  # weight of right half
    # Left of mu: use left-sigma Gaussian CDF, scaled to [0, w_l]
    # Right of mu: use right-sigma Gaussian CDF, scaled from [w_l, 1]
    cdf_left = w_l * 2.0 * norm.cdf(E, loc=mu, scale=sl)   # 2× because half-Gaussian
    cdf_right = w_l + w_r * 2.0 * (norm.cdf(E, loc=mu, scale=sr) - 0.5)
    return np.where(E <= mu, cdf_left, cdf_right)


# ── Low-energy shelf (erfc step) for alpha spectroscopy ─────────────────

def shelf_pdf_E(E, mu, sigma, E_lo, E_hi, shelf_range=None, shelf_cutoff_delta=None):
    """Low-energy shelf PDF for degraded alpha particles.

    Models the flat continuum of alphas that lose variable amounts of energy
    in the detector dead layer or source material.  Degraded alphas can only
    LOSE energy, so the shelf is strictly zero above the cutoff point.

    The cutoff is at ``mu - Δ`` where Δ (shelf_cutoff_delta) is a shared
    absolute energy offset in MeV below the peak centroid.

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
    shelf_cutoff_delta : float, optional
        Absolute energy offset (MeV) below the peak at which the shelf
        cuts off.  Default ``0.12`` MeV (~1 sigma).

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

    if shelf_cutoff_delta is None:
        shelf_cutoff_delta = 0.12
    _cutoff_E = mu - float(shelf_cutoff_delta)

    from scipy.special import erfc

    shape = erfc((E - mu) / (sigma * _SQRT_TWO))

    # Hard cutoff below the peak: shelf is zero above mu - cutoff_offset.
    # This prevents shelf from creating a notch at the peak centroid.
    shape = np.where(E <= _cutoff_E, shape, 0.0)

    # Gaussian taper below the peak to limit shelf extent
    _below = np.minimum(E - mu, 0.0)
    shape = shape * np.exp(-0.5 * (_below / shelf_range) ** 2)

    # Normalize over [E_lo, E_hi] via numerical integration
    n_pts = 2048
    grid = np.linspace(E_lo, E_hi, n_pts)
    shape_grid = erfc((grid - mu) / (sigma * _SQRT_TWO))
    shape_grid = np.where(grid <= _cutoff_E, shape_grid, 0.0)
    _below_g = np.minimum(grid - mu, 0.0)
    shape_grid = shape_grid * np.exp(-0.5 * (_below_g / shelf_range) ** 2)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    Z = _trapz(shape_grid, grid)
    Z = max(Z, 1e-300)

    return shape / Z


def shelf_cdf_E(E, mu, sigma, E_lo, E_hi, shelf_range=None, shelf_cutoff_delta=None):
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

    if shelf_cutoff_delta is None:
        shelf_cutoff_delta = 0.12
    _cutoff_E = mu - float(shelf_cutoff_delta)

    from scipy.special import erfc

    def _shelf_shape(grid):
        s = erfc((grid - mu) / (sigma * _SQRT_TWO))
        # Hard cutoff below the peak (matches shelf_pdf_E)
        s = np.where(grid <= _cutoff_E, s, 0.0)
        # Left-side taper
        _below = np.minimum(grid - mu, 0.0)
        return s * np.exp(-0.5 * (_below / shelf_range) ** 2)

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
