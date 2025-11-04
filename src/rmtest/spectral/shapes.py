"""Unit-normalized shape functions for spectral fitting.

This module provides properly normalized PDF shapes in energy space (MeV^-1)
for use in unbinned extended likelihood fitting. All shapes integrate to 1
over their support.

IMPORTANT: These functions support both scalar and vector sigma (for energy-
dependent resolution σ(E) = √(σ₀² + F·E)). All parameter checks are array-safe.

CRITICAL: For truncated fits over [E_min, E_max], use the CDFs to compute
the in-window mass Z_k = CDF(E_max) - CDF(E_min), then normalize the PDF
by dividing by Z_k. This ensures λ_k(E) = (N_k / Z_k) × f_k(E) integrates
to exactly N_k over the fit window.
"""

import numpy as np
from scipy.special import erfc, erf

__all__ = ["emg_pdf_E", "gaussian_pdf_E", "gaussian_cdf_E", "emg_cdf_E"]


def _phi(z):
    """Standard normal CDF: Φ(z) = (1 + erf(z/√2)) / 2."""
    return 0.5 * (1.0 + erf(z / np.sqrt(2.0)))


def gaussian_pdf_E(E, mu, sigma):
    """Unit-normalized Gaussian in energy (MeV^-1).

    Integrates to 1 over the entire energy domain. Handles both scalar and
    vector sigma (energy-dependent resolution).

    Parameters
    ----------
    E : array-like
        Energy values in MeV.
    mu : float
        Gaussian mean in MeV.
    sigma : float or array-like
        Gaussian standard deviation in MeV. Can be scalar or vector for
        energy-dependent resolution σ(E).

    Returns
    -------
    array-like
        Gaussian probability density values with units of MeV^-1. Integrates to 1.

    Notes
    -----
    For energy-dependent resolution, pass sigma as an array:
        sigma_E = np.sqrt(sigma0**2 + F * E)
        pdf = gaussian_pdf_E(E, mu, sigma_E)
    """
    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)

    # Vector-safe validation: return zeros if any sigma is invalid
    if np.any(~np.isfinite(sigma)) or np.any(sigma <= 0):
        return np.zeros_like(E, dtype=float)

    x = (E - mu) / sigma
    out = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * x * x)

    # Clean up any lingering non-finite values
    out = np.where(np.isfinite(out), out, 0.0)

    return out


def gaussian_cdf_E(E, mu, sigma):
    """Cumulative distribution function for Gaussian.

    Returns the probability P(X ≤ E) for X ~ N(mu, sigma²).

    Parameters
    ----------
    E : array-like or float
        Energy values in MeV.
    mu : float
        Gaussian mean in MeV.
    sigma : float
        Gaussian standard deviation in MeV.

    Returns
    -------
    array-like or float
        CDF values in [0, 1].

    Notes
    -----
    For truncated fits, use Z = CDF(E_max) - CDF(E_min) to get the in-window
    mass, then normalize: pdf_window(E) = pdf(E) / Z.
    """
    E = np.asarray(E, dtype=float)
    sigma = max(float(sigma), 1e-12)  # Guard against zero

    z = (E - mu) / sigma
    return _phi(z)


def emg_pdf_E(E, mu, sigma, tau):
    """Unit-normalized exponentially modified Gaussian in energy (MeV^-1).

    Correct, normalized PDF for unbinned likelihood:
        f(E) = (1/(2*tau)) * exp( sigma^2/(2*tau^2) - (E-mu)/tau ) *
               erfc( (sigma/tau - (E-mu)/sigma)/sqrt(2) )

    Integrates to 1 over the entire energy domain. Handles both scalar and
    vector sigma (energy-dependent resolution).

    Parameters
    ----------
    E : array-like
        Energy values in MeV.
    mu : float
        Gaussian mean (peak center) in MeV.
    sigma : float or array-like
        Gaussian standard deviation (resolution) in MeV. Can be scalar or
        vector for energy-dependent resolution σ(E).
    tau : float
        Exponential decay constant (tail parameter) in MeV.

    Returns
    -------
    array-like
        EMG probability density values with units of MeV^-1. Integrates to 1.

    Notes
    -----
    - Uses np.erfc (plain complementary error function), NOT erfcx.
    - If you need the scaled error function erfcx for numerical stability,
      you must compensate with exp(-z^2) exactly once to avoid blow-up.
    - The formula is the standard normalized EMG from convolution of
      Gaussian(mu, sigma) with Exponential(tau).

    For energy-dependent resolution:
        sigma_E = np.sqrt(sigma0**2 + F * E)
        pdf = emg_pdf_E(E, mu, sigma_E, tau)

    References
    ----------
    Kalambet et al. (2011) "Reconstruction of chromatographic peaks using the
    exponentially modified Gaussian function"
    """
    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    tau = np.asarray(tau, dtype=float)

    # Vector-safe validation: return zeros if any parameter is invalid
    if np.any(~np.isfinite(sigma)) or np.any(~np.isfinite(tau)) \
       or np.any(sigma <= 0) or np.any(tau <= 0):
        return np.zeros_like(E, dtype=float)

    inv_tau = 1.0 / tau
    E_mu = E - mu

    # Argument to erfc: z = (sigma/tau - (E-mu)/sigma) / sqrt(2)
    # Use broadcasting-friendly operations
    z = (sigma * inv_tau - E_mu / sigma) / np.sqrt(2.0)

    # Standard normalized EMG formula with plain erfc
    # f(E) = (1/(2*tau)) * exp(sigma^2/(2*tau^2) - (E-mu)/tau) * erfc(z)
    out = 0.5 * inv_tau * np.exp(0.5 * (sigma * inv_tau) ** 2 - E_mu * inv_tau) * erfc(z)

    # Clean up any non-finite values from numerical issues
    out = np.where(np.isfinite(out), out, 0.0)

    return out


def emg_cdf_E(E, mu, sigma, tau):
    """Cumulative distribution function for exponentially modified Gaussian.

    Returns the probability P(X ≤ E) for X ~ EMG(mu, sigma, tau).

    Parameters
    ----------
    E : array-like or float
        Energy values in MeV.
    mu : float
        Gaussian mean (peak center) in MeV.
    sigma : float
        Gaussian standard deviation in MeV.
    tau : float
        Exponential decay constant (tail parameter) in MeV.

    Returns
    -------
    array-like or float
        CDF values in [0, 1].

    Notes
    -----
    The EMG CDF is given by:
        F(E) = Φ(a) - exp(-b + c²/2) × Φ(a - c)

    where:
        a = (E - mu) / sigma
        c = sigma / tau
        b = (E - mu) / tau

    For truncated fits, use Z = CDF(E_max) - CDF(E_min) to get the in-window
    mass, then normalize: pdf_window(E) = pdf(E) / Z.

    References
    ----------
    Grushka (1972) "Characterization of exponentially modified Gaussian peaks"
    """
    E = np.asarray(E, dtype=float)
    sigma = max(float(sigma), 1e-12)  # Guard against zero
    tau = max(float(tau), 1e-12)

    a = (E - mu) / sigma
    c = sigma / tau
    b = (E - mu) / tau

    # F(E) = Φ(a) - exp(-b + c²/2) × Φ(a - c)
    cdf = _phi(a) - np.exp(-b + 0.5 * c * c) * _phi(a - c)

    # Clip to [0, 1] for numerical safety
    return np.clip(cdf, 0.0, 1.0)
