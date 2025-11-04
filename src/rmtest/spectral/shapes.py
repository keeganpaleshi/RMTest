"""Unit-normalized shape functions for spectral fitting.

This module provides properly normalized PDF shapes in energy space (MeV^-1)
for use in unbinned extended likelihood fitting. All shapes integrate to 1
over their support.
"""

import numpy as np
from scipy.special import erfc

__all__ = ["emg_pdf_E", "gaussian_pdf_E"]


def emg_pdf_E(E, mu, sigma, tau):
    """Unit-normalized exponentially modified Gaussian in energy (MeV^-1).

    Returns a proper probability density function that integrates to 1 over
    the entire energy domain. This is the correct form for unbinned likelihood
    fitting, where the intensity is λ(E) = N × emg_pdf_E(E).

    Parameters
    ----------
    E : array-like
        Energy values in MeV.
    mu : float
        Gaussian mean (peak center) in MeV.
    sigma : float
        Gaussian standard deviation (resolution) in MeV.
    tau : float
        Exponential decay constant (tail parameter) in MeV.

    Returns
    -------
    array-like
        EMG probability density values with units of MeV^-1. Integrates to 1.

    Notes
    -----
    The EMG is defined as the convolution of a Gaussian with an exponential
    decay. The PDF is:

        f(E) = 1/(2τ) exp(σ²/(2τ²) - (E-μ)/τ) erfc((σ/τ - (E-μ)/σ)/√2)

    For numerical stability, falls back to zero density when sigma or tau are
    non-positive.

    References
    ----------
    Kalambet et al. (2011) "Reconstruction of chromatographic peaks using the
    exponentially modified Gaussian function"
    """
    E = np.asarray(E)

    # Handle invalid parameters
    if sigma <= 0 or tau <= 0:
        return np.zeros_like(E, dtype=float)

    # Normalized variables
    z = (E - mu) / sigma
    t = sigma / tau

    # EMG formula: (1/2τ) exp(t²/2 - z/t) erfc((t - z)/√2)
    pref = 1.0 / (2.0 * tau)

    # Compute exponent with care for numerical stability
    exponent = 0.5 * (t ** 2) - z / t

    # Compute argument to erfc
    erfc_arg = (t - z) / np.sqrt(2.0)

    # Combine terms
    with np.errstate(over="ignore", under="ignore", invalid="ignore"):
        result = pref * np.exp(exponent) * erfc(erfc_arg)

    # Clean up any NaN or Inf values
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


def gaussian_pdf_E(E, mu, sigma):
    """Unit-normalized Gaussian in energy (MeV^-1).

    Parameters
    ----------
    E : array-like
        Energy values in MeV.
    mu : float
        Gaussian mean in MeV.
    sigma : float
        Gaussian standard deviation in MeV.

    Returns
    -------
    array-like
        Gaussian probability density values with units of MeV^-1. Integrates to 1.
    """
    E = np.asarray(E)

    if sigma <= 0:
        return np.zeros_like(E, dtype=float)

    z = (E - mu) / sigma
    norm = sigma * np.sqrt(2.0 * np.pi)

    with np.errstate(over="ignore", under="ignore"):
        result = np.exp(-0.5 * z ** 2) / norm

    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result
