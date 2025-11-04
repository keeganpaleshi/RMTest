"""Unit-normalized spectral shapes for unbinned fitting.

This module provides properly normalized probability density functions (PDFs)
for use in extended unbinned likelihood fits, where each shape integrates to
unity over the full energy domain.
"""

import numpy as np
from scipy.special import erfc


def emg_pdf_E(E, mu, sigma, tau):
    """Unit-normalized exponentially modified Gaussian in energy space.

    Returns a probability density in units of MeV^-1, properly normalized so
    that the integral over all E equals 1.

    Parameters
    ----------
    E : array-like
        Energy values in MeV.
    mu : float
        Gaussian mean (peak centroid) in MeV.
    sigma : float
        Gaussian standard deviation (resolution) in MeV.
    tau : float
        Exponential decay constant (tail parameter) in MeV.

    Returns
    -------
    array-like
        Probability density values [MeV^-1].

    Notes
    -----
    The EMG is defined as the convolution of a Gaussian with an exponential decay.
    This implementation uses the analytic form:

        f(E) = (1/(2*tau)) * exp((sigma^2)/(2*tau^2) - (E-mu)/tau)
               * erfc((sigma/tau - (E-mu)/sigma) / sqrt(2))

    For numerical stability when tau or sigma approach zero, returns zeros.
    """
    E = np.asarray(E, dtype=float)

    if sigma <= 0 or tau <= 0:
        return np.zeros_like(E)

    z = (E - mu) / sigma
    t = sigma / tau

    # f(E) = 1/(2*tau) * exp(t^2/2 - z/t) * erfc((t - z)/sqrt(2))
    pref = 1.0 / (2.0 * tau)

    # Compute exponent carefully to avoid overflow
    exponent = 0.5 * (t**2) - z / t
    expo = np.exp(np.clip(exponent, -700, 700))

    arg = (t - z) / np.sqrt(2.0)
    erfc_term = erfc(arg)

    result = pref * expo * erfc_term

    # Clean up any NaN/Inf values
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result


def gaussian_pdf_E(E, mu, sigma):
    """Unit-normalized Gaussian in energy space.

    Parameters
    ----------
    E : array-like
        Energy values in MeV.
    mu : float
        Mean (peak centroid) in MeV.
    sigma : float
        Standard deviation (resolution) in MeV.

    Returns
    -------
    array-like
        Probability density values [MeV^-1].
    """
    E = np.asarray(E, dtype=float)

    if sigma <= 0:
        return np.zeros_like(E)

    # Standard Gaussian PDF: (1/sqrt(2*pi*sigma^2)) * exp(-(E-mu)^2/(2*sigma^2))
    normalization = 1.0 / (sigma * np.sqrt(2.0 * np.pi))
    exponent = -0.5 * ((E - mu) / sigma) ** 2

    result = normalization * np.exp(np.clip(exponent, -700, 700))
    result = np.nan_to_num(result, nan=0.0, posinf=0.0, neginf=0.0)

    return result
