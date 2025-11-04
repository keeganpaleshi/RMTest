"""Negative log-likelihood for extended unbinned spectral fits.

This module implements the correct extended unbinned likelihood for spectral
fitting. The NLL is:

    NLL = μ - Σᵢ ln λ(Eᵢ) + ln(N!)

where:
    - μ = ∫ λ(E) dE is the total expected event count (Poisson parameter)
    - λ(Eᵢ) is the rate density at each observed energy
    - N is the observed event count
    - ln(N!) is the Poisson constant term (keeps NLL positive and finite)

The intensity λ(E) must be a proper density in counts/MeV, with each peak
component normalized as N_k × f_k(E) where ∫ f_k dE = 1 over the fit window.
"""

import numpy as np
from scipy.special import gammaln
from .intensity import spectral_intensity_E, integral_of_intensity

__all__ = ["nll_extended_unbinned", "nll_extended_unbinned_simple"]


def nll_extended_unbinned(E, params, domain, iso_list=None, use_emg=None):
    """Extended unbinned negative log-likelihood for spectral fitting.

    Parameters
    ----------
    E : array-like
        Observed energy values in MeV (unbinned data).
    params : dict or array-like
        Model parameters. See spectral_intensity_E for format.
    domain : tuple of float
        (E_min, E_max) fit window in MeV.
    iso_list : list of str, optional
        Isotope names. Default: ["Po210", "Po218", "Po214"].
    use_emg : dict, optional
        EMG flags per isotope. Default: all True.

    Returns
    -------
    float
        Negative log-likelihood value. Lower is better.

    Notes
    -----
    The extended likelihood accounts for both the probability of observing
    each event AND the total event count via Poisson statistics:

        L_ext = Poisson(N_obs | μ) × Πᵢ p(Eᵢ)
              = exp(-μ) μ^N / N! × Πᵢ [λ(Eᵢ) / μ]
              ∝ exp(-μ) × Πᵢ λ(Eᵢ)

    Taking -ln gives:

        NLL = μ - Σᵢ ln λ(Eᵢ)

    This penalizes both poor fit quality and incorrect total normalization.
    """
    E = np.asarray(E, dtype=float)

    if E.size == 0:
        return np.inf

    # Evaluate intensity at each data point
    lam = spectral_intensity_E(E, params, domain, iso_list=iso_list, use_emg=use_emg)

    # Check for invalid intensities
    if np.any(lam <= 0) or not np.isfinite(lam).all():
        return np.inf

    # Compute total expected count
    mu_tot = integral_of_intensity(params, domain, iso_list=iso_list)

    if not np.isfinite(mu_tot) or mu_tot <= 0:
        return np.inf

    # Extended NLL: μ - Σ ln λ(Eᵢ) + ln(N!)
    # The Poisson constant ln(N!) keeps NLL positive and finite
    n = E.size
    log_sum = np.sum(np.log(lam))
    nll = mu_tot - log_sum + gammaln(n + 1)

    return float(nll)


def nll_extended_unbinned_simple(E, intensity_fn, params, domain):
    """Extended unbinned NLL with custom intensity function.

    This is a lower-level interface that accepts a pre-built intensity function.
    Use this for custom models beyond the standard Po isotope fits.

    Parameters
    ----------
    E : array-like
        Observed energies in MeV.
    intensity_fn : callable
        Function with signature intensity_fn(E, params, domain) returning λ(E).
    params : dict
        Model parameters passed to intensity_fn.
    domain : tuple of float
        (E_min, E_max) fit window.

    Returns
    -------
    float
        Negative log-likelihood.
    """
    E = np.asarray(E, dtype=float)

    if E.size == 0:
        return np.inf

    # Evaluate intensity
    lam = intensity_fn(E, params, domain)

    if np.any(lam <= 0) or not np.isfinite(lam).all():
        return np.inf

    # For custom intensity, caller must provide integral
    # We expect params to include "mu_total" for extended term
    mu_tot = params.get("mu_total", np.sum(lam))

    if not np.isfinite(mu_tot) or mu_tot <= 0:
        return np.inf

    nll = mu_tot - np.sum(np.log(lam))

    return float(nll)
