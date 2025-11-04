"""Intensity functions for extended unbinned likelihood fitting.

This module constructs the total rate density λ(E) [counts/MeV] for the spectral
model, ensuring proper normalization for extended unbinned maximum likelihood fits.
"""

import numpy as np
from .shapes import emg_pdf_E, gaussian_pdf_E


def spectral_intensity_E(E, params, domain, use_emg=None):
    """Build the spectral rate density λ(E) in counts per MeV.

    For an extended unbinned fit, the intensity must satisfy:
        ∫ λ(E) dE = total expected counts

    This function constructs:
        λ(E) = Σ_k N_k f_k(E | θ_k) + b0 + b1*E

    where each f_k is a unit-normalized PDF (∫ f_k dE = 1) and N_k is the
    expected count for peak k.

    Parameters
    ----------
    E : array-like
        Energy values in MeV.
    params : dict or array-like
        If dict: parameter mapping including N210, mu210, sig0, tau210, ...
        If array: unpacked in the order given by the params list below.
    domain : tuple
        (Emin, Emax) defining the fit range in MeV.
    use_emg : dict, optional
        Mapping {iso: bool} indicating whether to use EMG (True) or Gaussian (False)
        for each isotope. Defaults to EMG for all if not specified.

    Expected parameters
    -------------------
    When params is a dict:
        N210, mu210, sig0, tau210,
        N218, mu218, tau218,
        N214, mu214, tau214,
        b0, b1

    When params is an array, it must follow this order. The function can handle
    variable resolution (sigma can depend on energy), but for simplicity we use
    sig0 as a constant resolution parameter.

    Returns
    -------
    array-like
        Rate density λ(E) in counts/MeV.

    Notes
    -----
    - The peak shapes f_k integrate to 1, so ∫ N_k f_k dE = N_k.
    - The background b0 + b1*E is already a density [counts/MeV], so its
      integral is b0*ΔE + b1/2*(E_max^2 - E_min^2).
    - Total integral: Σ N_k + b0*ΔE + b1/2*(E_max^2 - E_min^2).
    """
    Emin, Emax = domain
    E = np.asarray(E, dtype=float)

    # Parse parameters
    if isinstance(params, dict):
        N210 = params.get("N210", params.get("S_Po210", 0.0))
        mu210 = params.get("mu210", params.get("mu_Po210", 5.3))
        sig0 = params.get("sig0", params.get("sigma0", 0.1))
        tau210 = params.get("tau210", params.get("tau_Po210", 0.02))

        N218 = params.get("N218", params.get("S_Po218", 0.0))
        mu218 = params.get("mu218", params.get("mu_Po218", 6.0))
        tau218 = params.get("tau218", params.get("tau_Po218", 0.03))

        N214 = params.get("N214", params.get("S_Po214", 0.0))
        mu214 = params.get("mu214", params.get("mu_Po214", 7.687))
        tau214 = params.get("tau214", params.get("tau_Po214", 0.02))

        b0 = params.get("b0", 0.0)
        b1 = params.get("b1", 0.0)
    else:
        # Assume array-like in fixed order
        (
            N210,
            mu210,
            sig0,
            tau210,
            N218,
            mu218,
            tau218,
            N214,
            mu214,
            tau214,
            b0,
            b1,
        ) = params[:12]

    # Default to EMG for all isotopes if not specified
    if use_emg is None:
        use_emg = {"Po210": True, "Po218": True, "Po214": True}

    # Build intensity as sum of peaks + background
    lam = np.zeros_like(E, dtype=float)

    # Ensure yields are non-negative (safety check for direct calls)
    N210 = max(0.0, float(N210))
    N218 = max(0.0, float(N218))
    N214 = max(0.0, float(N214))

    # Po-210 peak
    if N210 > 0:
        if use_emg.get("Po210", True) and tau210 > 0:
            f210 = emg_pdf_E(E, mu210, sig0, tau210)
        else:
            f210 = gaussian_pdf_E(E, mu210, sig0)
        lam += N210 * f210

    # Po-218 peak
    if N218 > 0:
        if use_emg.get("Po218", True) and tau218 > 0:
            f218 = emg_pdf_E(E, mu218, sig0, tau218)
        else:
            f218 = gaussian_pdf_E(E, mu218, sig0)
        lam += N218 * f218

    # Po-214 peak
    if N214 > 0:
        if use_emg.get("Po214", True) and tau214 > 0:
            f214 = emg_pdf_E(E, mu214, sig0, tau214)
        else:
            f214 = gaussian_pdf_E(E, mu214, sig0)
        lam += N214 * f214

    # Linear background (already a density in counts/MeV)
    lam += b0 + b1 * E

    return lam


def integral_of_intensity(params, domain, use_emg=None):
    """Compute the total expected counts ∫ λ(E) dE over the fit domain.

    This is the Poisson mean μ_tot for the extended unbinned likelihood.

    Parameters
    ----------
    params : dict or array-like
        Same format as for :func:`spectral_intensity_E`.
    domain : tuple
        (Emin, Emax) in MeV.
    use_emg : dict, optional
        Isotope-to-bool mapping for EMG usage.

    Returns
    -------
    float
        Total expected counts in the fit window.

    Notes
    -----
    For unit-normalized peak shapes, ∫ N_k f_k dE = N_k.
    For the background: ∫ (b0 + b1*E) dE = b0*ΔE + b1/2*(E_max^2 - E_min^2).
    """
    Emin, Emax = domain

    if isinstance(params, dict):
        N210 = params.get("N210", params.get("S_Po210", 0.0))
        N218 = params.get("N218", params.get("S_Po218", 0.0))
        N214 = params.get("N214", params.get("S_Po214", 0.0))
        b0 = params.get("b0", 0.0)
        b1 = params.get("b1", 0.0)
    else:
        N210, _, _, _, N218, _, _, N214, _, _, b0, b1 = params[:12]

    dE = Emax - Emin
    peak_integral = N210 + N218 + N214
    background_integral = b0 * dE + 0.5 * b1 * (Emax**2 - Emin**2)

    return float(peak_integral + background_integral)
