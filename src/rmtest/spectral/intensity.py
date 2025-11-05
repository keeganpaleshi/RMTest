"""Intensity functions for unbinned extended likelihood fitting.

This module constructs the total rate density λ(E) [counts/MeV] for spectral
fits. The intensity is properly normalized such that its integral over the
fit domain equals the total expected event count.

For extended unbinned likelihood:
    NLL = ∫ λ(E) dE - Σᵢ ln λ(Eᵢ) + ln(N!)

where λ(E) = Σₖ Nₖ fₖ(E) + background(E), with each fₖ normalized over the
fit window [E_min, E_max] (not over all energy).

IMPORTANT: Component PDFs must be normalized over the truncated fit window
to prevent amplitude bleeding between components and background.
"""

import numpy as np
from .shapes import emg_pdf_E, gaussian_pdf_E, emg_cdf_E, gaussian_cdf_E

__all__ = [
    "build_spectral_intensity",
    "spectral_intensity_E",
    "integral_of_intensity",
]


def _component_p_in_window(kind, E_lo, E_hi, mu, sigma, tau=None):
    """Compute the in-window probability mass for a component.

    Parameters
    ----------
    kind : str
        Either "emg" or "gauss".
    E_lo, E_hi : float
        Energy window bounds in MeV.
    mu : float
        Peak mean in MeV.
    sigma : float
        Peak resolution in MeV.
    tau : float, optional
        EMG tail parameter in MeV (required if kind="emg").

    Returns
    -------
    float
        P(E_lo < E < E_hi) for this component, in range (0, 1].

    Notes
    -----
    This uses the stable CDF implementations to compute:
        p_window = CDF(E_hi) - CDF(E_lo)

    Guard against pathological roundoff by ensuring result >= 1e-12.
    """
    if kind == "emg":
        c_lo = emg_cdf_E(E_lo, mu, sigma, tau)
        c_hi = emg_cdf_E(E_hi, mu, sigma, tau)
    else:
        c_lo = gaussian_cdf_E(E_lo, mu, sigma)
        c_hi = gaussian_cdf_E(E_hi, mu, sigma)

    p = float(c_hi - c_lo)
    # Guard against pathological roundoff
    return max(p, 1e-12)


def build_spectral_intensity(iso_list, use_emg, domain):
    """Factory to create a spectral intensity function.

    Parameters
    ----------
    iso_list : list of str
        Isotope names (e.g., ["Po210", "Po218", "Po214"]).
    use_emg : dict
        Mapping {isotope: bool} indicating whether to use EMG (True) or
        Gaussian (False) for each isotope.
    domain : tuple of float
        (E_min, E_max) energy bounds in MeV.

    Returns
    -------
    callable
        Function with signature intensity(E, params) returning λ(E) in counts/MeV.

    Notes
    -----
    Each component PDF is renormalized to the fit window [E_min, E_max] by dividing
    by its in-window probability mass p_window = CDF(E_max) - CDF(E_min).

    This ensures λ(E) = Σ N_k * (f_k(E) / p_k) + bkg(E), where:
        ∫[E_min, E_max] N_k * (f_k(E) / p_k) dE = N_k exactly

    The extended likelihood Poisson mean μ is then simply Σ N_k + ∫ bkg dE.
    """
    E_lo, E_hi = domain

    def intensity(E, params):
        """Compute total rate density λ(E) in counts/MeV.

        Parameters
        ----------
        E : array-like
            Energy values in MeV.
        params : dict
            Must contain:
                - N_{iso} : yield for each isotope (counts)
                - mu_{iso} : peak position for each isotope (MeV)
                - sigma0 : baseline resolution (MeV)
                - F : Fano factor for energy-dependent resolution
                - tau_{iso} : tail parameter for EMG isotopes (MeV)
                - b0, b1 : background slope/offset (counts/MeV)

        Returns
        -------
        array-like
            Rate density λ(E) with units counts/MeV.
        """
        E = np.asarray(E, dtype=float)
        lam = np.zeros_like(E, dtype=float)

        # Extract resolution parameters
        sigma0 = params.get("sigma0", 0.134)
        F = params.get("F", 0.0)

        # Add each isotope peak, renormalized to the window
        for iso in iso_list:
            N = params.get(f"N_{iso}", 0.0)
            if N <= 0:
                continue

            mu = params[f"mu_{iso}"]
            kind = "emg" if use_emg.get(iso, False) else "gauss"
            tau = params.get(f"tau_{iso}", None) if kind == "emg" else None

            # Compute PDF at data points (with energy-dependent resolution if F > 0)
            sigma_E = np.sqrt(sigma0**2 + F * E)
            if kind == "emg":
                dens = emg_pdf_E(E, mu, sigma_E, tau)
            else:
                dens = gaussian_pdf_E(E, mu, sigma_E)

            # Compute in-window probability mass using stable CDF
            # For energy-dependent sigma, use effective sigma ≈ sigma0 for CDF
            # (exact only if F=0; for typical radon fits F is small)
            pwin = _component_p_in_window(kind, E_lo, E_hi, mu, sigma0, tau)

            # Renormalize to window: λ += N * (dens / pwin)
            # This ensures ∫[E_lo, E_hi] λ_k dE = N
            lam += N * (dens / pwin)

        # Background handling: use S_bkg (flat) if present, else b0/b1 (linear)
        # IMPORTANT: When S_bkg is present, ignore b0/b1 to prevent double counting
        # and optimizer competition (b0/b1 tilt can steal counts from peaks)
        if "S_bkg" in params:
            # Flat background with total counts S_bkg over window
            S_bkg = params.get("S_bkg", 0.0)
            if S_bkg > 0:
                window_length = max(1e-12, E_hi - E_lo)
                flat_density = S_bkg / window_length  # counts/MeV
                lam += flat_density
        else:
            # Linear background density (counts/MeV)
            b0 = params.get("b0", 0.0)
            b1 = params.get("b1", 0.0)
            lam += b0 + b1 * E

        return lam

    return intensity


def spectral_intensity_E(E, params, domain, iso_list=None, use_emg=None):
    """Direct evaluation of spectral intensity for standard Po isotopes.

    This is a convenience function for the common case of Po-210, Po-218, Po-214
    fits. For custom isotope lists, use build_spectral_intensity().

    Parameters
    ----------
    E : array-like
        Energy values in MeV.
    params : dict or array-like
        If dict: same as build_spectral_intensity.
        If array-like: unpacked as [N210, mu210, sig0, tau210, N218, mu218,
                                    tau218, N214, mu214, tau214, b0, b1].
    domain : tuple of float
        (E_min, E_max) energy bounds in MeV.
    iso_list : list of str, optional
        Isotope names. Default: ["Po210", "Po218", "Po214"].
    use_emg : dict, optional
        EMG flags per isotope. Default: all True.

    Returns
    -------
    array-like
        Rate density λ(E) in counts/MeV.
    """
    E = np.asarray(E, dtype=float)

    if iso_list is None:
        iso_list = ["Po210", "Po218", "Po214"]
    if use_emg is None:
        use_emg = {iso: True for iso in iso_list}

    # Handle array-style params (for compatibility with existing code)
    if not isinstance(params, dict):
        # Standard 12-parameter format for Po210/218/214 with EMG
        if len(params) == 12:
            N210, mu210, sig0, tau210, N218, mu218, tau218, N214, mu214, tau214, b0, b1 = params
            params = {
                "N_Po210": N210,
                "mu_Po210": mu210,
                "sigma0": sig0,
                "F": 0.0,
                "tau_Po210": tau210,
                "N_Po218": N218,
                "mu_Po218": mu218,
                "tau_Po218": tau218,
                "N_Po214": N214,
                "mu_Po214": mu214,
                "tau_Po214": tau214,
                "b0": b0,
                "b1": b1,
            }
        else:
            raise ValueError(
                f"Array params must have 12 elements for Po210/218/214; got {len(params)}"
            )

    intensity_fn = build_spectral_intensity(iso_list, use_emg, domain)
    return intensity_fn(E, params)


def integral_of_intensity(params, domain, iso_list=None):
    """Compute the total expected event count ∫ λ(E) dE.

    For extended likelihood, this is the Poisson mean parameter μ.

    Parameters
    ----------
    params : dict or array-like
        Same as spectral_intensity_E.
    domain : tuple of float
        (E_min, E_max) energy bounds in MeV.
    iso_list : list of str, optional
        Isotope names. Default: ["Po210", "Po218", "Po214"].

    Returns
    -------
    float
        Total expected counts over the fit window.
    """
    E_min, E_max = domain

    if iso_list is None:
        iso_list = ["Po210", "Po218", "Po214"]

    # Handle array-style params
    if not isinstance(params, dict):
        if len(params) == 12:
            N210, mu210, sig0, tau210, N218, mu218, tau218, N214, mu214, tau214, b0, b1 = params
            params = {
                "N_Po210": N210,
                "mu_Po210": mu210,
                "sigma0": sig0,
                "F": 0.0,
                "tau_Po210": tau210,
                "N_Po218": N218,
                "mu_Po218": mu218,
                "tau_Po218": tau218,
                "N_Po214": N214,
                "mu_Po214": mu214,
                "tau_Po214": tau214,
                "b0": b0,
                "b1": b1,
            }
        else:
            raise ValueError(
                f"Array params must have 12 elements; got {len(params)}"
            )

    # Sum of signal yields (PDFs integrate to 1 over window, so integral is just N_k)
    total = sum(params.get(f"N_{iso}", 0.0) for iso in iso_list)

    # Background integral: use S_bkg (flat) if present, else b0/b1 (linear)
    # This must match the logic in build_spectral_intensity
    if "S_bkg" in params:
        # Flat background: integral is simply S_bkg (counts over window)
        bkg_integral = max(0.0, params.get("S_bkg", 0.0))
    else:
        # Linear background: ∫(b0 + b1·E) dE from E_min to E_max
        b0 = params.get("b0", 0.0)
        b1 = params.get("b1", 0.0)
        dE = E_max - E_min
        bkg_integral = b0 * dE + 0.5 * b1 * (E_max ** 2 - E_min ** 2)

    total += bkg_integral

    return float(total)
