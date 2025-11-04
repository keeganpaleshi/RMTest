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
from .shapes import emg_pdf_E, gaussian_pdf_E


def _normalize_pdf_over_window(raw_pdf_func, E_min, E_max, n_grid=2048):
    """Normalize a PDF over a truncated energy window.

    Parameters
    ----------
    raw_pdf_func : callable
        Function that returns PDF values: raw_pdf_func(E) -> array
        This PDF is assumed to be normalized over all energy (−∞, +∞).
    E_min, E_max : float
        Energy window bounds in MeV.
    n_grid : int, optional
        Number of grid points for numerical integration. Default: 2048.

    Returns
    -------
    normalized_pdf_func : callable or None
        Window-normalized PDF function, or None if normalization failed.
    window_area : float
        Integral of raw_pdf over [E_min, E_max]. This is the "truncation factor".

    Notes
    -----
    For a PDF f(E) normalized over all energy:
        ∫_{-∞}^{+∞} f(E) dE = 1

    Over a truncated window [E_min, E_max]:
        ∫_{E_min}^{E_max} f(E) dE = α < 1

    The window-normalized PDF is:
        f_window(E) = f(E) / α

    so that:
        ∫_{E_min}^{E_max} f_window(E) dE = 1

    This ensures that when we compute:
        λ(E) = N × f_window(E)

    we get:
        ∫_{E_min}^{E_max} λ(E) dE = N

    i.e., the amplitude N equals the expected counts in the window.
    """
    # Compute integral of raw PDF over window
    grid = np.linspace(E_min, E_max, n_grid)
    raw_vals = raw_pdf_func(grid)

    # Safety checks
    if not np.isfinite(raw_vals).all() or np.any(raw_vals < 0):
        return None, 0.0

    window_area = np.trapz(raw_vals, grid)

    if not np.isfinite(window_area) or window_area <= 0:
        return None, 0.0

    # Create normalized PDF
    def normalized_pdf(E):
        return raw_pdf_func(E) / window_area

    return normalized_pdf, window_area

__all__ = [
    "build_spectral_intensity",
    "spectral_intensity_E",
    "integral_of_intensity",
]


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
    Each component PDF is normalized over the truncated fit window [E_min, E_max]
    to ensure that amplitude parameters represent actual expected counts in the window.
    This prevents amplitude bleeding between components and background.
    """
    E_min, E_max = domain

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

        # Add each isotope peak with window normalization
        for iso in iso_list:
            N = params.get(f"N_{iso}", 0.0)
            if N <= 0:
                continue

            mu = params[f"mu_{iso}"]

            # Create raw PDF function for this component
            # Energy-dependent resolution: σ(E) = √(σ₀² + F·E)
            # Use default arguments to capture values (avoid closure issues)
            if use_emg.get(iso, False):
                tau = params[f"tau_{iso}"]

                def raw_pdf(E_grid, mu=mu, sigma0=sigma0, F=F, tau=tau):
                    sigma_E_grid = np.sqrt(sigma0 ** 2 + F * E_grid)
                    return emg_pdf_E(E_grid, mu, sigma_E_grid, tau)
            else:
                def raw_pdf(E_grid, mu=mu, sigma0=sigma0, F=F):
                    sigma_E_grid = np.sqrt(sigma0 ** 2 + F * E_grid)
                    return gaussian_pdf_E(E_grid, mu, sigma_E_grid)

            # Normalize over window [E_min, E_max]
            pdf_norm, window_area = _normalize_pdf_over_window(raw_pdf, E_min, E_max)

            if pdf_norm is not None and window_area > 0:
                # Evaluate window-normalized PDF at data points
                sigma_E = np.sqrt(sigma0 ** 2 + F * E)
                if use_emg.get(iso, False):
                    tau = params[f"tau_{iso}"]
                    pdf_vals = emg_pdf_E(E, mu, sigma_E, tau) / window_area
                else:
                    pdf_vals = gaussian_pdf_E(E, mu, sigma_E) / window_area

                lam += N * pdf_vals

        # Add linear background (already a density in counts/MeV)
        b0 = params.get("b0", 0.0)
        b1 = params.get("b1", 0.0)
        lam += b0 + b1 * E

        return np.clip(lam, 1e-300, np.inf)

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

    # Sum of signal yields (PDFs integrate to 1, so integral is just N_k)
    total = sum(params.get(f"N_{iso}", 0.0) for iso in iso_list)

    # Background integral: ∫(b0 + b1·E) dE from E_min to E_max
    b0 = params.get("b0", 0.0)
    b1 = params.get("b1", 0.0)
    dE = E_max - E_min
    bkg_integral = b0 * dE + 0.5 * b1 * (E_max ** 2 - E_min ** 2)

    total += bkg_integral

    return float(total)
