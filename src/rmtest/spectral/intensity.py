# src/rmtest/spectral/intensity.py
"""
Spectral intensity functions for window-normalized extended likelihood fitting.

Parameter naming convention:
- N_* / S_* : In-window counts (post window-renormalization), not densities
- After window normalization, N_{iso} and S_{iso} both represent the total
  expected counts for that isotope within the fit window [E_lo, E_hi]
"""
import numpy as np
from .window_norm import normalize_pdf_to_window

def build_spectral_intensity(iso_list, use_emg, domain):
    """
    Build spectral intensity function λ(E) in counts/MeV for extended likelihood.

    Each peak is normalized to the fit window exactly once via CDF-based probability
    mass calculation. The returned intensity function produces a density (counts/MeV)
    that when integrated over the window equals the sum of peak yields plus background.

    Parameters
    ----------
    iso_list : sequence of str
        Isotope names (e.g., ["Po210", "Po218", "Po214"])
    use_emg : bool
        If True, use EMG tails where tau parameters are provided
    domain : tuple of (E_lo, E_hi)
        Energy window bounds in MeV

    Returns
    -------
    spectral_intensity_E : callable
        Function signature: spectral_intensity_E(E, params, domain, **kwargs)
        Returns λ(E) in counts/MeV
    """
    E_lo, E_hi = domain
    kind = "emg" if use_emg else "gauss"

    def spectral_intensity_E(E, params, domain, **kwargs):
        E = np.asarray(E, dtype=float)
        lam = np.zeros_like(E, dtype=float)

        # Peaks
        for iso in iso_list:
            N = float(params.get(f"N_{iso}", params.get(f"S_{iso}", 0.0)))
            if N <= 0.0:
                continue
            mu = float(params[f"mu_{iso}"])
            sigma = float(params["sigma0"])

            # Check if this isotope has a valid tau parameter (for EMG)
            tau = float(params.get(f"tau_{iso}", 0.0))
            use_emg_for_iso = use_emg and tau > 1e-9

            if use_emg_for_iso:
                pdf_win, _ = normalize_pdf_to_window("emg", mu, sigma, E_lo, E_hi, tau=tau)
            else:
                pdf_win, _ = normalize_pdf_to_window("gauss", mu, sigma, E_lo, E_hi)
            lam += N * pdf_win(E)

        # Background density: b0 + b1*E (counts/MeV). If S_bkg is present, it is total counts in window.
        b0 = float(params.get("b0", 0.0))
        b1 = float(params.get("b1", 0.0))
        lam += (b0 + b1 * E)

        # If S_bkg is explicitly given, add it as a flat density over the window.
        if "S_bkg" in params:
            S_bkg = float(params["S_bkg"])
            width = max(E_hi - E_lo, 1e-12)
            lam += (S_bkg / width)

        return lam

    return spectral_intensity_E

def integral_of_intensity(params, domain, iso_list=None, use_emg=False, **kwargs):
    """
    In-window expected counts μ = sum(N_iso) + background counts in [E_lo,E_hi].
    """
    E_lo, E_hi = domain
    if iso_list is None:
        iso_list = ("Po210", "Po218", "Po214")

    mu_signal = 0.0
    for iso in iso_list:
        N = float(params.get(f"N_{iso}", params.get(f"S_{iso}", 0.0)))
        if N > 0:
            mu_signal += N

    width = max(E_hi - E_lo, 1e-12)
    # background from density b0 + b1*E
    b0 = float(params.get("b0", 0.0))
    b1 = float(params.get("b1", 0.0))
    mu_bkg_shape = b0 * width + 0.5 * b1 * (E_hi**2 - E_lo**2)

    # optional S_bkg (flat over window) is total counts, not a density
    S_bkg = float(params.get("S_bkg", 0.0))

    return mu_signal + mu_bkg_shape + S_bkg
