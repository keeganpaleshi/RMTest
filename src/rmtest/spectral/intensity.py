from collections.abc import Mapping

import numpy as np

from constants import safe_exp as _safe_exp
from .window_norm import normalize_pdf_to_window


def _loglin_unit_shape(E, beta0, beta1, Emin, Emax, *, n_norm=512):
    """Return a unit-area log-linear background shape."""

    if n_norm <= 0:
        n_norm = 512
    Eref = 0.5 * (Emin + Emax)
    grid = np.linspace(Emin, Emax, int(n_norm))
    exp_grid = _safe_exp(beta0 + beta1 * (grid - Eref))
    Z = np.trapz(exp_grid, grid)
    Z = max(Z, 1e-300)

    E = np.asarray(E, dtype=float)
    return _safe_exp(beta0 + beta1 * (E - Eref)) / Z


def build_spectral_intensity(
    iso_list,
    use_emg,
    domain,
    clip_floor=1e-300,
    *,
    background_model=None,
):
    """
    Returns spectral_intensity_E(E, params, domain, ...) that yields λ(E) in counts/MeV.
    Peaks are normalized to the fit window exactly once.

    Parameters
    ----------
    iso_list : list
        List of isotope names.
    use_emg : bool or dict
        Whether to use EMG tails for each isotope.
    domain : tuple
        Energy window (E_lo, E_hi) in MeV.
    clip_floor : float, optional
        Small positive floor applied to per-E PDFs to avoid log(0). Default 1e-300.
        Values are clipped below clip_floor to keep log-likelihood finite.

    Returns
    -------
    callable
        Function that computes spectral intensity at given energies.
    """
    E_lo, E_hi = domain
    if isinstance(use_emg, Mapping):
        def kind_for_iso(iso):
            return "emg" if bool(use_emg.get(iso, False)) else "gauss"
    else:
        kind = "emg" if use_emg else "gauss"

        def kind_for_iso(_iso):
            return kind

    E_lo, E_hi = domain

    if str(background_model).lower() == "loglin_unit":
        background_shape = lambda E, beta0, beta1: _loglin_unit_shape(
            E, beta0, beta1, E_lo, E_hi
        )
    else:
        background_shape = None

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
            iso_kind = kind_for_iso(iso)
            tau = (
                float(params.get(f"tau_{iso}", 0.0))
                if iso_kind == "emg"
                else None
            )
            if iso_kind == "emg" and (tau is None or tau <= 0.0):
                iso_kind = "gauss"
                tau = None

            pdf_win, _ = normalize_pdf_to_window(
                iso_kind,
                mu,
                sigma,
                E_lo,
                E_hi,
                tau=tau,
            )
            lam += N * pdf_win(E)

        if background_shape is not None:
            beta0 = float(params.get("b0", 0.0))
            beta1 = float(params.get("b1", 0.0))
            S_bkg = float(params.get("S_bkg", 0.0))
            lam += S_bkg * background_shape(E, beta0, beta1)
        else:
            # Background density: b0 + b1*E (counts/MeV). If S_bkg is present, it is total counts in window.
            b0 = float(params.get("b0", 0.0))
            b1 = float(params.get("b1", 0.0))
            lam += (b0 + b1 * E)

            # If S_bkg is explicitly given, add it as a flat density over the window.
            if "S_bkg" in params:
                S_bkg = float(params["S_bkg"])
                width = max(E_hi - E_lo, 1e-12)
                lam += (S_bkg / width)

        # Apply clipping floor to keep log-likelihood finite
        return np.clip(lam, clip_floor, np.inf)

    return spectral_intensity_E


def integral_of_intensity(
    params, domain, iso_list=None, use_emg=False, *, background_model=None, **kwargs
):
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

    if str(background_model).lower() == "loglin_unit":
        S_bkg = float(params.get("S_bkg", 0.0))
        mu_bkg_shape = S_bkg
    else:
        width = max(E_hi - E_lo, 1e-12)
        # background from density b0 + b1*E
        b0 = float(params.get("b0", 0.0))
        b1 = float(params.get("b1", 0.0))
        mu_bkg_shape = b0 * width + 0.5 * b1 * (E_hi**2 - E_lo**2)

        # optional S_bkg (flat over window) is total counts, not a density
        S_bkg = float(params.get("S_bkg", 0.0))

    return mu_signal + mu_bkg_shape + S_bkg
