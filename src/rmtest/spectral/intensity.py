from collections.abc import Mapping

import numpy as np

from constants import safe_exp as _safe_exp
from .window_norm import normalize_pdf_to_window


LOG_LIN_UNIT_NORM_SAMPLES = 512


def _loglin_unit_shape(E, beta0, beta1, Emin, Emax, *, n_norm=None, beta2=0.0, beta3=0.0):
    """Unit-area log-polynomial background shape.

    The profile is ``exp(beta0 + beta1*x + beta2*x^2 + beta3*x^3)``
    normalized over ``[Emin, Emax]`` where ``x = E - Eref`` and
    ``Eref`` is the window midpoint.

    When *beta2* and *beta3* are zero this reduces to the log-linear model.
    """

    if n_norm is None or n_norm <= 0:
        n_norm = LOG_LIN_UNIT_NORM_SAMPLES
    Eref = 0.5 * (Emin + Emax)
    grid = np.linspace(Emin, Emax, int(n_norm))
    dE = grid - Eref
    poly = beta0 + beta1 * dE + beta2 * dE * dE + beta3 * dE * dE * dE
    exp_grid = _safe_exp(poly)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    Z = _trapz(exp_grid, grid)
    Z = max(Z, 1e-300)

    E = np.asarray(E, dtype=float)
    dE_eval = E - Eref
    poly_eval = beta0 + beta1 * dE_eval + beta2 * dE_eval * dE_eval + beta3 * dE_eval * dE_eval * dE_eval
    return _safe_exp(poly_eval) / Z


def build_spectral_intensity(
    iso_list,
    use_emg,
    domain,
    clip_floor=1e-300,
    *,
    background_model=None,
    loglin_n_norm=None,
    use_shelf=None,
    use_halo=None,
    shelf_range=None,
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
    use_shelf : dict, optional
        Mapping of isotope name to bool indicating whether to add a low-energy
        shelf (erfc step) component to each peak.
    use_halo : dict, optional
        Mapping of isotope name to bool indicating whether to add a broad
        "halo" component (second wider peak) for each isotope.  Standard in
        alpha spectroscopy to model detector-response broadening from
        dead-layer scattering and angular effects.

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

    if use_shelf is None:
        use_shelf = {}
    if isinstance(use_shelf, bool):
        use_shelf = {iso: use_shelf for iso in iso_list}

    if use_halo is None:
        use_halo = {}
    if isinstance(use_halo, bool):
        use_halo = {iso: use_halo for iso in iso_list}

    is_logquad = str(background_model).lower() in ("loglin_unit", "logquad_unit")

    if is_logquad:
        background_shape = lambda E, beta0, beta1, beta2=0.0, beta3=0.0: _loglin_unit_shape(
            E, beta0, beta1, E_lo, E_hi, n_norm=loglin_n_norm, beta2=beta2, beta3=beta3
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
            sigma = float(params.get(f"sigma_{iso}", params["sigma0"]))
            iso_kind = kind_for_iso(iso)
            tau = (
                float(params.get(f"tau_{iso}", 0.0))
                if iso_kind == "emg"
                else None
            )
            if iso_kind == "emg" and (tau is None or tau <= 0.0):
                iso_kind = "gauss"
                tau = None

            # Shelf fraction for this isotope
            f_shelf = float(params.get(f"f_shelf_{iso}", 0.0))
            # Halo fraction for this isotope
            f_halo = float(params.get(f"f_halo_{iso}", 0.0))

            # Core peak PDF (narrow component)
            pdf_win, _ = normalize_pdf_to_window(
                iso_kind,
                mu,
                sigma,
                E_lo,
                E_hi,
                tau=tau,
            )

            # Build the composite peak shape
            # Fractions: f_core + f_halo + f_shelf = 1
            # where f_core = 1 - f_halo - f_shelf
            f_core = max(1.0 - f_halo - f_shelf, 0.0)
            peak_density = f_core * pdf_win(E)

            # Halo (broad) component — a wider Gaussian/EMG peak that
            # captures detector-response broadening from dead-layer
            # scattering and angular effects.
            if use_halo.get(iso, False) and f_halo > 0.0:
                sigma_halo = float(params.get(f"sigma_halo_{iso}", sigma * 2.0))
                # Halo may have its own EMG tail (decoupled from core tau)
                tau_halo = (
                    float(params.get(f"tau_halo_{iso}", tau if tau is not None else 0.0))
                    if iso_kind == "emg"
                    else None
                )
                halo_kind = iso_kind
                if halo_kind == "emg" and (tau_halo is None or tau_halo <= 0.0):
                    halo_kind = "gauss"
                    tau_halo = None
                halo_win, _ = normalize_pdf_to_window(
                    halo_kind,
                    mu,
                    sigma_halo,
                    E_lo,
                    E_hi,
                    tau=tau_halo,
                )
                peak_density += f_halo * halo_win(E)

            # Shelf component
            if use_shelf.get(iso, False) and f_shelf > 0.0:
                # Use separate shelf width if provided, otherwise fall back to peak sigma
                sigma_shelf = float(params.get(f"sigma_shelf_{iso}", sigma))
                shelf_win, _ = normalize_pdf_to_window(
                    "shelf",
                    mu,
                    sigma_shelf,
                    E_lo,
                    E_hi,
                    shelf_range=shelf_range,
                )
                peak_density += f_shelf * shelf_win(E)

            lam += N * peak_density

        if background_shape is not None:
            beta0 = float(params.get("b0", 0.0))
            beta1 = float(params.get("b1", 0.0))
            beta2 = float(params.get("b2", 0.0))
            beta3 = float(params.get("b3", 0.0))
            S_bkg = float(params.get("S_bkg", 0.0))
            lam += S_bkg * background_shape(E, beta0, beta1, beta2, beta3)
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
        # log-linear background is unit-area; S_bkg is total background counts
        mu_bkg = float(params.get("S_bkg", 0.0))
    else:
        # background from density b0 + b1*E
        b0 = float(params.get("b0", 0.0))
        b1 = float(params.get("b1", 0.0))
        width = max(E_hi - E_lo, 1e-12)
        mu_bkg_shape = b0 * width + 0.5 * b1 * (E_hi**2 - E_lo**2)

        # optional S_bkg (flat over window) is total counts, not a density
        S_bkg = float(params.get("S_bkg", 0.0))
        mu_bkg = mu_bkg_shape + S_bkg

    return mu_signal + mu_bkg
