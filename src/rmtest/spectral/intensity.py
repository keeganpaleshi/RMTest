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


def _sigmoid_unit_shape(E, slope, E_half, Emin, Emax, *, n_norm=None):
    """Unit-area sigmoid (logistic) background shape.

    The profile is ``1 / (1 + exp(slope * (E - E_half)))``
    normalized over ``[Emin, Emax]``.

    This models a background that is high at low energy and falls off
    smoothly towards high energy — characteristic of degraded-alpha
    continua in PIN photodiode alpha spectrometers.

    Parameters
    ----------
    E : array-like
        Energies at which to evaluate.
    slope : float
        Steepness of the sigmoid transition (MeV^-1).  Positive slope
        means high on the left, low on the right.
    E_half : float
        Energy at which the background drops to half its maximum.
    Emin, Emax : float
        Integration window for normalization.
    n_norm : int, optional
        Number of quadrature points for normalization integral.
    """
    if n_norm is None or n_norm <= 0:
        n_norm = LOG_LIN_UNIT_NORM_SAMPLES
    grid = np.linspace(Emin, Emax, int(n_norm))
    sig_grid = 1.0 / (1.0 + _safe_exp(slope * (grid - E_half)))
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    Z = _trapz(sig_grid, grid)
    Z = max(Z, 1e-300)

    E = np.asarray(E, dtype=float)
    sig_eval = 1.0 / (1.0 + _safe_exp(slope * (E - E_half)))
    return sig_eval / Z


def _double_logit_unit_shape(E, E_half1, slope1, E_half2, slope2, Emin, Emax, *, n_norm=None):
    """Unit-area double-logit background shape with two inflection points.

    Models a background with two sigmoid transitions:
        shape(E) ∝ sigmoid(E; slope1, E_half1) * sigmoid(E; slope2, E_half2)

    The first sigmoid (positive slope1) produces the low-energy drop-off
    (e.g., microphonics dying off).  The second sigmoid (positive slope2)
    produces the high-energy drop-off above the alpha peaks.

    Each sigmoid has the form 1/(1+exp(slope*(E-E_half))).

    Parameters
    ----------
    E : array-like
        Energies at which to evaluate.
    E_half1 : float
        Midpoint of the first (low-energy) sigmoid transition.
    slope1 : float
        Steepness of the first sigmoid (MeV^-1).  Positive = high left, low right.
    E_half2 : float
        Midpoint of the second (high-energy) sigmoid transition.
    slope2 : float
        Steepness of the second sigmoid (MeV^-1).
    Emin, Emax : float
        Integration window for normalization.
    n_norm : int, optional
        Number of quadrature points.
    """
    if n_norm is None or n_norm <= 0:
        n_norm = LOG_LIN_UNIT_NORM_SAMPLES

    def _double_logit(x):
        s1 = 1.0 / (1.0 + _safe_exp(slope1 * (x - E_half1)))
        s2 = 1.0 / (1.0 + _safe_exp(slope2 * (x - E_half2)))
        return s1 * s2

    grid = np.linspace(Emin, Emax, int(n_norm))
    shape_grid = _double_logit(grid)
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    Z = _trapz(shape_grid, grid)
    Z = max(Z, 1e-300)

    E = np.asarray(E, dtype=float)
    return _double_logit(E) / Z


def _exp_unit_shape(E, alpha, Emin, Emax, *, n_norm=None):
    """Unit-area exponential background shape.

    The profile is ``exp(-alpha * (E - Emin))`` normalized over
    ``[Emin, Emax]``.
    """
    if n_norm is None or n_norm <= 0:
        n_norm = LOG_LIN_UNIT_NORM_SAMPLES
    grid = np.linspace(Emin, Emax, int(n_norm))
    exp_grid = _safe_exp(-alpha * (grid - Emin))
    _trapz = getattr(np, "trapezoid", None) or np.trapz
    Z = _trapz(exp_grid, grid)
    Z = max(Z, 1e-300)

    E = np.asarray(E, dtype=float)
    return _safe_exp(-alpha * (E - Emin)) / Z


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
    shelf_cutoff_delta=None,
    adc_edge_components=False,
    bkg_range=None,
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

    _bkg_lower = str(background_model).lower() if background_model else ""
    is_logquad = _bkg_lower in ("loglin_unit", "logquad_unit")
    is_sigmoid = _bkg_lower == "sigmoid_unit"
    is_exp = _bkg_lower == "exp_unit"
    is_double_logit = _bkg_lower == "double_logit_unit"
    is_none = _bkg_lower == "none"

    # Background range: if bkg_range is set, the shaped background (exp, loglin, etc.)
    # is restricted to [bkg_lo, bkg_hi] and zero outside.  This prevents the
    # exponential from dominating ADC edge regions when the fit range is very wide.
    bkg_lo = E_lo
    bkg_hi = E_hi
    if bkg_range is not None:
        bkg_lo, bkg_hi = float(bkg_range[0]), float(bkg_range[1])

    if is_logquad:
        background_shape = lambda E, beta0, beta1, beta2=0.0, beta3=0.0: _loglin_unit_shape(
            E, beta0, beta1, bkg_lo, bkg_hi, n_norm=loglin_n_norm, beta2=beta2, beta3=beta3
        )
    elif is_sigmoid:
        background_shape = lambda E, beta0, beta1, beta2=0.0, beta3=0.0: _sigmoid_unit_shape(
            E, slope=beta1, E_half=beta0, Emin=bkg_lo, Emax=bkg_hi, n_norm=loglin_n_norm,
        )
    elif is_exp:
        background_shape = lambda E, beta0, beta1, beta2=0.0, beta3=0.0: _exp_unit_shape(
            E, alpha=beta1, Emin=bkg_lo, Emax=bkg_hi, n_norm=loglin_n_norm,
        )
    elif is_double_logit:
        # Double logit: b0=E_half1, b1=slope1, b2=E_half2, b3=slope2
        background_shape = lambda E, beta0, beta1, beta2=0.0, beta3=0.0: _double_logit_unit_shape(
            E, E_half1=beta0, slope1=beta1, E_half2=beta2, slope2=beta3,
            Emin=bkg_lo, Emax=bkg_hi, n_norm=loglin_n_norm,
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

            # Right-side broadening: sigma_asym makes the right side of the
            # peak fall off with a wider Gaussian.  The left side (EMG tail)
            # is unchanged.  sigma_right = sigma * (1 + sigma_asym).
            # When sigma_asym = 0, this is a pure symmetric EMG.
            # Per-isotope sigma_asym_{iso} (from energy model) takes precedence
            # over shared sigma_asym.
            # Free sigma_right_{iso} takes precedence over sigma_asym scaling
            _free_sr_key = f"sigma_right_{iso}"
            _free_sr = float(params.get(_free_sr_key, 0.0))
            sigma_asym = float(params.get(f"sigma_asym_{iso}", params.get("sigma_asym", 0.0)))
            if _free_sr > 0.0:
                sigma_right = _free_sr
            elif sigma_asym > 0.0:
                sigma_right = sigma * (1.0 + sigma_asym)
            else:
                sigma_right = 0.0
            if sigma_right > 0.0:
                # Use split-Gaussian for the core, window-normalized
                split_win, _ = normalize_pdf_to_window(
                    "split_gauss",
                    mu,
                    sigma,  # sigma_left
                    E_lo,
                    E_hi,
                    sigma_right=sigma_right,
                )
                if iso_kind == "emg":
                    # Splice: use left-EMG below mu (preserves exponential tail),
                    # split-Gaussian above mu (wider right falloff).
                    # Scale to match at mu for continuity.
                    emg_at_mu = pdf_win(np.array([mu]))[0]
                    split_at_mu = split_win(np.array([mu]))[0]
                    if split_at_mu > 0:
                        scale = emg_at_mu / split_at_mu
                    else:
                        scale = 1.0
                    core_pdf = np.where(
                        E <= mu,
                        pdf_win(E),                    # left-EMG below peak
                        scale * split_win(E),          # wider Gaussian above peak
                    )
                    # Renormalize to integrate to 1 over [E_lo, E_hi]
                    dx = (E_hi - E_lo) / len(E)
                    total = np.sum(core_pdf) * dx
                    if total > 0:
                        core_pdf = core_pdf / total
                else:
                    core_pdf = split_win(E)
            else:
                core_pdf = pdf_win(E)

            # Additive right-side exponential tail: captures slow falloff
            # beyond what the wider Gaussian (sigma_asym) can model.
            # core = core + f_tail_right * (right_EMG - Gaussian)
            # The (right_EMG - Gaussian) term isolates the tail excess;
            # integrates to ~0 so normalization is preserved.
            # When tau_right → 0, correction → 0 regardless of f_tail_right.
            f_tail_right = float(params.get("f_tail_right", 0.0))
            tau_tail_right = float(params.get("tau_tail_right", 0.0))
            if f_tail_right > 0.0 and tau_tail_right > 0.0:
                right_win, _ = normalize_pdf_to_window(
                    "right_emg", mu, sigma, E_lo, E_hi, tau=tau_tail_right,
                )
                gauss_win, _ = normalize_pdf_to_window(
                    "gauss", mu, sigma, E_lo, E_hi,
                )
                right_excess = right_win(E) - gauss_win(E)
                core_pdf = core_pdf + f_tail_right * right_excess
                core_pdf = np.maximum(core_pdf, 0.0)

            # Build the composite peak shape
            # Fractions: f_core + f_halo + f_shelf = 1
            # where f_core = 1 - f_halo - f_shelf
            f_core = max(1.0 - f_halo - f_shelf, 0.0)
            peak_density = f_core * core_pdf

            # Halo (broad) component — a wider Gaussian/EMG that captures
            # detector-response broadening from dead-layer scattering.
            # Halo is LEFT-SIDE ONLY: degraded alphas lose energy, so the
            # halo must not extend above the peak.  Cutoff at mu - delta
            # (same as shelf) ensures physical correctness.
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
                halo_pdf = halo_win(E)
                # Apply left-side cutoff: zero above mu - delta
                _halo_cutoff = mu
                if shelf_cutoff_delta is not None and shelf_cutoff_delta > 0:
                    _halo_cutoff = mu - shelf_cutoff_delta
                halo_pdf = np.where(E <= _halo_cutoff, halo_pdf, 0.0)
                # Renormalize after cutoff
                dx = (E_hi - E_lo) / len(E)
                _halo_total = np.sum(halo_pdf) * dx
                if _halo_total > 0:
                    halo_pdf = halo_pdf / _halo_total
                peak_density += f_halo * halo_pdf

            # Shelf component
            if use_shelf.get(iso, False) and f_shelf > 0.0:
                # Use separate shelf width if provided, otherwise fall back to peak sigma
                sigma_shelf = float(params.get(f"sigma_shelf_{iso}", sigma))
                # Per-isotope shelf range: dict or scalar
                if isinstance(shelf_range, Mapping):
                    _iso_shelf_range = float(shelf_range.get(iso, shelf_range.get("default", 1.0)))
                else:
                    _iso_shelf_range = shelf_range
                shelf_win, _ = normalize_pdf_to_window(
                    "shelf",
                    mu,
                    sigma_shelf,
                    E_lo,
                    E_hi,
                    shelf_range=_iso_shelf_range,
                    shelf_cutoff_delta=shelf_cutoff_delta,
                )
                peak_density += f_shelf * shelf_win(E)

            # Beta coincidence: high-energy tail from simultaneous detection
            # of alpha + beta (e.g. Bi-214 beta + Po-214 alpha).  One-sided
            # exponential above the peak, normalized over [E_lo, E_hi].
            # f_beta is an *additional* fraction (not subtracted from f_core).
            f_beta = float(params.get(f"f_beta_{iso}", 0.0))
            if f_beta > 0.0:
                lam_beta = max(float(params.get(f"lambda_beta_{iso}", 0.5)), 1e-6)
                # Analytic normalization: integral of exp(-(E-mu)/lam)/lam
                # from mu to E_hi = 1 - exp(-(E_hi-mu)/lam)
                _beta_norm = lam_beta * (1.0 - np.exp(-max(E_hi - mu, 0.0) / lam_beta))
                if _beta_norm > 1e-30:
                    beta_pdf = np.where(
                        E >= mu,
                        np.exp(-(E - mu) / lam_beta) / _beta_norm,
                        0.0,
                    )
                    lam += N * f_beta * beta_pdf

            lam += N * peak_density

        if background_shape is not None:
            beta0 = float(params.get("b0", 0.0))
            beta1 = float(params.get("b1", 0.0))
            beta2 = float(params.get("b2", 0.0))
            beta3 = float(params.get("b3", 0.0))
            S_bkg = float(params.get("S_bkg", 0.0))
            bkg_density = S_bkg * background_shape(E, beta0, beta1, beta2, beta3)
            # Zero the shaped background outside bkg_range (ADC edges handle those regions)
            if bkg_range is not None:
                bkg_density = np.where((E >= bkg_lo) & (E <= bkg_hi), bkg_density, 0.0)
            lam += bkg_density
        elif is_none:
            # "none" background model: no polynomial background at all.
            # Only an optional flat S_bkg (microphonics / residual noise floor).
            if "S_bkg" in params:
                S_bkg = float(params["S_bkg"])
                width = max(E_hi - E_lo, 1e-12)
                lam += (S_bkg / width)
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

        # ADC edge components: exponential decay from the low/high energy
        # boundaries, hard-clipped to a limited range so they CANNOT leak
        # into the alpha peak region and distort peak shapes.
        # Low edge:  S * exp(-(E - E_lo) / w)  for E in [E_lo, E_lo + cutoff]
        # High edge: S * exp(-(E_hi - E) / w)  for E in [E_hi - cutoff, E_hi]
        if adc_edge_components:
            _adc_cutoff = 1.5  # MeV — max reach from boundary (keeps edges away from alpha peaks)

            S_lo = float(params.get("S_adc_lo", 0.0))
            if S_lo > 0.0:
                w_lo = float(params.get("w_adc_lo", 0.3))
                w_lo = max(w_lo, 1e-6)
                # Exponential decay from low edge, zero beyond cutoff
                adc_lo_density = np.where(
                    E <= E_lo + _adc_cutoff,
                    np.exp(-(E - E_lo) / w_lo),
                    0.0,
                )
                # Normalize to unit area over active region
                dx = (E_hi - E_lo) / max(len(E), 1)
                norm_lo = np.sum(adc_lo_density) * dx
                if norm_lo > 0:
                    adc_lo_density = adc_lo_density / norm_lo
                lam += S_lo * adc_lo_density

            S_hi = float(params.get("S_adc_hi", 0.0))
            if S_hi > 0.0:
                w_hi = float(params.get("w_adc_hi", 0.3))
                w_hi = max(w_hi, 1e-6)
                # Exponential decay from high edge, zero beyond cutoff
                adc_hi_density = np.where(
                    E >= E_hi - _adc_cutoff,
                    np.exp(-(E_hi - E) / w_hi),
                    0.0,
                )
                dx = (E_hi - E_lo) / max(len(E), 1)
                norm_hi = np.sum(adc_hi_density) * dx
                if norm_hi > 0:
                    adc_hi_density = adc_hi_density / norm_hi
                lam += S_hi * adc_hi_density

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
            # Beta coincidence adds extra counts (not subtracted from core)
            f_beta = float(params.get(f"f_beta_{iso}", 0.0))
            if f_beta > 0.0:
                mu_signal += N * f_beta

    _bkg_model_lower = str(background_model).lower() if background_model else ""
    if _bkg_model_lower in ("loglin_unit", "sigmoid_unit", "exp_unit", "double_logit_unit"):
        # Unit-area shaped background; S_bkg is total background counts
        mu_bkg = float(params.get("S_bkg", 0.0))
    elif _bkg_model_lower == "none":
        # No polynomial background — only optional flat S_bkg (microphonics)
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
