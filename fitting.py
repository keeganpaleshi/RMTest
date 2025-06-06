# -----------------------------------------------------
# fitting.py
# -----------------------------------------------------

import logging
import numpy as np
from iminuit import Minuit
from scipy.optimize import curve_fit
from calibration import emg_left, gaussian
from constants import _TAU_MIN

# Prevent overflow in exp calculations. Values beyond ~700 in magnitude
# lead to inf/0 under IEEE-754 doubles.  Clip the exponent to a safe range
# so the likelihood remains finite during optimization.
_EXP_LIMIT = 700.0

# Minimum allowed value for the exponential tail constant to avoid
# divide-by-zero overflow when evaluating the EMG component. The
# value itself lives in :mod:`constants` as ``_TAU_MIN``.

def _safe_exp(x):
    """Return ``exp(x)`` with the input clipped to ``[-_EXP_LIMIT, _EXP_LIMIT]``."""
    return np.exp(np.clip(x, -_EXP_LIMIT, _EXP_LIMIT))

__all__ = ["fit_time_series", "fit_decay", "fit_spectrum"]


def fit_decay(times, priors, t0=0.0, t_end=None, flags=None):
    """Simple rate estimator used for unit tests.

    Parameters
    ----------
    times : array-like
        Event times (relative to ``t0``) in seconds.
    priors : dict
        Dictionary that may contain an ``"eff"`` entry specifying the detection
        efficiency.
    t0 : float, optional
        Start time of the interval.  Only used to compute the total exposure
        ``t_end - t0``.
    t_end : float, optional
        End time of the interval.  If ``None`` the maximum time in ``times`` is
        used.
    flags : dict, optional
        Additional flags (ignored by this simple implementation).

    Returns
    -------
    dict
        Dictionary with at least the keys ``"E"``, ``"N0"``, ``"B"`` and
        ``"eff"`` representing a naive rate estimate and placeholder values for
        initial population and background.
    """

    if flags is None:
        flags = {}

    t = np.asarray(times, dtype=float)
    if t_end is None:
        T = float(t.max() if t.size > 0 else 0.0) - float(t0)
    else:
        T = float(t_end) - float(t0)

    eff = float(priors.get("eff", (1.0, 0.0))[0])

    count = len(t)
    rate = count / (T * eff) if (T > 0 and eff > 0) else 0.0

    return {
        "E": rate,
        "N0": 0.0,
        "B": 0.0,
        "eff": eff,
    }


def fit_spectrum(energies, priors, flags=None, bins=None, bin_edges=None, bounds=None):
    """Fit three Gaussian peaks with a linear background to the spectrum.

    Parameters
    ----------
    energies : array-like
        Energy values (MeV).
    priors : dict
        Parameter priors of the form {name: (mu, sigma)}.
    flags : dict, optional
        Flags such as ``{"fix_sigma_E": True}`` to fix parameters. Fixed
        parameters are implemented by constraining the optimizer to a tiny
        interval (``±1e-12``) around the provided mean value.
    bins : int or sequence, optional
        Number of bins or bin edges to use when histogramming the input
        energies.  Ignored if ``bin_edges`` is provided.  If both ``bins``
        and ``bin_edges`` are ``None``, the Freedman--Diaconis rule is used.
    bin_edges : array-like, optional
        Explicit bin edges for histogramming the energies.  Takes precedence
        over ``bins`` when given.
    bounds : dict, optional
        Mapping of parameter name to ``(lower, upper)`` tuples overriding the
        default ±5σ range derived from the priors.  ``None`` values disable a
        limit on that side.

    Returns
    -------
    dict
        Best fit values and uncertainties.
    """

    if flags is None:
        flags = {}

    e = np.asarray(energies, dtype=float)
    if e.size == 0:
        raise RuntimeError("No energies provided to fit_spectrum")

    # Histogram according to provided binning parameters
    if bin_edges is not None:
        hist, edges = np.histogram(e, bins=np.asarray(bin_edges))
    elif bins is not None:
        hist, edges = np.histogram(e, bins=bins)
    else:
        # Default: Freedman-Diaconis rule
        hist, edges = np.histogram(e, bins="fd")
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    # Guard against NaNs/Infs arising from unstable histogramming or EMG evals
    if not np.isfinite(hist).all():
        raise RuntimeError(
            "fit_spectrum: histogram contains non-finite values; "
            "check input energies and binning parameters"
        )
    if not np.isfinite(centers).all():
        raise RuntimeError(
            "fit_spectrum: histogram centers contain non-finite values; "
            "check input energies and binning parameters"
        )

    # Helper to fetch prior values
    def p(name, default):
        return priors.get(name, (default, 1.0))

    # Determine which peaks should include an EMG tail based on provided priors
    use_emg = {
        "Po210": "tau_Po210" in priors,
        "Po218": "tau_Po218" in priors,
        "Po214": "tau_Po214" in priors,
    }

    param_order = ["sigma_E"]
    for iso in ("Po210", "Po218", "Po214"):
        param_order.extend([f"mu_{iso}", f"S_{iso}"])
        if use_emg[iso]:
            param_order.append(f"tau_{iso}")
    param_order.extend(["b0", "b1"])

    p0 = []
    bounds_lo, bounds_hi = [], []
    eps = 1e-12
    for name in param_order:
        mean, sig = p(name, 1.0)
        # Enforce a strictly positive initial tau to avoid singular EMG tails
        if name.startswith("tau_"):
            mean = max(mean, _TAU_MIN)
        p0.append(mean)
        if flags.get(f"fix_{name}", False) or sig == 0:
            # curve_fit requires lower < upper; use a tiny width around fixed values
            lo = mean - eps
            hi = mean + eps
        else:
            delta = 5 * sig if np.isfinite(sig) else np.inf
            lo = mean - delta
            hi = mean + delta
        if bounds and name in bounds:
            user_lo, user_hi = bounds[name]
            if user_lo is not None:
                lo = max(lo, user_lo)
            if user_hi is not None:
                hi = min(hi, user_hi)
        if name.startswith("tau_"):
            lo = max(lo, _TAU_MIN)
        bounds_lo.append(lo)
        bounds_hi.append(hi)

    iso_list = ["Po210", "Po218", "Po214"]

    def model(x, *params):
        idx = 0
        sigma_E = params[idx]
        idx += 1
        y = np.zeros_like(x)
        for iso in iso_list:
            mu = params[idx]
            idx += 1
            S = params[idx]
            idx += 1
            if use_emg[iso]:
                tau = params[idx]
                idx += 1
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    y_emg = emg_left(x, mu, sigma_E, tau)
                y_emg = np.nan_to_num(y_emg, nan=0.0, posinf=0.0, neginf=0.0)
                y += S * y_emg
            else:
                y += S * gaussian(x, mu, sigma_E)
        b0 = params[idx]
        b1 = params[idx + 1]
        y = y + b0 + b1 * x
        return y * width

    popt, pcov = curve_fit(
        model,
        centers,
        hist,
        p0=p0,
        bounds=(bounds_lo, bounds_hi),
        maxfev=10000,
    )

    perr = np.sqrt(np.diag(pcov))
    try:
        eigvals = np.linalg.eigvals(pcov)
        fit_valid = bool(np.all(eigvals > 0))
    except np.linalg.LinAlgError:
        fit_valid = False

    if not fit_valid:
        logging.warning("fit_spectrum: covariance matrix not positive definite")
        # Add a small diagonal jitter to attempt stabilising the matrix
        jitter = 1e-12 * np.mean(np.diag(pcov))
        if not np.isfinite(jitter) or jitter <= 0:
            jitter = 1e-12
        pcov = pcov + jitter * np.eye(pcov.shape[0])
        try:
            np.linalg.cholesky(pcov)
            fit_valid = True
            perr = np.sqrt(np.diag(pcov))
        except np.linalg.LinAlgError:
            pass
    out = {}
    for i, name in enumerate(param_order):
        out[name] = float(popt[i])
        out["d" + name] = float(perr[i])

    out["fit_valid"] = fit_valid
    return out


def _integral_model(E, N0, B, lam, eff, T):
    """
    Analytic integral of: eff * [E*(1 - exp(-lam*t)) + lam*N0*exp(-lam*t)] + B
    from t=0 to t=T:
           eff*(E - E*exp(-lam*t) + lam*N0*exp(-lam*t)) dt + B*T
    = eff * [ E*( T - (1 - exp(-lam*T))/lam ) + lam*N0*( (1 - exp(-lam*T))/lam ) ] + B*T
    """
    if lam <= 0:
        # In principle lam should never be <=0; return a large number to penalize
        return 1e50
    if T < 0:
        logging.debug("_integral_model called with negative T; using |T|")
        T = abs(T)
    # Term1 = E * (T - (1 - e^{-lam T})/lam )
    exp_term = _safe_exp(-lam * T)
    decay_term = (1.0 - exp_term) / lam
    term_E = E * (T - decay_term)
    term_N0 = N0 * (1.0 - exp_term)
    return eff * (term_E + term_N0) + B * T


def _neg_log_likelihood_time(
    params,  # flattened list of all parameters in order
    times_dict,
    weights_dict,
    t_start,
    t_end,
    iso_list,
    lam_map,
    eff_map,
    fix_b_map,
    fix_n0_map,
    param_indices,
):
    """
    params: tuple of all (E_iso, [B_iso], [N0_iso], for each iso in iso_list, in the order recorded by param_indices)
    times_dict: mapping of isotope -> array of event timestamps.
    weights_dict: mapping of isotope -> array of per-event weights or None.
    t_start, t_end: floats (absolute UNIX seconds)
    lam_map: { iso: decay_constant (1/s) }
    eff_map: { iso: detection efficiency }
    fix_b_map, fix_n0_map: booleans per iso
    param_indices: dictionary mapping each parameter name ("E_Po214", "B_Po214", "N0_Po214", etc.)
                     index into the params tuple.
    Returns: scalar negative log likelihood
    """
    nll = 0.0
    T_rel = t_end - t_start

    # Build a dict for this iteration s parameter values:
    p = {}
    for pname, idx in param_indices.items():
        p[pname] = params[idx]

    # For each isotope, compute its contribution to NLL:
    for iso in iso_list:
        lam = lam_map[iso]
        eff = eff_map[iso]

        # Extract parameters (some may be fixed to zero):
        E_iso = p[f"E_{iso}"]
        B_iso = 0.0 if fix_b_map[iso] else p[f"B_{iso}"]
        N0_iso = 0.0 if fix_n0_map[iso] else p[f"N0_{iso}"]

        # 1) Integral term:
        integral = _integral_model(E_iso, N0_iso, B_iso, lam, eff, T_rel)
        # 2) Sum of log[r(t_i)] for each event t_i in times_dict[iso]:
        times_iso = times_dict.get(iso, np.empty(0))
        weights = weights_dict.get(iso)
        if len(times_iso) > 0:
            # Calculate rate r(t_i_rel) at each observed time:
            t_rel = times_iso - t_start
            if np.any(t_rel < 0):
                logging.debug(
                    "fit_time_series: negative relative times detected; check t_start"
                )
            # r_iso(t_rel) = eff * [ E*(1 - exp(-lam*t_rel)) + lam*N0*exp(-lam*t_rel) ] + B
            exp_term = _safe_exp(-lam * t_rel)
            rate_vals = (
                eff
                * (
                    E_iso * (1.0 - exp_term)
                    + lam * N0_iso * exp_term
                )
                + B_iso
            )
            # If any rate_vals   0, penalize heavily:
            if np.any(rate_vals <= 0):
                return 1e50
            if weights is None:
                nll -= np.sum(np.log(rate_vals))
            else:
                nll -= np.sum(weights * np.log(rate_vals))
        # Add the integral term:
        nll += integral

    return nll


def fit_time_series(times_dict, t_start, t_end, config, weights=None):
    """
    times_dict: mapping of isotope -> array of timestamps in seconds.
    weights : dict or None
        Optional mapping of isotope -> per-event weights matching
        ``times_dict``.
    t_start, t_end: floats (absolute UNIX seconds) defining the fit window
    config: JSON dict with these keys:
          "isotopes": { "Po214": {"half_life_s": , "efficiency": ,  }, "Po218": {   } }
          "fit_background": bool
          "fit_initial": bool
          "background_guess": float  (initial guess for B_iso)
          "initial_guess":    float  (initial guess for N0_iso)
    Returns: dict with best fit values & 1  uncertainties, e.g.:
        {
          "E_Po214": 12.3,  "dE_Po214": 1.4,
          "B_Po214": 0.02, "dB_Po214": 0.005,
          "N0_Po214": 50,  "dN0_Po214": 10,
          "E_Po218": 5.6,  "dE_Po218": 0.8,
          "B_Po218": 0.01, "dB_Po218": 0.003,
          "N0_Po218": 10,  "dN0_Po218": 3,
          "fit_valid": True
        }
    """
    iso_list = list(config["isotopes"].keys())

    # Normalize weights mapping
    if weights is None:
        weights_dict = {iso: None for iso in iso_list}
    else:
        weights_dict = {iso: np.asarray(weights.get(iso), dtype=float) if weights.get(iso) is not None else None for iso in iso_list}

    # 1) Build maps: lam_map, eff_map, fix_b_map, fix_n0_map
    lam_map, eff_map = {}, {}
    fix_b_map, fix_n0_map = {}, {}
    for iso in iso_list:
        iso_cfg = config["isotopes"][iso]
        hl = float(iso_cfg["half_life_s"])
        if hl <= 0:
            raise ValueError("half_life_s must be positive")
        lam_map[iso] = np.log(2.0) / hl
        eff_map[iso] = float(iso_cfg.get("efficiency", 1.0))
        fix_b_map[iso] = not bool(config.get("fit_background", False))
        fix_n0_map[iso] = not bool(config.get("fit_initial", False))

    # 2) Decide parameter ordering. We always fit E_iso, then optionally B_iso, N0_iso.
    param_indices = {}  # name   index in the flat parameter tuple
    initial_guesses = []
    limits = {}

    background_guess = float(config.get("background_guess", 0.0))
    n0_guess_frac = float(config.get("n0_guess_fraction", 0.1))

    idx = 0
    for iso in iso_list:
        #    E_iso
        param_indices[f"E_{iso}"] = idx
        # Make a  smart  initial guess: (#events)/(T_rel*eff) or 1e-3 if zero
        times_arr = np.asarray(times_dict.get(iso, []), dtype=float)
        w_arr = weights_dict.get(iso)
        if w_arr is None:
            Ntot = len(times_arr)
        else:
            Ntot = float(np.sum(w_arr))
        T_rel = t_end - t_start
        eff = eff_map[iso]
        guess_E = max((Ntot / (T_rel * eff))
                      if (T_rel > 0 and eff > 0) else 0.0, 1e-6)
        initial_guesses.append(guess_E)
        limits[f"E_{iso}"] = (0.0, None)
        idx += 1

        #    B_iso (if not fixed)
        if not fix_b_map[iso]:
            param_indices[f"B_{iso}"] = idx
            initial_guesses.append(background_guess)
            limits[f"B_{iso}"] = (0.0, None)
            idx += 1

        #    N0_iso (if not fixed)
        if not fix_n0_map[iso]:
            param_indices[f"N0_{iso}"] = idx
            # N0 guess = fraction of total events (very rough) or zero
            guess_N0 = Ntot * n0_guess_frac if Ntot > 0 else 0.0
            initial_guesses.append(guess_N0)
            limits[f"N0_{iso}"] = (0.0, None)
            idx += 1

    # 3) Build the Minuit minimizer
    def _nll_minuit_wrapper(*args):
        return _neg_log_likelihood_time(
            args,
            times_dict,
            weights_dict,
            t_start,
            t_end,
            iso_list,
            lam_map,
            eff_map,
            fix_b_map,
            fix_n0_map,
            param_indices,
        )

    # Collect parameter names in the same order as initial_guesses
    ordered_params = [None] * len(initial_guesses)
    for name, i in param_indices.items():
        ordered_params[i] = name

    m = Minuit(_nll_minuit_wrapper, *initial_guesses, name=ordered_params)
    m.errordef = Minuit.LIKELIHOOD

    # 4) Apply the limits
    for pname, (lo, hi) in limits.items():
        m.limits[pname] = (lo, hi)

    # 5) Run the fit
    m.migrad()
    if not m.valid:
        m.simplex()
        m.migrad()

    out = {}
    if not m.valid:
        out["fit_valid"] = False
        # Still return whatever values Minuit has found
        for pname in ordered_params:
            val = float(m.values[pname])
            err = float(m.errors[pname]) if pname in m.errors else np.nan
            out[pname] = val
            out["d" + pname] = err
        return out

    m.hesse()  # compute uncertainties
    cov = np.array(m.covariance)
    perr = np.sqrt(np.diag(cov))
    try:
        eigvals = np.linalg.eigvals(cov)
        fit_valid = bool(np.all(eigvals > 0))
    except np.linalg.LinAlgError:
        fit_valid = False
    if not fit_valid:
        logging.warning("fit_time_series: covariance matrix not positive definite")
        jitter = 1e-12 * np.mean(np.diag(cov))
        if not np.isfinite(jitter) or jitter <= 0:
            jitter = 1e-12
        cov = cov + jitter * np.eye(cov.shape[0])
        try:
            np.linalg.cholesky(cov)
            fit_valid = True
            perr = np.sqrt(np.diag(cov))
        except np.linalg.LinAlgError:
            fit_valid = False
    out["fit_valid"] = fit_valid
    for i, pname in enumerate(ordered_params):
        out[pname] = float(m.values[pname])
        out["d" + pname] = float(perr[i] if i < len(perr) else np.nan)

    return out


# -----------------------------------------------------
# End of fitting.py
# -----------------------------------------------------
