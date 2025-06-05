# -----------------------------------------------------
# fitting.py
# -----------------------------------------------------

import numpy as np
from iminuit import Minuit
from scipy.optimize import curve_fit

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


def fit_spectrum(energies, priors, flags=None):
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

    # Histogram for chi2 fit using Freedman-Diaconis rule
    hist, edges = np.histogram(e, bins="fd")
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    # Helper to fetch prior values
    def p(name, default):
        return priors.get(name, (default, 1.0))

    param_order = [
        "sigma_E",
        "mu_Po210",
        "S_Po210",
        "mu_Po218",
        "S_Po218",
        "mu_Po214",
        "S_Po214",
        "b0",
        "b1",
    ]

    p0 = []
    bounds_lo, bounds_hi = [], []
    eps = 1e-12
    for name in param_order:
        mean, sig = p(name, 1.0)
        p0.append(mean)
        if flags.get(f"fix_{name}", False) or sig == 0:
            # curve_fit requires lower < upper; use a tiny width around fixed values
            bounds_lo.append(mean - eps)
            bounds_hi.append(mean + eps)
        else:
            delta = 5 * sig if np.isfinite(sig) else np.inf
            bounds_lo.append(mean - delta)
            bounds_hi.append(mean + delta)

    def model(x, sigma_E, mu210, S210, mu218, S218, mu214, S214, b0, b1):
        y = b0 + b1 * x
        y += S210 / (sigma_E * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu210) / sigma_E) ** 2)
        y += S218 / (sigma_E * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu218) / sigma_E) ** 2)
        y += S214 / (sigma_E * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu214) / sigma_E) ** 2)
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
    out = {}
    for i, name in enumerate(param_order):
        out[name] = float(popt[i])
        out["d" + name] = float(perr[i])

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
    # Term1 = E * (T - (1 - e^{-lam T})/lam )
    decay_term = (1.0 - np.exp(-lam * T)) / lam
    term_E = E * (T - decay_term)
    term_N0 = N0 * (1.0 - np.exp(-lam * T))
    return eff * (term_E + term_N0) + B * T


def _neg_log_likelihood_time(
    params,  # flattened list of all parameters in order
    times_dict,
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
    times_dict: { iso: 1D np.ndarray of absolute timestamps }
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
        if len(times_iso) > 0:
            # Calculate rate r(t_i_rel) at each observed time:
            t_rel = times_iso - t_start
            # r_iso(t_rel) = eff * [ E*(1 - exp(-lam*t_rel)) + lam*N0*exp(-lam*t_rel) ] + B
            rate_vals = (
                eff
                * (
                    E_iso * (1.0 - np.exp(-lam * t_rel))
                    + lam * N0_iso * np.exp(-lam * t_rel)
                )
                + B_iso
            )
            # If any rate_vals   0, penalize heavily:
            if np.any(rate_vals <= 0):
                return 1e50
            nll -= np.sum(np.log(rate_vals))  # because we want NEG log L
        # Add the integral term:
        nll += integral

    return nll


def fit_time_series(times_dict, t_start, t_end, config):
    """
    times_dict: { "Po214": np.ndarray([...]), "Po218": np.ndarray([...]) }, absolute UNIX time (sec)
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

    # 1) Build maps: lam_map, eff_map, fix_b_map, fix_n0_map
    lam_map, eff_map = {}, {}
    fix_b_map, fix_n0_map = {}, {}
    for iso in iso_list:
        iso_cfg = config["isotopes"][iso]
        hl = float(iso_cfg["half_life_s"])
        lam_map[iso] = np.log(2.0) / hl
        eff_map[iso] = float(iso_cfg.get("efficiency", 1.0))
        fix_b_map[iso] = not bool(config.get("fit_background", False))
        fix_n0_map[iso] = not bool(config.get("fit_initial", False))

    # 2) Decide parameter ordering. We always fit E_iso, then optionally B_iso, N0_iso.
    param_indices = {}  # name   index in the flat parameter tuple
    initial_guesses = []
    limits = {}

    idx = 0
    for iso in iso_list:
        #    E_iso
        param_indices[f"E_{iso}"] = idx
        # Make a  smart  initial guess: (#events)/(T_rel*eff) or 1e-3 if zero
        Ntot = len(times_dict.get(iso, []))
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
            initial_guesses.append(float(config.get("background_guess", 0.0)))
            limits[f"B_{iso}"] = (0.0, None)
            idx += 1

        #    N0_iso (if not fixed)
        if not fix_n0_map[iso]:
            param_indices[f"N0_{iso}"] = idx
            # N0 guess = 10% of total events (very rough) or zero
            guess_N0 = Ntot * 0.1 if Ntot > 0 else 0.0
            initial_guesses.append(guess_N0)
            limits[f"N0_{iso}"] = (0.0, None)
            idx += 1

    # 3) Build the Minuit minimizer
    def _nll_minuit_wrapper(*args):
        return _neg_log_likelihood_time(
            args,
            times_dict,
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
    out["fit_valid"] = True
    for pname in ordered_params:
        out[pname] = float(m.values[pname])
        out["d" + pname] = float(m.errors[pname]
                                 if pname in m.errors else np.nan)

    return out


# -----------------------------------------------------
# End of fitting.py
# -----------------------------------------------------
