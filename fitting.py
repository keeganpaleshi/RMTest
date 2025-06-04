# -----------------------------------------------------
# fitting.py
# -----------------------------------------------------

import numpy as np
from math import exp, log
from iminuit import Minuit

__all__ = ["fit_time_series", "fit_decay"]


def fit_decay(event_times, total_time, lambda_decay, efficiency, _cfg=None):
    """Simple rate estimator used for unit tests."""
    event_times = np.asarray(event_times)
    count = len(event_times)
    rate = count / (total_time * efficiency) if total_time > 0 and efficiency > 0 else 0.0
    # Return tuple mimicking (E, N0, B)
    return (rate, 0.0, 0.0), {}


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
