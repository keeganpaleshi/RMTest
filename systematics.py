import math
import numpy as np
from typing import Callable, Dict, Tuple, List


def apply_linear_adc_shift(
    adc_values,
    timestamps,
    rate,
    t_ref=None,
    mode: str = "linear",
    params: dict | None = None,
):
    """Apply a time-dependent shift to ADC values.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC readings.
    timestamps : array-like
        Event timestamps in seconds.
    rate : float
        ADC shift per second when ``mode`` is ``"linear"``.  Ignored otherwise.
    t_ref : float, optional
        Reference time for zero shift.  Defaults to the first timestamp.
    mode : {"linear", "quadratic", "piecewise"}, optional
        Type of time dependence.  ``"linear"`` reproduces the original
        behaviour.  ``"quadratic"`` uses coefficients from ``params`` and
        ``"piecewise"`` performs linear interpolation between points given in
        ``params``.
    params : dict, optional
        Additional parameters required for ``"quadratic"`` and ``"piecewise"``
        modes.  For ``"quadratic"`` provide ``{"a": A, "b": B}`` such that the
        shift is ``A*(t - t_ref)**2 + B*(t - t_ref)``.  For ``"piecewise"``
        provide ``{"times": [...], "shifts": [...]}`` arrays of equal length.

    Returns
    -------
    numpy.ndarray
        Array of shifted ADC values.
    """

    adc_arr = np.asarray(adc_values, dtype=float)
    time_arr = np.asarray(timestamps, dtype=float)

    if adc_arr.shape != time_arr.shape:
        raise ValueError("adc_values and timestamps must have the same shape")

    if t_ref is None:
        t_ref = float(time_arr[0]) if time_arr.size else 0.0

    dt = time_arr - t_ref

    if mode == "linear":
        shift = rate * dt
    elif mode == "quadratic":
        p = params or {}
        a = float(p.get("a", 0.0))
        b = float(p.get("b", rate))
        shift = a * dt ** 2 + b * dt
    elif mode == "piecewise":
        if not params or "times" not in params or "shifts" not in params:
            raise ValueError("piecewise mode requires 'times' and 'shifts'")
        times = np.asarray(params["times"], dtype=float)
        shifts = np.asarray(params["shifts"], dtype=float)
        if times.shape != shifts.shape:
            raise ValueError("times and shifts must have the same shape")
        shift = np.interp(time_arr, times, shifts)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    return adc_arr + shift


def scan_systematics(
    fit_func: Callable,
    priors: Dict[str, Tuple[float, float]],
    sigma_dict: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    """Scan parameter level systematics.

    Parameters
    ----------
    fit_func : callable
        Function that performs the fit and returns either a mapping of
        parameters or a scalar.
    priors : dict
        Mapping of parameter name to ``(mean, sigma)`` tuples.
    sigma_dict : dict
        Dictionary of parameter shift magnitudes.  Keys may be suffixed
        with ``_frac`` to indicate a fractional shift or ``_keV`` for an
        absolute shift provided in keV.  ``_keV`` values are converted to
        MeV automatically.  Keys without a suffix are interpreted as
        absolute shifts.

    Returns
    -------
    dict
        Mapping of parameter names to absolute deviations induced by the
        shifts.
    float
        Total systematic uncertainty combined in quadrature.
    """

    central = fit_func(priors)
    is_dict = isinstance(central, dict)
    if not is_dict and not isinstance(central, (int, float)):
        raise RuntimeError(
            "scan_systematics: fit_func must return dict or scalar")

    deltas = {}
    for full_key, val in sigma_dict.items():
        if full_key.endswith("_frac"):
            key = full_key[:-5]
            if key not in priors:
                raise RuntimeError(f"No prior entry for '{key}'")
            delta = val * priors[key][0]
        elif full_key.endswith("_keV"):
            key = full_key[:-4]
            delta = val / 1000.0  # convert keV → MeV
        else:
            key = full_key
            delta = val

        if is_dict:
            if key not in central:
                raise RuntimeError(f"Central fit missing parameter '{key}'")
            v0 = central[key]
        else:
            v0 = central

        if key not in priors:
            raise RuntimeError(f"No prior entry for '{key}'")

        p_plus = priors.copy()
        mu, sig = p_plus[key]
        p_plus[key] = (mu + delta, sig)
        res_plus = fit_func(p_plus)
        v_plus = res_plus[key] if is_dict else res_plus

        p_minus = priors.copy()
        p_minus[key] = (mu - delta, sig)
        res_minus = fit_func(p_minus)
        v_minus = res_minus[key] if is_dict else res_minus

        deltas[key] = max(abs(v_plus - v0), abs(v_minus - v0))

    total_unc = math.sqrt(sum(v ** 2 for v in deltas.values()))
    return deltas, total_unc
