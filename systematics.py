import math
import numpy as np
from typing import Callable, Dict, Tuple


def apply_linear_adc_shift(adc_values, timestamps, rate, t_ref=None):
    """Apply a linear time-dependent shift to ADC values.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC readings.
    timestamps : array-like
        Event timestamps in seconds.
    rate : float
        ADC shift per second.  Positive values shift later events upward.
    t_ref : float, optional
        Reference time for zero shift.  Defaults to the first timestamp.

    Returns
    -------
    numpy.ndarray
        Array of shifted ADC values.
    """

    adc_arr = [float(v) for v in adc_values]
    time_arr = [float(t) for t in timestamps]

    if len(adc_arr) != len(time_arr):
        raise ValueError("adc_values and timestamps must have the same shape")

    if t_ref is None:
        t_ref = float(time_arr[0]) if len(time_arr) else 0.0

    return np.array(
        [a + rate * (t - t_ref) for a, t in zip(adc_arr, time_arr)],
        dtype=float,
    )


def scan_systematics(
    fit_func: Callable,
    priors: Dict[str, Tuple[float, float]],
    shifts: Dict[str, float],
) -> Tuple[Dict[str, float], float]:
    """Evaluate systematic uncertainties from fractional or absolute shifts.

    Parameters
    ----------
    fit_func : callable
        Function returning either a scalar or a mapping of fit parameters.
    priors : dict
        Mapping of parameter name to (mean, sigma).
    shifts : dict
        Keys ending in ``_frac`` are interpreted as fractional adjustments to
        the corresponding parameter.  Keys ending in ``_keV`` are absolute
        shifts in keV and are converted to MeV.  All other keys are absolute
        shifts in the same units as the priors.
    """

    central = fit_func(priors)
    is_dict = isinstance(central, dict)
    if not is_dict and not isinstance(central, (int, float)):
        raise RuntimeError("scan_systematics: fit_func must return dict or scalar")

    deltas = {}
    for raw_key, value in shifts.items():
        if raw_key == "tail_fraction":
            key = "tail"
            if key not in priors:
                raise RuntimeError("No prior for 'tail'")
            delta = priors[key][0] * float(value)
        elif raw_key.endswith("_frac"):
            key = raw_key[:-5]
            if key not in priors:
                raise RuntimeError(f"No prior for '{key}'")
            delta = priors[key][0] * float(value)
        elif raw_key.endswith("_keV"):
            key = raw_key[:-4]
            if key not in priors:
                raise RuntimeError(f"No prior for '{key}'")
            delta = float(value) / 1000.0
        else:
            key = raw_key
            if key not in priors:
                raise RuntimeError(f"No prior for '{key}'")
            delta = float(value)

        central_val = central[key] if is_dict else central

        p_plus = priors.copy()
        mu, sig = p_plus[key]
        p_plus[key] = (mu + delta, sig)
        res_plus = fit_func(p_plus)
        v_plus = res_plus[key] if is_dict else res_plus

        p_minus = priors.copy()
        p_minus[key] = (mu - delta, sig)
        res_minus = fit_func(p_minus)
        v_minus = res_minus[key] if is_dict else res_minus

        deltas[key] = max(abs(v_plus - central_val), abs(v_minus - central_val))

    # Combine all systematic shifts in quadrature using only the values
    total_unc = math.sqrt(sum(v ** 2 for v in deltas.values()))
    return deltas, total_unc
