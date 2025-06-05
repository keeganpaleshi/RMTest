import math
from typing import Callable, Dict, Tuple


def scan_systematics(
    fit_func: Callable,
    priors: Dict[str, Tuple[float, float]],
    sigma_dict: Dict[str, float],
    keys: [str],
) -> Tuple[Dict[str, float], float]:
    """Scan parameter level systematics as per the imported implementation."""
    central = fit_func(priors)
    is_dict = isinstance(central, dict)
    if not is_dict and not isinstance(central, (int, float)):
        raise RuntimeError(
            "scan_systematics: fit_func must return dict or scalar")

    deltas = {}
    for key in keys:
        if is_dict:
            if key not in central:
                raise RuntimeError(f"Central fit missing parameter '{key}'")
            v0 = central[key]
        else:
            v0 = central

        if key not in sigma_dict:
            raise RuntimeError(f"No sigma_dict entry for '{key}'")
        delta = sigma_dict[key]

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

    # Combine all systematic shifts in quadrature using only the values
    total_unc = math.sqrt(sum(v ** 2 for v in deltas.values()))
    return deltas, total_unc
