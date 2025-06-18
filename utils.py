# utils.py

import numpy as np
from scipy.signal import find_peaks
import math
from dataclasses import is_dataclass, asdict
import argparse
from datetime import datetime, timezone
from dateutil import parser as date_parser

__all__ = ["to_native", "find_adc_bin_peaks", "cps_to_cpd", "cps_to_bq", "parse_time"]

try:
    import pandas as pd
except Exception:  # pragma: no cover - pandas is optional for utils
    pd = None


def to_native(obj):
    """
    Recursively convert NumPy scalar types to native Python types
    (int, float) so that JSON serialization works.
    """
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(x) for x in obj]
    if is_dataclass(obj):
        return to_native(asdict(obj))
    if pd is not None:
        # Handle pandas scalar types
        if obj is pd.NA:
            return None
        elif isinstance(obj, (pd.Timestamp, pd.Timedelta)):
            return obj.isoformat()
        elif isinstance(obj, (pd.Series, pd.Index)):
            return [to_native(x) for x in obj.tolist()]
    if isinstance(obj, np.ndarray):
        # Convert array into list of native types
        return [to_native(x) for x in obj.tolist()]

    if isinstance(obj, np.generic):
        obj = obj.item()

    if isinstance(obj, float):
        if math.isnan(obj) or not math.isfinite(obj):
            return None

    return obj


def find_adc_bin_peaks(adc_values, expected, window=50, prominence=0.0, width=None):
    """Locate peak centroids in an ADC array.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values (integers).
    expected : dict
        Mapping of peak name -> expected ADC centroid.
    window : int, optional
        Search window around each expected centroid (± window).
    prominence : float, optional
        Minimum prominence passed to :func:`scipy.signal.find_peaks`.
    width : float or None, optional
        Minimum peak width passed to :func:`scipy.signal.find_peaks`.

    Returns
    -------
    dict
        {peak_name: adc_centroid}
    """
    adc_arr = np.asarray(adc_values, dtype=float)
    if adc_arr.size == 0:
        return {k: float(v) for k, v in expected.items()}

    # Build 1-channel histogram to locate maxima
    min_adc = int(adc_arr.min())
    max_adc = int(adc_arr.max())
    edges = np.arange(min_adc, max_adc + 2)
    hist, _ = np.histogram(adc_arr, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Global peak search with optional thresholds
    peak_indices, _ = find_peaks(hist, prominence=prominence, width=width)

    results = {}
    for name, guess in expected.items():
        lo = guess - window
        hi = guess + window
        mask = (centers >= lo) & (centers <= hi)
        if np.any(mask):
            idx_candidates = np.where(mask)[0]
            candidate_peak_indices = [idx for idx in idx_candidates if idx in peak_indices]
            if not candidate_peak_indices:
                # Default to candidate bin with the highest histogram count
                best_idx = idx_candidates[np.argmax(hist[idx_candidates])]
            else:
                # Choose among detected peaks with the highest count
                best_idx = candidate_peak_indices[np.argmax(hist[candidate_peak_indices])]

            results[name] = float(centers[best_idx])
        else:
            # No histogram bin in range -> just return the guess
            results[name] = float(guess)

    return results


def cps_to_cpd(rate_cps):
    """Convert counts/s to counts/day."""
    return rate_cps * 86400.0


def cps_to_bq(rate_cps, volume_liters=None):
    """Convert counts/s to activity in Bq.

    If ``volume_liters`` is provided, the result is returned as Bq/m^3
    assuming ``volume_liters`` describes the detector volume.
    """

    if volume_liters is None:
        return float(rate_cps)

    volume_m3 = volume_liters / 1000.0
    if volume_m3 <= 0:
        raise ValueError("volume_liters must be positive")
    return float(rate_cps) / volume_m3


def _fit_params(obj):
    """Return fit parameters dictionary from either a FitResult or mapping."""
    from fitting import FitResult

    if isinstance(obj, FitResult):
        return obj.params
    if isinstance(obj, dict):
        return obj
    return {}


def _cov_entry(fit, p1: str, p2: str) -> float:
    """Return covariance between two parameters from a FitResult."""
    from fitting import FitResult

    if isinstance(fit, FitResult) and fit.cov is not None:
        ordered = [k for k in fit.params.keys() if k != "fit_valid" and not k.startswith("d")]
        try:
            i1 = ordered.index(p1)
            i2 = ordered.index(p2)
        except ValueError:
            return 0.0
        cov = np.asarray(fit.cov, dtype=float)
        if cov.ndim >= 2 and i1 < cov.shape[0] and i2 < cov.shape[1]:
            return float(cov[i1, i2])
    return 0.0


def window_prob(E, sigma, lo, hi):
    """Return probability that each ``E`` lies in [lo, hi]."""

    from scipy.stats import norm

    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    E, sigma = np.broadcast_arrays(E, sigma)
    lo_val = float(lo) if np.isscalar(lo) else float(lo)
    hi_val = float(hi) if np.isscalar(hi) else float(hi)

    prob = np.empty_like(E, dtype=float)
    zero_mask = sigma == 0

    if np.any(zero_mask):
        prob[zero_mask] = ((E[zero_mask] >= lo_val) & (E[zero_mask] <= hi_val)).astype(float)

    if np.any(~zero_mask):
        nz = ~zero_mask
        prob[nz] = norm.cdf(hi_val, loc=E[nz], scale=sigma[nz]) - norm.cdf(
            lo_val, loc=E[nz], scale=sigma[nz]
        )

    if prob.ndim == 0:
        return float(prob)
    return prob


_spike_eff_cache = {}


def get_spike_efficiency(spike_cfg):
    """Return spike efficiency using :func:`calc_spike_efficiency` with caching."""

    key = (
        spike_cfg.get("counts"),
        spike_cfg.get("activity_bq"),
        spike_cfg.get("live_time_s"),
        spike_cfg.get("error"),
    )
    if key not in _spike_eff_cache:
        from efficiency import calc_spike_efficiency

        _spike_eff_cache[key] = calc_spike_efficiency(key[0], key[1], key[2])
    return _spike_eff_cache[key]


def parse_time(s: str) -> int:
    """Parse a timestamp string or integer into Unix epoch seconds."""
    if isinstance(s, (int, float)):
        return float(s)

    if isinstance(s, str):
        try:
            return float(s)
        except ValueError:
            pass

        try:
            dt = date_parser.isoparse(s)
        except (ValueError, OverflowError) as e:
            raise argparse.ArgumentTypeError(f"could not parse time: {s!r}") from e

        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)

        return int(dt.timestamp())

    raise argparse.ArgumentTypeError(f"could not parse time: {s!r}")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Utility conversions")
    p.add_argument("rate_cps", type=float, help="Count rate in cps")
    p.add_argument("--to", choices=["cpd", "bq"], required=True)
    p.add_argument("--volume_liters", type=float, help="Detector volume in liters")
    args = p.parse_args()

    if args.to == "cpd":
        print(cps_to_cpd(args.rate_cps))
    else:
        print(cps_to_bq(args.rate_cps, args.volume_liters))
