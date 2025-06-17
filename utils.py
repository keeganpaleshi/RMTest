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


def parse_time(s: str) -> int:
    """Parse a timestamp string or integer into Unix epoch seconds."""
    try:
        return int(s)
    except ValueError:
        pass

    try:
        dt = date_parser.parse(s, dayfirst=False)
    except (ValueError, OverflowError) as e:
        raise argparse.ArgumentTypeError(f"could not parse time: {s!r}") from e

    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)

    return int(dt.timestamp())


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
