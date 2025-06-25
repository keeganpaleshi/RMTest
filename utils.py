# utils.py

import numpy as np
from scipy.signal import find_peaks
import math
from dataclasses import is_dataclass, asdict
import argparse
from datetime import datetime, timezone, tzinfo, timedelta
from dateutil import parser as date_parser
from dateutil.tz import gettz

__all__ = [
    "to_native",
    "find_adc_bin_peaks",
    "adc_hist_edges",
    "cps_to_cpd",
    "cps_to_bq",
    "to_utc_datetime",
    "parse_time_arg",
    "parse_timestamp",
    "parse_datetime",
    "parse_time",
    "LITERS_PER_M3",
]

# Conversion factor from cubic meters to liters
LITERS_PER_M3 = 1000.0

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
    if isinstance(obj, datetime):
        if obj.tzinfo is not None and obj.tzinfo.utcoffset(obj) == timedelta(0):
            return obj.strftime("%Y-%m-%dT%H:%M:%SZ")
        return obj.isoformat()
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


def adc_hist_edges(adc_values, hist_bins=None):
    """Return histogram bin edges for raw ADC values.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values used to define the histogram range.
    hist_bins : int or None, optional
        Number of bins to divide the range into. When ``None`` each
        ADC channel becomes its own bin.

    Returns
    -------
    np.ndarray
        Array of bin edges suitable for :func:`numpy.histogram`.
    """

    adc_arr = np.asarray(adc_values, dtype=float)
    if adc_arr.size == 0:
        return np.asarray([0.0, 1.0], dtype=float)

    min_adc = int(np.min(adc_arr))
    max_adc = int(np.max(adc_arr))

    if hist_bins is None:
        edges = np.arange(min_adc, max_adc + 2)
    else:
        edges = np.linspace(min_adc, max_adc, int(hist_bins) + 1)

    return edges


def cps_to_cpd(rate_cps):
    """Convert counts/s to counts/day."""
    return rate_cps * 86400.0


def cps_to_bq(rate_cps, volume_liters=None):
    """Convert counts/s to activity in Bq.

    If ``volume_liters`` is provided, the result is returned as Bq/m^3
    assuming ``volume_liters`` describes the detector volume. Conversion
    from liters to cubic meters uses :data:`LITERS_PER_M3`.
    """

    if volume_liters is None:
        return float(rate_cps)

    # Convert liters -> m^3 using the LITERS_PER_M3 factor
    volume_m3 = volume_liters / LITERS_PER_M3
    if volume_m3 <= 0:
        raise ValueError("volume_liters must be positive")
    return float(rate_cps) / volume_m3


def parse_timestamp(s) -> float:
    """Parse an ISO-8601 string, numeric seconds, or ``datetime``.

    Any string without timezone information is interpreted as UTC.
    The return value is the Unix epoch time in seconds (UTC).
    """

    if isinstance(s, (int, float)):
        return float(s)

    if isinstance(s, datetime):
        dt = s
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return float(dt.timestamp())

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
        else:
            dt = dt.astimezone(timezone.utc)

        return float(dt.timestamp())

    raise argparse.ArgumentTypeError(f"could not parse time: {s!r}")


def to_utc_datetime(value, tz="UTC") -> datetime:
    """Return ``value`` converted to a UTC ``datetime`` object."""

    tzinfo_obj = tz if isinstance(tz, tzinfo) else gettz(tz)
    if tzinfo_obj is None:
        tzinfo_obj = timezone.utc

    if isinstance(value, datetime):
        dt = value
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=tzinfo_obj)
    elif isinstance(value, str):
        try:
            ts = float(value)
            return datetime.fromtimestamp(ts, tz=timezone.utc)
        except ValueError:
            try:
                dt = date_parser.isoparse(value)
            except (ValueError, OverflowError) as e:
                raise ValueError(f"invalid datetime: {value!r}") from e
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=tzinfo_obj)
    elif isinstance(value, (int, float)):
        dt = datetime.fromtimestamp(float(value), tz=timezone.utc)
    else:
        raise ValueError(f"invalid datetime: {value!r}")

    return dt.astimezone(timezone.utc)


def parse_datetime(value):
    """Parse an ISO-8601 string or numeric epoch value to a timezone-aware
    ``Timestamp`` in UTC.

    The function accepts strings like ``"2023-09-28T13:45:00-04:00"`` or
    numeric Unix timestamps (as ``int``, ``float`` or numeric ``str``). Any
    parsed time lacking a timezone is interpreted as UTC. On success a
    ``pandas.Timestamp`` object with ``UTC`` timezone information is returned.
    ``ValueError`` is raised if the input cannot be parsed.
    """

    try:
        dt = to_utc_datetime(value)
    except ValueError as e:
        raise ValueError(f"invalid datetime: {value!r}") from e

    if pd is None:
        raise RuntimeError("pandas is required for parse_datetime")

    ts = pd.Timestamp(dt)
    if ts.tz is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")

    try:
        # pandas >= 2.2 with NumPy >= 2.0 can retain the timezone
        return ts.to_datetime64("UTC")
    except TypeError:
        # Older pandas versions drop the timezone when converting to
        # ``datetime64``.  In that case return the Timestamp itself.
        return ts


def parse_time(s, tz="UTC") -> float:
    """Parse a timestamp string, number or ``datetime`` into Unix seconds."""

    return parse_timestamp(s)


def parse_time_arg(val, tz="UTC") -> datetime:
    """Parse a time argument into a UTC ``datetime`` object.

    ``tz`` specifies the timezone for naïve inputs.
    """

    try:
        return to_utc_datetime(val, tz=tz)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


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
