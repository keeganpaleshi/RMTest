# utils.py

import numpy as np
from scipy.signal import find_peaks, find_peaks_cwt
import math
from dataclasses import is_dataclass, asdict
import argparse
import sys
from datetime import datetime, timezone, timedelta
from .time_utils import parse_timestamp as _parse_timestamp, to_epoch_seconds, to_utc_datetime as _to_utc_datetime
import warnings

__all__ = [
    "to_native",
    "find_adc_bin_peaks",
    "adc_hist_edges",
    "rebin_histogram",
    "fd_rebin_factor",
    "cps_to_cpd",
    "cps_to_bq",
    "to_utc_datetime",
    "parse_time_arg",
    "to_epoch_seconds",
    "parse_datetime",
    "to_seconds",
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


def find_adc_bin_peaks(
    adc_values,
    expected,
    window=50,
    prominence=0.0,
    width=None,
    *,
    method="prominence",
    cwt_widths=None,
):
    """Locate peak centroids in an ADC array.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values (integers).
    expected : dict
        Mapping of peak name -> expected ADC centroid.
    window : int, optional
        Search window around each expected centroid (+/- window).
    prominence : float, optional
        Minimum prominence passed to :func:`scipy.signal.find_peaks`.
        Ignored when ``method`` is ``"cwt"``.
    width : float or None, optional
        Minimum peak width passed to :func:`scipy.signal.find_peaks`.
        Ignored when ``method`` is ``"cwt"``.
    method : {"prominence", "cwt"}, optional
        Peak detection algorithm. ``"prominence"`` uses
        :func:`scipy.signal.find_peaks` with the ``prominence`` and ``width``
        thresholds. ``"cwt"`` applies
        :func:`scipy.signal.find_peaks_cwt` with ``cwt_widths`` to search across
        multiple scales.
    cwt_widths : sequence of float, optional
        Widths passed to :func:`scipy.signal.find_peaks_cwt` when ``method`` is
        ``"cwt"``. Defaults to ``np.arange(1, 10)``.

    Returns
    -------
    dict
        {peak_name: adc_centroid}
    """
    adc_arr = np.asarray(adc_values, dtype=float)
    if adc_arr.size == 0:
        return {k: float(v) for k, v in expected.items()}

    # Build a channel-centered histogram so discrete ADC counts stay aligned.
    edges = adc_hist_edges(adc_arr, channel_width=1.0)
    hist, _ = np.histogram(adc_arr, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # Global peak search with optional thresholds
    if method not in {"prominence", "cwt"}:
        raise ValueError("Unsupported method: %s" % method)
    if method == "prominence":
        peak_indices, _ = find_peaks(hist, prominence=prominence, width=width)
    else:
        if cwt_widths is None:
            cwt_widths = np.arange(1, 10)
        peak_indices = find_peaks_cwt(hist, cwt_widths)

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


def adc_hist_edges(adc_values, hist_bins=None, *, channel_width=None):
    """Return histogram bin edges for raw ADC values.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values used to define the histogram range.
    hist_bins : int or None, optional
        Number of bins to divide the range into. When ``None`` each
        ADC channel becomes its own bin.
    channel_width : float or None, optional
        Width of grouped ADC bins. When given, the edges stay centered on the
        integer ADC channels.

    Returns
    -------
    np.ndarray
        Array of bin edges suitable for :func:`numpy.histogram`.
    """

    adc_arr = np.asarray(adc_values, dtype=float)
    if adc_arr.size == 0:
        return np.asarray([0.0, 1.0], dtype=float)

    finite_adc = adc_arr[np.isfinite(adc_arr)]
    if finite_adc.size == 0:
        return np.asarray([0.0, 1.0], dtype=float)

    min_adc = int(np.floor(np.min(finite_adc)))
    max_adc = int(np.ceil(np.max(finite_adc)))

    if channel_width is not None:
        if hist_bins is not None:
            raise ValueError("Specify either hist_bins or channel_width, not both")
        width = float(channel_width)
        if not np.isfinite(width) or width <= 0.0:
            raise ValueError("channel_width must be positive")
        n_bins = max(1, int(np.ceil((max_adc - min_adc + 1) / width)))
        start = min_adc - 0.5
        return start + width * np.arange(n_bins + 1, dtype=float)

    if hist_bins is None:
        edges = np.arange(min_adc - 0.5, max_adc + 1.5, dtype=float)
    else:
        edges = np.linspace(min_adc - 0.5, max_adc + 0.5, int(hist_bins) + 1)

    return edges


def fd_rebin_factor(adc_values, cal_slope, n_full_bins=None):
    """Compute optimal rebin factor for ADC channels.

    For spectral data (peaked distributions), the standard Freedman-Diaconis
    IQR-based rule gives unreliable results because the IQR is dominated by
    the narrowest alpha peak.  Instead, this function uses the Freedman-Diaconis
    rule applied to the *occupied histogram bins* (treating each bin as one
    observation weighted by its content).  This gives a rebin factor that
    produces a manageable number of bins (~100-400) suitable for spectral
    fitting.

    Falls back to Scott's rule with the full ADC range when the IQR-based
    estimate gives factor <= 1.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values.
    cal_slope : float
        Calibration slope in MeV per ADC channel.
    n_full_bins : int, optional
        Number of full-resolution bins.  If provided, used for the
        range-based fallback.

    Returns
    -------
    int
        Optimal rebin factor (number of ADC channels per bin).
    """
    adc = np.asarray(adc_values, dtype=float)
    adc = adc[np.isfinite(adc)]
    n = adc.size
    if n < 4:
        return 1

    # --- Primary: FD rule on the raw events ---
    energies = adc * abs(cal_slope)
    q25, q75 = np.percentile(energies, [25, 75])
    iqr = q75 - q25
    if iqr > 0:
        h = 2.0 * iqr / (n ** (1.0 / 3.0))
        k_iqr = max(1, round(h / abs(cal_slope)))
    else:
        k_iqr = 1

    if k_iqr > 1:
        return int(k_iqr)

    # --- Fallback: range-based Scott's rule ---
    # For peaked spectral data, IQR is tiny.  Use the full ADC range
    # to compute a reasonable rebin factor targeting ~200 final bins.
    adc_range = float(adc.max() - adc.min())
    if adc_range <= 0:
        return 1
    if n_full_bins is not None and n_full_bins > 0:
        # Target ~200 bins (empirically good for spectral fitting)
        k_range = max(1, round(n_full_bins / 200))
    else:
        # Scott's rule with std: h = 3.49 * sigma / n^(1/3)
        sigma_adc = float(np.std(adc))
        if sigma_adc > 0:
            h_scott = 3.49 * sigma_adc * abs(cal_slope) / (n ** (1.0 / 3.0))
            k_range = max(1, round(h_scott / abs(cal_slope)))
        else:
            k_range = max(1, round(adc_range / 200))
    return int(k_range)


def rebin_histogram(counts, edges, rebin_factor):
    """Rebin a histogram by grouping consecutive bins.

    Parameters
    ----------
    counts : array-like
        Bin counts (may be non-integer after DNL correction).
    edges : array-like
        Bin edges (length = len(counts) + 1).
    rebin_factor : int
        Number of original bins to merge into each new bin.

    Returns
    -------
    rebinned_counts : np.ndarray
    rebinned_edges : np.ndarray
    """
    counts = np.asarray(counts, dtype=float)
    edges = np.asarray(edges, dtype=float)
    k = max(1, int(rebin_factor))
    n = counts.size
    n_full = (n // k) * k
    remainder = n - n_full

    # Full groups
    rebinned = counts[:n_full].reshape(-1, k).sum(axis=1)
    new_edges = edges[::k][:len(rebinned)]
    # Append the right edge of the last full group
    new_edges = np.append(new_edges, edges[n_full])

    # Partial last bin if remainder > 0
    if remainder > 0:
        partial = counts[n_full:].sum()
        rebinned = np.append(rebinned, partial)
        new_edges = np.append(new_edges, edges[-1])

    return rebinned, new_edges


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




def to_utc_datetime(value, tz="UTC") -> datetime:
    """Return ``value`` converted to a UTC ``datetime`` object."""

    return _to_utc_datetime(value, tz=tz)


def parse_datetime(value):
    """Deprecated alias for :func:`utils.time_utils.parse_timestamp`."""

    warnings.warn(
        "parse_datetime is deprecated; use utils.time_utils.parse_timestamp",
        DeprecationWarning,
        stacklevel=2,
    )
    return _parse_timestamp(value)


def _dt_to_epoch_seconds(dt: pd.Series) -> np.ndarray:
    """Convert a UTC datetime Series to float epoch seconds.

    Works with both pandas 1.x (.view) and pandas 2.x (.astype).
    """
    try:
        return (dt.view("int64") / 1e9).astype(float)
    except AttributeError:
        # pandas >= 2.0: .view removed; resolution may vary
        res = str(dt.dtype)
        if "[us" in res:
            divisor = 1e6
        elif "[ms" in res:
            divisor = 1e3
        elif "[ns" in res:
            divisor = 1e9
        else:
            divisor = 1.0
        return (dt.astype("int64") / divisor).astype(float)


def to_seconds(series: pd.Series) -> np.ndarray:
    """Return float seconds from a timestamp series.

    Behavior:
        - numeric input -> pass-through as float
        - datetime-like -> epoch seconds (UTC)
        - string/object -> try datetime parse; if parse fails, numeric coercion
    """
    from pandas.api import types as pdt

    # 1) Strict numeric -> identity (float)
    if pdt.is_numeric_dtype(series):
        return series.to_numpy(dtype=float)

    # 2) Already datetime64 -> ensure UTC then convert to epoch seconds
    if pdt.is_datetime64_any_dtype(series):
        dt = series.dt.tz_convert("UTC") if series.dt.tz is not None else series.dt.tz_localize("UTC")
        return _dt_to_epoch_seconds(dt)

    # 3) Object/strings: try datetime parse first; if it fails, fall back to numeric
    dt = pd.to_datetime(series, errors="coerce", utc=True)
    if dt.notna().any():
        return _dt_to_epoch_seconds(dt)
    return pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)


def parse_timestamp(value):
    """Deprecated wrapper for :func:`utils.time_utils.parse_timestamp`."""

    warnings.warn(
        "utils.parse_timestamp is deprecated; use utils.time_utils.parse_timestamp",
        DeprecationWarning,
        stacklevel=2,
    )
    return _parse_timestamp(value)


def parse_time(s, tz="UTC") -> float:
    """Deprecated wrapper for :func:`utils.time_utils.to_epoch_seconds`."""

    warnings.warn(
        "utils.parse_time is deprecated; use utils.time_utils.to_epoch_seconds",
        DeprecationWarning,
        stacklevel=2,
    )
    return to_epoch_seconds(s, tz=tz)


def parse_time_arg(val, tz="UTC") -> datetime:
    """Parse a time argument into a UTC ``datetime`` object.

    ``tz`` specifies the timezone for naive inputs.
    """

    try:
        return to_utc_datetime(val, tz=tz)
    except ValueError as e:
        raise argparse.ArgumentTypeError(str(e)) from e


def build_cli_parser() -> argparse.ArgumentParser:
    """Build the lightweight conversion CLI parser."""

    parser = argparse.ArgumentParser(
        description="Convert count rates between cps, cpd, and Bq.",
        epilog=(
            "Hyphenated long flags are canonical. Deprecated underscore aliases "
            "remain accepted for compatibility."
        ),
    )
    parser.add_argument("rate_cps", type=float, help="Input count rate in counts per second.")
    parser.add_argument(
        "--to",
        choices=["cpd", "bq"],
        required=True,
        help="Requested output unit.",
    )
    parser.add_argument(
        "--volume-liters",
        dest="volume_liters",
        type=float,
        help="Detector volume in liters. Required when converting to Bq.",
    )
    parser.add_argument(
        "--volume_liters",
        dest="volume_liters",
        type=float,
        help=argparse.SUPPRESS,
        default=argparse.SUPPRESS,
    )
    return parser



def parse_cli_args(argv=None):
    """Parse arguments for the lightweight conversion CLI."""

    argv_list = list(sys.argv[1:] if argv is None else argv)
    for token in argv_list:
        if token.split("=", 1)[0] == "--volume_liters":
            warnings.warn(
                "--volume_liters is deprecated; use --volume-liters",
                DeprecationWarning,
                stacklevel=2,
            )
            break
    return build_cli_parser().parse_args(argv_list)



def main(argv=None):
    """Run the lightweight conversion CLI."""

    args = parse_cli_args(argv)

    if args.to == "cpd":
        print(cps_to_cpd(args.rate_cps))
    else:
        print(cps_to_bq(args.rate_cps, args.volume_liters))


if __name__ == "__main__":
    main()
