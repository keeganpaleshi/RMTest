# utils.py

import numpy as np
from scipy.signal import find_peaks

__all__ = [
    "to_native",
    "find_adc_peaks",
]


def to_native(obj):
    """
    Recursively convert NumPy scalar types to native Python types
    (int, float) so that JSON serialization works.
    """
    if isinstance(obj, dict):
        return {to_native(k): to_native(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_native(x) for x in obj]
    elif isinstance(obj, np.generic):
        return obj.item()
    elif isinstance(obj, np.ndarray):
        # Convert array into list of native types
        return [to_native(x) for x in obj.tolist()]
    else:
        return obj


def find_adc_peaks(adc_values, expected, window=50, noise_cutoff=0, hist_bins=1000):
    """Simple peak search around expected ADC values.

    Parameters
    ----------
    adc_values : array-like
        Array of ADC values.
    expected : dict
        Mapping of peak name -> approximate ADC location.
    window : int, optional
        Half-width of the search window in ADC channels.
    noise_cutoff : int, optional
        Minimum height in the histogram to consider a peak valid.
    hist_bins : int, optional
        Number of histogram bins to use when scanning for peaks.

    Returns
    -------
    dict
        Dictionary of peak name -> estimated centroid ADC value.  If no peak is
        found in the window, the expected value is returned.
    """

    adc_arr = np.asarray(adc_values, dtype=float)
    if adc_arr.size == 0:
        return {k: float(v) for k, v in expected.items()}

    hist, edges = np.histogram(adc_arr, bins=hist_bins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    results = {}
    for name, guess in expected.items():
        lo = guess - window
        hi = guess + window
        mask = (centers >= lo) & (centers <= hi)
        if not np.any(mask):
            results[name] = float(guess)
            continue

        sub_hist = hist[mask]
        sub_centers = centers[mask]

        if np.all(sub_hist < noise_cutoff):
            # No significant peak; just use mean within window
            results[name] = float(np.mean(adc_arr[(adc_arr >= lo) & (adc_arr <= hi)]))
            continue

        # Use scipy.signal.find_peaks within the window
        peak_idx, _ = find_peaks(sub_hist)
        if len(peak_idx) == 0:
            idx = int(np.argmax(sub_hist))
        else:
            idx = peak_idx[np.argmax(sub_hist[peak_idx])]

        results[name] = float(sub_centers[idx])

    return results

