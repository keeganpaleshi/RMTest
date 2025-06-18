import numpy as np
from scipy.optimize import curve_fit
from constants import CURVE_FIT_MAX_EVALS

__all__ = ["estimate_baseline_noise"]


def _constant(x, A):
    return A * np.ones_like(x)


def _exponential(x, A, k):
    A = np.minimum(A, 1e300)
    return A * np.exp(-k * x)


def estimate_baseline_noise(
    adc_values,
    peak_adc=None,
    pedestal_cut=None,
    nbins=50,
    model="constant",
    return_mask=False,
):
    """Estimate electronic noise level from baseline ADC values.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values from the baseline period.
    peak_adc : float, optional
        Location of the Po-210 peak. Values above this are ignored.
    pedestal_cut : float, optional
        Minimum ADC value to include in the fit. Values below this are
        ignored.
    nbins : int, optional
        Number of histogram bins.
    model : {"constant", "exponential"}
        Functional form used for the fit.

    Returns
    -------
    noise_level : float
        Estimated amplitude of the noise peak in counts (histogram
        height). This is a count estimate, not an ADC value.
    params : dict
        Additional fitted parameters.
    mask : ndarray of bool, optional
        Boolean mask selecting the ADC values used for the fit. Only returned
        when ``return_mask`` is ``True``.
    """
    adc_arr = np.asarray(adc_values, dtype=float)
    mask = np.ones_like(adc_arr, dtype=bool)
    if pedestal_cut is not None:
        mask &= adc_arr > pedestal_cut
    if peak_adc is not None:
        mask &= adc_arr < peak_adc
    adc = adc_arr[mask]
    if adc.size == 0:
        if return_mask:
            return 0.0, {}, mask
        return 0.0, {}

    hist, edges = np.histogram(adc, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if model == "constant":
        A = float(np.mean(hist))
        A = min(A, 1e300)
        if return_mask:
            return A, {"A": A}, mask
        return A, {"A": A}

    if model == "exponential":
        p0 = [hist.max(), 0.001]
        try:
            popt, _ = curve_fit(
                _exponential, centers, hist, p0=p0, maxfev=CURVE_FIT_MAX_EVALS
            )
            A, k = popt
            A = min(float(A), 1e300)
            if return_mask:
                return A, {"A": A, "k": float(k)}, mask
            return A, {"A": A, "k": float(k)}
        except Exception:
            A = float(np.mean(hist))
            A = min(A, 1e300)
            if return_mask:
                return A, {"A": A}, mask
            return A, {"A": A}
    raise ValueError("Unsupported model: %s" % model)
