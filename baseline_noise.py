import numpy as np
from scipy.optimize import curve_fit

__all__ = ["estimate_baseline_noise"]


def _constant(x, A):
    return A * np.ones_like(x)


def _exponential(x, A, k):
    return A * np.exp(-k * x)


def estimate_baseline_noise(adc_values, peak_adc=None, nbins=50, model="constant"):
    """Estimate electronic noise level from baseline ADC values.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values from the baseline period.
    peak_adc : float, optional
        Location of the Po-210 peak. Values above this are ignored.
    nbins : int, optional
        Number of histogram bins.
    model : {"constant", "exponential"}
        Functional form used for the fit.

    Returns
    -------
    noise_level : float
        Fitted noise level (constant or amplitude of exponential).
    params : dict
        Additional fitted parameters.
    """
    adc = np.asarray(adc_values, dtype=float)
    if peak_adc is not None:
        adc = adc[adc < peak_adc]
    if adc.size == 0:
        return 0.0, {}

    hist, edges = np.histogram(adc, bins=nbins)
    centers = 0.5 * (edges[:-1] + edges[1:])

    if model == "constant":
        A = float(np.mean(hist))
        return A, {"A": A}

    if model == "exponential":
        p0 = [hist.max(), 0.001]
        try:
            popt, _ = curve_fit(_exponential, centers, hist, p0=p0, maxfev=10000)
            A, k = popt
            return float(A), {"A": float(A), "k": float(k)}
        except Exception:
            A = float(np.mean(hist))
            return A, {"A": A}

    raise ValueError("Unsupported model: %s" % model)
