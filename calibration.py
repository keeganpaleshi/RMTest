import numpy as np
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import exponnorm

# Known α energies (MeV) from config or central constants:
# Default alpha energies (MeV) used for calibration when not specified
# in the configuration file.  Values are Po-210, Po-218 and Po-214
DEFAULT_KNOWN_ENERGIES = {
    "Po210": 5.304,   # MeV
    "Po218": 6.002,   # MeV
    "Po214": 7.687    # MeV  (use the SNOLAB‐observed centroids if desired)
}


def emg_left(x, mu, sigma, tau):
    """Exponentially modified Gaussian (left-skewed) PDF."""

    if tau <= 0:
        return gaussian(x, mu, sigma)

    # SciPy's `exponnorm` uses shape parameter K = tau / sigma
    K = tau / sigma
    return exponnorm.pdf(x, K, loc=mu, scale=sigma)


def gaussian(x, mu, sigma):
    """Standard Gaussian PDF (unit area)."""
    return np.exp(-0.5 * ((x - mu) / sigma) ** 2) / (sigma * np.sqrt(2 * np.pi))


def two_point_calibration(adc_centroids, energies):
    """
    Given two reference points (adc_centroids = [adc1, adc2], energies = [E1, E2]),
    solve for slope a and intercept c:  E = a * ADC + c.
    Returns (a, c).
    """
    x1, x2 = adc_centroids
    E1, E2 = energies
    a = (E2 - E1) / (x2 - x1)
    c = E1 - a * x1
    return float(a), float(c)


def apply_calibration(adc_values, slope, intercept):
    """Convert ADC values to energy using slope and intercept."""
    adc_arr = np.asarray(adc_values, dtype=float)
    return slope * adc_arr + intercept


def calibrate_run(adc_values, config, hist_bins=None):
    """
    Main entry to derive calibration constants for a single run:
      - Build histogram of ADC values.
        By default each ADC channel is its own bin.
        If ``hist_bins`` is provided the range is divided into that
        many bins instead.
      - Identify approximate peak locations (Po-210, Po-218, Po-214).
      - Fit each peak with (EMG or Gaussian) depending on config flags.
      - Compute two-point linear calibration (using Po-210 & Po-214).
      - Return a dict with {a, c, sigma_E, peak_centroids_ADC, peak_sigmas_ADC, tau_ADC (if any)}.
    """
    # 1) Build histogram
    min_adc = int(np.min(adc_values))
    max_adc = int(np.max(adc_values))

    if hist_bins is None:
        # 1 ADC channel per bin (backwards compatible)
        edges = np.arange(min_adc, max_adc + 2)
    else:
        edges = np.linspace(min_adc, max_adc, hist_bins + 1)

    hist, edges = np.histogram(adc_values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])  # effectively integer centers

    # 2) Peak‐finding with SciPy:
    #    We look for up to 5 peaks (Po-210, Po-218, Po-214, maybe background bumps).
    #    Use config thresholds for prominence & width.
    prom = config["calibration"]["peak_prominence"]
    wid = config["calibration"]["peak_width"]
    peaks, props = find_peaks(hist, prominence=prom, width=wid)

    # 3) From the found peaks, pick the three that best match expected energies:
    #    Convert expected energy->ADC guess via last calibration or central ADC->MeV mapping if available.
    #    For the first run, use rough nominal ADC guesses from config:
    # e.g. {"Po210": 800, "Po218": 900, "Po214": 1200}
    nominal_adc = config["calibration"]["nominal_adc"]

    # Build a dictionary: { isotope : list of candidate indices within +/- radius }
    # radius (in ADC channels) to match found peaks to nominal guesses
    # 2024-03: renamed from ``peak_search_radius_adc`` to ``peak_search_radius``
    radius = config["calibration"]["peak_search_radius"]
    candidates = {iso: [] for iso in ("Po210", "Po218", "Po214")}
    for iso, adc_guess in nominal_adc.items():
        for idx in peaks:
            if abs(centers[idx] - adc_guess) <= radius:
                candidates[iso].append(idx)
    # If multiple candidates per isotope, pick the one with highest histogram count:
    chosen_idx = {}
    for iso in candidates:
        if not candidates[iso]:
            raise RuntimeError(
                f"No candidate peak found for {iso} around ADC={nominal_adc[iso]}."
            )
        # pick the one with max(hist) among candidates
        best = max(candidates[iso], key=lambda i: hist[i])
        chosen_idx[iso] = best

    # 4) For each chosen peak, perform a local fit with EMG or Gaussian:
    peak_fits = {}
    window = config["calibration"]["fit_window_adc"]  # e.g. ±50 ADC channels
    use_emg = config["calibration"]["use_emg"]        # True/False
    for iso, idx_center in chosen_idx.items():
        x0 = centers[idx_center]
        lo = x0 - window
        hi = x0 + window
        # Extract local slice:
        mask = (centers >= lo) & (centers <= hi)
        x_slice = centers[mask]
        y_slice = hist[mask].astype(float)
        if len(x_slice) < 5:
            raise RuntimeError(
                f"Not enough points to fit peak for {iso} (only {len(x_slice)} bins)."
            )

        # Initial guesses:
        amp0 = float(np.max(y_slice))
        mu0 = float(x0)
        # e.g. ~10 ADC channels
        sigma0 = config["calibration"]["init_sigma_adc"]
        tau_cfg = config["calibration"].get("init_tau_adc", 0.0)
        # Avoid zero or negative starting tau which can cause numerical issues
        tau0 = max(tau_cfg, 1e-6) if use_emg else 0.0

        if use_emg and iso in ("Po210", "Po218"):
            # Fit EMG: parameters [amp, mu, sigma, tau]
            def model_emg(x, A, mu, sigma, tau):
                return A * emg_left(x, mu, sigma, tau)
            p0 = [amp0, mu0, sigma0, tau0]
            bounds = (
                [0, mu0 - window, 1e-3, 0],         # lower bounds
                [np.inf, mu0 + window, 50.0, 200.0]  # upper bounds (tunable)
            )
            popt, pcov = curve_fit(
                model_emg, x_slice, y_slice, p0=p0, bounds=bounds)
            A_fit, mu_fit, sigma_fit, tau_fit = popt
            peak_fits[iso] = {
                "centroid_adc": float(mu_fit),
                "sigma_adc": float(sigma_fit),
                "tau_adc": float(tau_fit),
                "amplitude": float(A_fit),
                "covariance": pcov.tolist()
            }
        else:
            # Fit pure Gaussian: [A, mu, sigma]
            def model_gauss(x, A, mu, sigma):
                return A * gaussian(x, mu, sigma)
            p0 = [amp0, mu0, sigma0]
            bounds = ([0, mu0 - window, 1e-3], [np.inf, mu0 + window, 50.0])
            popt, pcov = curve_fit(model_gauss, x_slice,
                                   y_slice, p0=p0, bounds=bounds)
            A_fit, mu_fit, sigma_fit = popt
            peak_fits[iso] = {
                "centroid_adc": float(mu_fit),
                "sigma_adc": float(sigma_fit),
                "tau_adc": 0.0,
                "amplitude": float(A_fit),
                "covariance": pcov.tolist()
            }

    # 5) Two-point linear calibration using Po-210 & Po-214:
    adc210 = peak_fits["Po210"]["centroid_adc"]
    adc214 = peak_fits["Po214"]["centroid_adc"]

    cfg_energies = config.get("calibration", {}).get("known_energies")
    energies = DEFAULT_KNOWN_ENERGIES if cfg_energies is None else {
        **DEFAULT_KNOWN_ENERGIES,
        **cfg_energies,
    }

    E210 = energies["Po210"]
    E214 = energies["Po214"]
    a, c = two_point_calibration([adc210, adc214], [E210, E214])

    # 6) Convert σADC -> σE (MeV) by σE = a * σADC.  For simplicity, we ignore error propagation of slope/intercept here.
    # use Po-214 width as representative
    sigma_E = abs(a) * (peak_fits["Po214"]["sigma_adc"])

    # 7) Build result dict:
    calib_dict = {
        "slope": a,
        "intercept": c,
        "sigma_E": float(sigma_E),
        "peaks": peak_fits
    }
    return calib_dict


def derive_calibration_constants(adc_values, config):
    """Wrapper returning calibration constants in legacy format."""
    res = calibrate_run(adc_values, config)
    out = {
        "a": (float(res["slope"]), 0.0),
        "c": (float(res["intercept"]), 0.0),
        "sigma_E": (float(res["sigma_E"]), 0.0),
        "peaks": res.get("peaks", {}),
    }
    return out


def derive_calibration_constants_auto(
    adc_values,
    noise_cutoff=300,
    hist_bins=2000,
    peak_search_radius=200,
    nominal_adc=None,
):
    """Simple auto-calibration using default configuration.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values.
    noise_cutoff : int
        Minimum ADC to consider when building histogram.
    hist_bins : int
        Number of histogram bins.
    peak_search_radius : int
        Window around nominal peaks used for matching.
    nominal_adc : dict or None
        Optional mapping of isotope -> ADC guess.
        Expected ADC centroids for each isotope.  Keys should be
        ``"Po210"``, ``"Po218"`` and ``"Po214"``.  If ``None`` a
        reasonable default of ``{"Po210": 1250, "Po218": 1400, "Po214": 1800}``
        is used.

    """
    if len(adc_values) == 0:
        raise RuntimeError("No ADC values provided")

    adc_arr = np.asarray(adc_values, dtype=float)
    mask = adc_arr >= noise_cutoff
    adc_arr = adc_arr[mask]

    if nominal_adc is None:
        nominal_adc = {"Po210": 1250, "Po218": 1400, "Po214": 1800}

    config = {
        "calibration": {
            "peak_prominence": 10,
            "peak_width": 3,
            "nominal_adc": nominal_adc,
            # parameter renamed to ``peak_search_radius`` to match config.json
            "peak_search_radius": peak_search_radius,
            "fit_window_adc": 50,
            "use_emg": False,
            "init_sigma_adc": 10.0,
            "init_tau_adc": 1.0,
        }
    }

    if nominal_adc is not None:
        config["calibration"]["nominal_adc"] = dict(nominal_adc)

    # Run calibration with custom histogram binning and convert to legacy format
    res = calibrate_run(adc_arr, config, hist_bins=hist_bins)
    out = {
        "a": (float(res["slope"]), 0.0),
        "c": (float(res["intercept"]), 0.0),
        "sigma_E": (float(res["sigma_E"]), 0.0),
        "peaks": res.get("peaks", {}),
    }
    return out


__all__ = [
    "two_point_calibration",
    "apply_calibration",
    "calibrate_run",
    "derive_calibration_constants",
    "derive_calibration_constants_auto",
]
