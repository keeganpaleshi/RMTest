import numpy as np
from dataclasses import dataclass
from collections.abc import Sequence, Mapping
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import exponnorm
from constants import (
    _TAU_MIN,
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_NOMINAL_ADC,
    DEFAULT_KNOWN_ENERGIES,
    safe_exp as _safe_exp,
)


@dataclass(init=False)
class CalibrationResult:
    """Polynomial calibration coefficients and covariance."""

    coeffs: Sequence[float] | Mapping[int, float]
    cov: np.ndarray | None
    peaks: dict | None = None
    sigma_E: float = 0.0
    sigma_E_error: float = 0.0

    def __init__(
        self,
        coeffs,
        cov=None,
        *,
        covariance=None,
        peaks=None,
        sigma_E=0.0,
        sigma_E_error=0.0,
    ):
        if cov is None:
            cov = covariance
        self.coeffs = coeffs
        self.cov = cov
        self.peaks = peaks
        self.sigma_E = sigma_E
        self.sigma_E_error = sigma_E_error
        self.__post_init__()

    def __post_init__(self):
        if isinstance(self.coeffs, Mapping):
            items = sorted(self.coeffs.items())
            self._exponents = [int(k) for k, _ in items]
            self.coeffs = [float(v) for _, v in items]
        else:
            self._exponents = list(range(len(self.coeffs)))
            self.coeffs = [float(v) for v in self.coeffs]
        self.cov = np.asarray(self.cov, dtype=float)

    def predict(self, x):
        """Return calibrated energies for ``x``."""
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        powers = np.stack([x_arr ** p for p in self._exponents], axis=-1)
        out = powers @ np.asarray(self.coeffs, dtype=float)
        return out[0] if np.ndim(x) == 0 else out

    def apply(self, adc_values):
        """Alias for :py:meth:`predict`."""
        return self.predict(adc_values)

    def uncertainty(self, x):
        """Return propagated 1-sigma energy uncertainty for ``x``."""
        x_arr = np.atleast_1d(np.asarray(x, dtype=float))
        sig2 = []
        for val in x_arr:
            J = np.array([val ** p for p in self._exponents], dtype=float)
            sig2.append(J @ self.cov @ J)
        out = np.sqrt(np.clip(sig2, 0, None))
        return out[0] if np.ndim(x) == 0 else out

def emg_left(x, mu, sigma, tau):
    """Exponentially modified Gaussian (left-skewed) PDF.

    ``exponnorm.pdf`` can overflow for large ``(x - mu)/sigma``.  Use the
    logarithmic form with explicit clipping for numerical stability.
    """

    if tau <= 0:
        return gaussian(x, mu, sigma)

    K = tau / sigma
    logpdf = exponnorm.logpdf(x, K, loc=mu, scale=sigma)
    return _safe_exp(logpdf)


def gaussian(x, mu, sigma):
    """Standard Gaussian PDF (unit area)."""
    expo = -0.5 * ((x - mu) / sigma) ** 2
    return _safe_exp(expo) / (sigma * np.sqrt(2 * np.pi))


def two_point_calibration(adc_centroids, energies):
    """
    Given two reference points (adc_centroids = [adc1, adc2], energies = [E1, E2]),
    solve for slope a and intercept c:  E = a * ADC + c.
    Returns (a, c).
    """
    x1, x2 = adc_centroids
    E1, E2 = energies
    if x1 == x2:
        raise ValueError("ADC centroids must be distinct for calibration")
    a = (E2 - E1) / (x2 - x1)
    c = E1 - a * x1
    return float(a), float(c)


def apply_calibration(adc_values, slope, intercept, quadratic_coeff=0.0):
    """Convert ADC values to energy using calibration coefficients.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values to convert.
    slope : float
        Linear calibration coefficient ``a`` in ``E = a*ADC + c``.
    intercept : float
        Calibration intercept ``c``.
    quadratic_coeff : float, optional
        Quadratic coefficient ``a2`` when using ``E = a2*ADC**2 + a*ADC + c``.
        Defaults to ``0.0`` for the legacy linear behaviour.
    """

    adc_arr = np.asarray(adc_values, dtype=float)
    return quadratic_coeff * adc_arr ** 2 + slope * adc_arr + intercept


def calibrate_run(adc_values, config, hist_bins=None):
    """
    Main entry to derive calibration constants for a single run:
      - Build a histogram of the ADC spectrum. By default each ADC
        channel forms its own bin. When ``hist_bins`` is given the
        range is divided into that many bins instead.
      - Locate the approximate Po-210, Po-218 and Po-214 peaks.
      - Fit each peak with an EMG or Gaussian depending on the
        configuration flags.
      - Compute the two-point calibration using Po-210 and Po-214.

    Returns
    -------
    dict
        Dictionary with calibration results. Keys include ``slope_MeV_per_ch``,
        ``quadratic_MeV_per_ch2``, ``intercept``, ``sigma_E`` and
        ``sigma_E_error`` together with the covariance terms
        ``ac_covariance``, ``a2_variance``, ``cov_a_a2``, ``cov_a2_c`` and the
        fitted ``peaks`` mapping.
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
    use_emg = config["calibration"]["use_emg"]  # True/False
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
        tau0 = max(tau_cfg, _TAU_MIN) if use_emg else 0.0

        if use_emg and iso in ("Po210", "Po218"):
            # Fit EMG: parameters [amp, mu, sigma, tau]
            def model_emg(x, A, mu, sigma, tau):
                return A * emg_left(x, mu, sigma, tau)

            p0 = [amp0, mu0, sigma0, tau0]
            bounds = (
                [0, mu0 - window, 1e-3, _TAU_MIN],  # lower bounds
                [np.inf, mu0 + window, 50.0, 200.0],  # upper bounds (tunable)
            )
            popt, pcov = curve_fit(
                model_emg, x_slice, y_slice, p0=p0, bounds=bounds
            )
            A_fit, mu_fit, sigma_fit, tau_fit = popt
            peak_fits[iso] = {
                "centroid_adc": float(mu_fit),
                "sigma_adc": float(sigma_fit),
                "tau_adc": float(tau_fit),
                "amplitude": float(A_fit),
                "covariance": pcov.tolist(),
            }
        else:
            # Fit pure Gaussian: [A, mu, sigma]
            def model_gauss(x, A, mu, sigma):
                return A * gaussian(x, mu, sigma)

            p0 = [amp0, mu0, sigma0]
            bounds = ([0, mu0 - window, 1e-3], [np.inf, mu0 + window, 50.0])
            popt, pcov = curve_fit(
                model_gauss, x_slice, y_slice, p0=p0, bounds=bounds
            )
            A_fit, mu_fit, sigma_fit = popt
            peak_fits[iso] = {
                "centroid_adc": float(mu_fit),
                "sigma_adc": float(sigma_fit),
                "tau_adc": 0.0,
                "amplitude": float(A_fit),
                "covariance": pcov.tolist(),
            }

    # 5) Calibration coefficients
    adc210 = peak_fits["Po210"]["centroid_adc"]
    adc214 = peak_fits["Po214"]["centroid_adc"]
    adc218 = peak_fits["Po218"]["centroid_adc"]

    cfg_energies = config.get("calibration", {}).get("known_energies")
    energies = (
        DEFAULT_KNOWN_ENERGIES
        if cfg_energies is None
        else {
            **DEFAULT_KNOWN_ENERGIES,
            **cfg_energies,
        }
    )

    E210 = energies["Po210"]
    E214 = energies["Po214"]
    E218 = energies["Po218"]

    quadratic = bool(
        config.get("calibration", {}).get("quadratic", False)
    )

    if quadratic:
        # Solve for quadratic coefficients a2, a, c using all three peaks
        A = np.array([
            [adc210 ** 2, adc210, 1.0],
            [adc218 ** 2, adc218, 1.0],
            [adc214 ** 2, adc214, 1.0],
        ])
        y = np.array([E210, E218, E214], dtype=float)
        a2, a, c = np.linalg.solve(A, y)
    else:
        a, c = two_point_calibration([adc210, adc214], [E210, E214])
        a2 = 0.0

    # 6) Convert fitted centroids to energy for sanity checks
    for iso, info in peak_fits.items():
        info["centroid_mev"] = float(
            apply_calibration(info["centroid_adc"], a, c, quadratic_coeff=a2)
        )

    # Sanity check that fitted energies match expectations within tolerance
    tol = float(config.get("calibration", {}).get("sanity_tolerance_mev", 0.5))
    for iso in ("Po210", "Po218", "Po214"):
        diff = abs(peak_fits[iso]["centroid_mev"] - energies[iso])
        if diff > tol:
            raise RuntimeError(
                f"{iso} peak at {peak_fits[iso]['centroid_mev']:.3f} MeV "
                f"outside ±{tol} MeV of expected {energies[iso]}"
            )

    # 7) Propagate centroid uncertainties to calibration coefficients
    mu_err_210 = float(np.sqrt(peak_fits["Po210"]["covariance"][1][1]))
    mu_err_214 = float(np.sqrt(peak_fits["Po214"]["covariance"][1][1]))
    mu_err_218 = float(np.sqrt(peak_fits["Po218"]["covariance"][1][1]))

    if quadratic:
        def solve_coeff(m):
            A = np.array(
                [
                    [m[0] ** 2, m[0], 1.0],
                    [m[1] ** 2, m[1], 1.0],
                    [m[2] ** 2, m[2], 1.0],
                ]
            )
            return np.linalg.solve(A, np.array([E210, E218, E214], dtype=float))

        mus = np.array([adc210, adc218, adc214], dtype=float)
        coeff0 = np.array([a2, a, c], dtype=float)
        eps = 1.0e-6
        J = np.zeros((3, 3))
        for i in range(3):
            m_step = mus.copy()
            m_step[i] += eps
            J[:, i] = (solve_coeff(m_step) - coeff0) / eps

        sigma_vec = np.array([mu_err_210, mu_err_218, mu_err_214], dtype=float)
        cov_coeff = J @ np.diag(sigma_vec ** 2) @ J.T
        var_a2, var_a, var_c = np.diag(cov_coeff)
        cov_a_a2 = cov_coeff[1, 0]
        cov_ac = cov_coeff[1, 2]
        cov_a2_c = cov_coeff[0, 2]
    else:
        delta = adc214 - adc210
        var_a = (a / delta) ** 2 * (mu_err_210 ** 2 + mu_err_214 ** 2)
        var_c = (
            (a * adc214 / delta) ** 2 * mu_err_210 ** 2
            + (a * adc210 / delta) ** 2 * mu_err_214 ** 2
        )
        cov_ac = (a ** 2 / delta ** 2) * (adc214 * mu_err_210 ** 2 + adc210 * mu_err_214 ** 2)
        var_a2 = 0.0
        cov_a_a2 = 0.0
        cov_a2_c = 0.0

    # 8) Convert σADC -> σE (MeV) using local derivative at the Po-214 peak.
    slope_local = 2 * a2 * adc214 + a
    sigma_adc214 = peak_fits["Po214"]["sigma_adc"]
    sigma_E = abs(slope_local) * sigma_adc214

    var_slope_local = var_a + (2 * adc214) ** 2 * var_a2 + 2 * (2 * adc214) * cov_a_a2
    var_sigma_adc = peak_fits["Po214"]["covariance"][2][2]
    var_sigma_E = (sigma_adc214 ** 2) * var_slope_local + (slope_local ** 2) * var_sigma_adc
    dsigma_E = float(np.sqrt(max(var_sigma_E, 0.0)))

    # 8) Build result object:
    if quadratic:
        coeffs = [c, a, a2]
        cov = np.array(
            [
                [var_c, cov_ac, cov_a2_c],
                [cov_ac, var_a, cov_a_a2],
                [cov_a2_c, cov_a_a2, var_a2],
            ]
        )
    else:
        coeffs = [c, a]
        cov = np.array(
            [
                [var_c, cov_ac],
                [cov_ac, var_a],
            ]
        )

    result = CalibrationResult(
        coeffs=coeffs,
        cov=cov,
        sigma_E=float(sigma_E),
        sigma_E_error=dsigma_E,
        peaks=peak_fits,
    )
    return result


def derive_calibration_constants(adc_values, config):
    """Return :class:`CalibrationResult` for ``adc_values`` with ``config``."""
    return calibrate_run(adc_values, config)


def derive_calibration_constants_auto(
    adc_values,
    noise_cutoff=DEFAULT_NOISE_CUTOFF,
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
        ``"Po210"``, ``"Po218"`` and ``"Po214"``.  If ``None`` the
        :data:`constants.DEFAULT_NOMINAL_ADC` values are used.

    """
    if len(adc_values) == 0:
        raise RuntimeError("No ADC values provided")

    adc_arr = np.asarray(adc_values, dtype=float)
    mask = adc_arr >= noise_cutoff
    adc_arr = adc_arr[mask]

    if nominal_adc is None:
        nominal_adc = DEFAULT_NOMINAL_ADC

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
            "sanity_tolerance_mev": 0.5,
        }
    }

    if nominal_adc is not None:
        config["calibration"]["nominal_adc"] = dict(nominal_adc)

    # Run calibration with custom histogram binning
    return calibrate_run(adc_arr, config, hist_bins=hist_bins)


__all__ = [
    "CalibrationResult",
    "two_point_calibration",
    "apply_calibration",
    "calibrate_run",
    "derive_calibration_constants",
    "derive_calibration_constants_auto",
]
