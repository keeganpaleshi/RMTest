import logging
import numpy as np

logger = logging.getLogger(__name__)
from dataclasses import dataclass
from collections.abc import Sequence, Mapping
from copy import deepcopy
import warnings
from scipy.signal import find_peaks
from scipy.optimize import curve_fit
from scipy.stats import exponnorm
import utils
from constants import (
    DEFAULT_NOISE_CUTOFF,
    DEFAULT_NOMINAL_ADC,
    DEFAULT_KNOWN_ENERGIES,
    CURVE_FIT_MAX_EVALS,
    safe_exp as _safe_exp,
)
try:
    from rmtest.emg_constants import EMG_MIN_TAU as _TAU_MIN
except ImportError:
    from constants import _TAU_MIN
import emg_stable as _emg_module
import constants as _constants_module
from emg_stable import StableEMG, emg_left_stable

try:
    from rmtest.spectral.shapes import emg_loc_to_mode as _emg_loc_to_mode
except ImportError:  # pragma: no cover - fallback for local scripts
    try:
        from src.rmtest.spectral.shapes import emg_loc_to_mode as _emg_loc_to_mode
    except ImportError:  # pragma: no cover - defensive fallback
        def _emg_loc_to_mode(loc, sigma, tau):
            return loc

_USE_STABLE_EMG_DEFAULT = True
_LOGGER = logging.getLogger(__name__)

# Default fitting bounds and thresholds (can be overridden via config)
_MIN_PEAK_SEPARATION_ADC_DEFAULT = 10.0
_GAUSSIAN_SIGMA_UPPER_BOUND_DEFAULT = 500.0
_EMG_SIGMA_UPPER_BOUND_DEFAULT = 250.0
_EMG_TAU_UPPER_BOUND_DEFAULT = 200.0
_QUADRATIC_RESIDUAL_THRESHOLD_DEFAULT = 0.005
_JACOBIAN_EPSILON_DEFAULT = 1.0e-6
_TAU_UPPER_BOUND_DEFAULT = 50.0
_TAU_UPPER_BOUND_PO218_DEFAULT = 8.0


_EMG_MODE_ALIASES = {
    "": "scipy_safe",
    "auto": "scipy_safe",
    "default": "scipy_safe",
    "erfcx": "erfcx_exact",
    "erfcx_exact": "erfcx_exact",
    "direct": "scipy_safe",      # "direct" method uses scipy_safe strategy
    "legacy": "legacy",
    "exponnorm": "legacy",
    "off": "legacy",
    "disabled": "legacy",
    "scipy": "legacy",
    "scipy_safe": "scipy_safe",
    "stable": "scipy_safe",
}


def _normalize_emg_mode(mode):
    text = "" if mode is None else str(mode).strip()
    key = text.lower()
    if key in _EMG_MODE_ALIASES:
        return _EMG_MODE_ALIASES[key]
    return text or "scipy_safe"


def _mode_prefers_stable(mode):
    key = str(mode).strip().lower()
    if not key:
        return None
    if key in {"legacy"}:
        return False
    if key in {"scipy_safe", "erfcx_exact"}:
        return True
    return None


def _sync_fitting_emg_stable_mode(value: bool | None = None) -> None:
    try:
        import fitting as _fitting  # type: ignore
    except ImportError:  # pragma: no cover - optional dependency path
        return

    if value is None:
        current = bool(globals().get("USE_STABLE_EMG", _USE_STABLE_EMG_DEFAULT))
    else:
        current = bool(value)

    updater = getattr(_fitting, "_update_emg_stable_mode_from_config", None)
    if callable(updater):
        updater({"fitting": {"emg_stable_mode": current}})
    else:  # pragma: no cover - compatibility with older versions
        setattr(_fitting, "EMG_STABLE_MODE", current)


if not hasattr(_emg_module, "EMG_STABLE_MODE"):
    setattr(_emg_module, "EMG_STABLE_MODE", "scipy_safe")

EMG_STABLE_MODE = _normalize_emg_mode(getattr(_emg_module, "EMG_STABLE_MODE", "scipy_safe"))
setattr(_emg_module, "EMG_STABLE_MODE", EMG_STABLE_MODE)

# Backwards-compatibility attribute; prefer using the getters/setters below.
_mode_default = _mode_prefers_stable(EMG_STABLE_MODE)
USE_STABLE_EMG = _USE_STABLE_EMG_DEFAULT if _mode_default is None else _mode_default


def get_use_stable_emg() -> bool:
    """Return whether the stable EMG implementation should be used."""

    mode_pref = _mode_prefers_stable(globals().get("EMG_STABLE_MODE", EMG_STABLE_MODE))
    if mode_pref is not None:
        return mode_pref
    return bool(globals().get("USE_STABLE_EMG", _USE_STABLE_EMG_DEFAULT))


def set_use_stable_emg(value: bool) -> None:
    """Configure whether the stable EMG implementation should be used."""

    use_stable = bool(value)
    mode = globals().get("EMG_STABLE_MODE", EMG_STABLE_MODE)
    if use_stable:
        normalized = _normalize_emg_mode(mode)
        if _mode_prefers_stable(normalized) is False:
            normalized = "scipy_safe"
    else:
        normalized = "legacy"

    set_emg_stable_mode(normalized)
    globals()["USE_STABLE_EMG"] = bool(use_stable)
    _sync_fitting_emg_stable_mode()


def set_emg_stable_mode(mode: str) -> None:
    """Set the legacy EMG mode string and keep module state in sync."""

    normalized = _normalize_emg_mode(mode)
    globals()["EMG_STABLE_MODE"] = normalized
    setattr(_emg_module, "EMG_STABLE_MODE", normalized)

    preference = _mode_prefers_stable(normalized)
    if preference is not None:
        globals()["USE_STABLE_EMG"] = preference
        _sync_fitting_emg_stable_mode(preference)
    else:
        _sync_fitting_emg_stable_mode()


def get_emg_tau_min() -> float:
    """Return the EMG tau floor shared with the stable implementation."""

    return float(_TAU_MIN)


def set_emg_tau_min(value: float) -> None:
    """Update the EMG tau floor for both calibration and stable EMG helpers."""

    tau_min = float(value)
    globals()["_TAU_MIN"] = tau_min
    setattr(_constants_module, "_TAU_MIN", tau_min)
    _set_tau_min(tau_min)


def _make_tau_bounds(
    tau_min: float,
    tau_upper_default: float = _TAU_UPPER_BOUND_DEFAULT,
    tau_upper_po218: float = _TAU_UPPER_BOUND_PO218_DEFAULT,
) -> dict[str, tuple[float, float]]:
    return {
        "default": (tau_min, tau_upper_default),
        "Po218": (tau_min, tau_upper_po218),
    }


_DEFAULT_TAU_BOUNDS = _make_tau_bounds(_TAU_MIN)

def _set_tau_min(tau_min: float) -> None:
    """Update the minimum tau value used throughout the calibration module."""

    global _TAU_MIN, _DEFAULT_TAU_BOUNDS
    _TAU_MIN = float(tau_min)
    _DEFAULT_TAU_BOUNDS = _make_tau_bounds(_TAU_MIN)


def configure_emg(use_stable_emg: bool, tau_min: float) -> None:
    """Apply EMG configuration sourced from the analysis configuration file."""

    set_use_stable_emg(use_stable_emg)
    set_emg_tau_min(tau_min)

    try:
        import fitting as _fitting  # type: ignore

        _fitting._TAU_MIN = _TAU_MIN
    except ImportError:  # pragma: no cover - fitting may be optional
        pass


def _coerce_tau_bounds(bounds, iso):
    """Return validated ``(lo, hi)`` tau bounds for ``iso``."""

    if isinstance(bounds, str) or not isinstance(bounds, Sequence):
        raise TypeError(
            f"Tau bounds for {iso} must be a length-2 sequence, got {type(bounds)!r}"
        )

    if len(bounds) != 2:
        raise ValueError(
            f"Tau bounds for {iso} must contain exactly two values, got {bounds!r}"
        )

    lo, hi = float(bounds[0]), float(bounds[1])

    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError(f"Tau bounds for {iso} must be finite, got {bounds!r}")

    lo = max(lo, _TAU_MIN)
    if hi <= lo:
        raise ValueError(
            f"Upper tau bound must exceed lower bound for {iso}, got {bounds!r}"
        )

    return lo, hi


def _resolve_tau_bounds(cal_cfg: Mapping[str, object], iso: str) -> tuple[float, float]:
    """Determine the tau bounds for ``iso`` using defaults and config overrides."""

    tau_cfg = cal_cfg.get("tau_bounds_adc") if isinstance(cal_cfg, Mapping) else None

    if tau_cfg is None:
        return _DEFAULT_TAU_BOUNDS.get(iso, _DEFAULT_TAU_BOUNDS["default"])

    if isinstance(tau_cfg, Mapping):
        bounds = tau_cfg.get(iso, tau_cfg.get("default"))
        if bounds is None:
            return _DEFAULT_TAU_BOUNDS.get(iso, _DEFAULT_TAU_BOUNDS["default"])
    else:
        bounds = tau_cfg

    lo, hi = _coerce_tau_bounds(bounds, iso)
    return lo, hi


def _resolve_curve_fit_max_evals(cal_cfg: Mapping[str, object] | None) -> int:
    """Return the max ``curve_fit`` evaluations requested by the configuration."""

    if not isinstance(cal_cfg, Mapping):
        return CURVE_FIT_MAX_EVALS

    key_used = None
    value = None
    for key in ("fit_maxfev", "curve_fit_max_evaluations", "curve_fit_max_evals"):
        if cal_cfg.get(key) is not None:
            key_used = key
            value = cal_cfg[key]
            break

    if value is None:
        return CURVE_FIT_MAX_EVALS

    try:
        max_evals = int(value)
    except (TypeError, ValueError) as exc:  # pragma: no cover - safety net
        raise ValueError(
            f"calibration.{key_used} must be a positive integer, got {value!r}"
        ) from exc

    if max_evals <= 0:
        raise ValueError(
            f"calibration.{key_used} must be a positive integer, got {value!r}"
        )

    return max_evals


def _resolve_retry_window(
    cal_cfg: Mapping[str, object] | None, default_window: float
) -> float:
    """Return the retry window size for peak fits."""

    if not isinstance(cal_cfg, Mapping):
        return default_window

    retry_window = cal_cfg.get("fit_retry_window")
    if retry_window is None:
        return default_window

    try:
        window_val = float(retry_window)
    except (TypeError, ValueError) as exc:  # pragma: no cover - safety net
        raise ValueError(
            f"calibration.fit_retry_window must be positive, got {retry_window!r}"
        ) from exc

    if window_val <= 0:
        raise ValueError(
            f"calibration.fit_retry_window must be positive, got {retry_window!r}"
        )

    return window_val


@dataclass(init=False)
class CalibrationResult:
    """Polynomial calibration coefficients and covariance.

    The ``cov`` argument may also be provided as ``covariance`` for backward
    compatibility. Only one of the two may be given.
    """

    coeffs: Sequence[float] | Mapping[int, float]
    cov: np.ndarray
    peaks: dict | None = None
    sigma_E: float = 0.0
    sigma_E_error: float = 0.0
    status: Mapping[str, str] | None = None

    def __init__(
        self,
        *,
        coeffs,
        cov=None,
        peaks=None,
        sigma_E=0.0,
        sigma_E_error=0.0,
        status=None,
        covariance=None,
    ):
        if covariance is not None:
            if cov is not None:
                raise TypeError("Specify either 'cov' or 'covariance', not both")
            cov = covariance

        self.coeffs = coeffs
        if cov is None:
            raise TypeError("Missing required argument 'cov'")
        self.cov = cov
        self.peaks = peaks
        self.sigma_E = sigma_E
        self.sigma_E_error = sigma_E_error
        self.status = status
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
        powers = np.stack([x_arr**p for p in self._exponents], axis=-1)
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
            J = np.array([val**p for p in self._exponents], dtype=float)
            sig2.append(J @ self.cov @ J)
        out = np.sqrt(np.clip(sig2, 0, None))
        return out[0] if np.ndim(x) == 0 else out

    def _coef_index(self, label):
        """Return coefficient index for ``label``.

        ``label`` may be an integer exponent or one of ``"c"``, ``"a"`` or
        ``"a2"``. String representations of integer exponents are also
        accepted. A :class:`KeyError` is raised if the coefficient is not
        present.
        """

        if isinstance(label, str):
            key = label.lower()
            mapping = {"c": 0, "a": 1, "a2": 2, "a3": 3}
            if key in mapping:
                exp = mapping[key]
            else:
                try:
                    exp = int(key)
                except ValueError:
                    raise KeyError(f"Unknown coefficient: {label}") from None
        elif isinstance(label, (int, np.integer)):
            exp = int(label)
        else:
            raise KeyError(f"Invalid coefficient label: {label!r}")

        try:
            return self._exponents.index(exp)
        except ValueError as exc:
            raise KeyError(f"Coefficient not present: {label}") from exc

    def get_cov(self, name1, name2):
        """Return covariance entry for two coefficients.

        Parameters can be coefficient names (``"a"``, ``"c"``, ``"a2"``) or
        integer exponents.
        """

        if self.cov is None:
            return 0.0

        i1 = self._coef_index(name1)
        i2 = self._coef_index(name2)
        cov = np.asarray(self.cov, dtype=float)

        if cov.ndim >= 2 and i1 < cov.shape[0] and i2 < cov.shape[1]:
            return float(cov[i1, i2])

        raise KeyError(f"Coefficient(s) missing in covariance: {name1}, {name2}")


def emg_left(x, mu, sigma, tau):
    """Exponentially modified Gaussian (left-skewed) PDF.

    Provides two interchangeable implementations that share the same
    ``scipy.stats.exponnorm`` lineshape: the legacy direct call and a
    stabilized wrapper. The stabilized version primarily cleans NaN/Inf
    outputs and falls back to a Gaussian when ``tau`` is extremely small,
    without altering the underlying physics model.

    Args:
        x: Input values (energy in MeV or ADC units)
        mu: Gaussian mean (peak center)
        sigma: Gaussian standard deviation (resolution)
        tau: Exponential decay constant (tail parameter)

    Returns:
        EMG probability density values (unit area PDF)
    """

    tau_min = get_emg_tau_min()
    if tau <= tau_min:
        return gaussian(x, mu, sigma)

    if get_use_stable_emg():
        # Mirror the standard exponnorm lineshape so the exponential tail sits on the low-energy side.
        return emg_left_stable(x, mu, sigma, tau, amplitude=1.0, use_log_scale=False)
    else:
        # Legacy implementation using a mirrored exponnorm to produce a left tail.
        K = tau / sigma
        x_mirror = 2.0 * np.asarray(mu, dtype=float) - np.asarray(x, dtype=float)
        logpdf = exponnorm.logpdf(x_mirror, K, loc=mu, scale=sigma)
        return _safe_exp(logpdf)


def gaussian(x, mu, sigma):
    """Standard Gaussian PDF (unit area)."""
    if np.any(sigma <= 0):
        raise ValueError(f"sigma must be positive, got {sigma}")
    expo = -0.5 * ((x - mu) / sigma) ** 2
    return _safe_exp(expo) / (sigma * np.sqrt(2 * np.pi))



def _peak_centroid_from_fit(mu_fit: float, sigma_fit: float, tau_fit: float) -> float:
    """Return the visible peak centroid from fit parameters."""
    if tau_fit <= _TAU_MIN:
        return float(mu_fit)
    return float(_emg_loc_to_mode(mu_fit, sigma_fit, tau_fit))

def _build_peak_slice(iso: str, centers, hist, x0: float, window: float):
    """Return histogram slices for a peak centered at ``x0`` within ``window``."""

    lo = x0 - window
    hi = x0 + window
    mask = (centers >= lo) & (centers <= hi)
    x_slice = centers[mask]
    y_slice = hist[mask].astype(float)

    if len(x_slice) < 5:
        raise RuntimeError(
            f"Not enough points to fit peak for {iso} (only {len(x_slice)} bins)."
        )

    return x_slice, y_slice


def _fit_gaussian_peak(
    iso: str,
    centers,
    hist,
    x0: float,
    window: float,
    sigma0: float,
    maxfev: int,
    retry_window: float | None,
    sigma_upper_bound: float = _GAUSSIAN_SIGMA_UPPER_BOUND_DEFAULT,
):
    """Fit a Gaussian peak with one retry using a larger window."""

    def _attempt(window_size: float):
        x_slice, y_slice = _build_peak_slice(iso, centers, hist, x0, window_size)
        amp_guess = float(np.max(y_slice))
        # Prefer the configured width guess; a broad slice std can drag the
        # optimizer to window edges when the spectrum is sparse (e.g., the
        # pipeline smoke test input).
        sigma_guess = float(max(sigma0, 1e-3))

        def model_gauss(x, A, mu, sigma):
            return A * gaussian(x, mu, sigma)

        p0 = [amp_guess, float(x0), sigma_guess]
        # Upper sigma bound: generous limit to avoid constraining wide peaks
        bounds = ([0, x0 - window_size, 1e-3], [np.inf, x0 + window_size, sigma_upper_bound])
        popt, pcov = curve_fit(
            model_gauss, x_slice, y_slice, p0=p0, bounds=bounds, maxfev=maxfev
        )
        return popt, pcov

    try:
        return _attempt(window)
    except RuntimeError as exc:
        retry_size = retry_window if retry_window is not None else window
        if retry_size == window:
            raise
        _LOGGER.warning(
            "Gaussian fit for %s failed (%s); retrying with window %.3g", iso, exc, retry_size
        )
        return _attempt(retry_size)


def _fit_emg_peak(
    iso: str,
    centers,
    hist,
    x0: float,
    window: float,
    sigma0: float,
    tau0: float,
    tau_bounds: tuple[float, float],
    maxfev: int,
    retry_window: float | None,
    sigma_upper_bound: float = _EMG_SIGMA_UPPER_BOUND_DEFAULT,
):
    """Fit an EMG peak with a Gaussian retry fallback."""

    tau_lo, tau_hi = tau_bounds

    def _attempt(window_size: float):
        x_slice, y_slice = _build_peak_slice(iso, centers, hist, x0, window_size)
        amp_guess = float(np.max(y_slice))
        # Keep EMG starting width aligned with configuration; using the slice
        # std can overly broaden the initial guess on small spectra.
        sigma_guess = float(max(sigma0, 1e-3))

        def model_emg(x, A, mu, sigma, tau):
            return A * emg_left(x, mu, sigma, tau)

        p0 = [amp_guess, float(x0), sigma_guess, tau0]
        # Upper sigma bound: generous limit to avoid constraining wide peaks
        bounds = ([0, x0 - window_size, 1e-3, tau_lo], [np.inf, x0 + window_size, sigma_upper_bound, tau_hi])
        popt, pcov = curve_fit(
            model_emg, x_slice, y_slice, p0=p0, bounds=bounds, maxfev=maxfev
        )
        return popt, pcov

    try:
        return _attempt(window)
    except RuntimeError as exc:
        retry_size = retry_window if retry_window is not None else window
        _LOGGER.warning(
            "EMG fit for %s failed (%s); retrying with Gaussian fallback (window %.3g)",
            iso,
            exc,
            retry_size,
        )
        popt_gauss, pcov_gauss = _fit_gaussian_peak(
            iso, centers, hist, x0, retry_size, sigma0, maxfev, None
        )
        popt = np.concatenate([popt_gauss, [0.0]])
        pcov = np.zeros((4, 4))
        pcov[:3, :3] = pcov_gauss
        return popt, pcov
def two_point_calibration(adc_centroids, energies):
    """
    Given two reference points (adc_centroids = [adc1, adc2], energies = [E1, E2]),
    solve for slope a and intercept c:  E = a * ADC + c.
    Returns (a, c).
    """
    x1, x2 = adc_centroids
    E1, E2 = energies
    if abs(x1 - x2) < 1e-10:
        raise ValueError("ADC centroids must be distinct for calibration")
    if abs(E1 - E2) < 1e-10:
        raise ValueError("Energies must be distinct for calibration")
    a = (E2 - E1) / (x2 - x1)
    c = E1 - a * x1
    return float(a), float(c)


def fixed_slope_calibration(adc_values, cfg, *, status=None):
    """Return calibration constants when the slope is fixed.

    Parameters
    ----------
    adc_values : array-like
        Raw ADC values from a single run.
    cfg : dict
        Configuration dictionary containing ``calibration`` options.

    Notes
    -----
    - Returns a CalibrationResult dataclass (keeps #1049 API).
    - If ``calibration.sigma_E_init`` is provided (scalar or mapping),
      it is used to set the **initial guess** for the peak width by
      converting MeV -> ADC via the fixed slope. It does not override
      the fitted sigma_E in the result.
    """

    cal_cfg = cfg.get("calibration", {})
    a = cal_cfg.get("slope_mev_per_ch", cal_cfg.get("slope_MeV_per_ch"))

    fit_maxfev = _resolve_curve_fit_max_evals(cal_cfg)
    expected = {"Po214": cal_cfg["nominal_adc"]["Po214"]}
    window = cal_cfg.get("peak_search_radius", 50)
    prominence = cal_cfg.get("peak_prominence", 0.0)

    # Optional per-isotope width overrides (used by the discrete peak finder)
    width_cfg = cal_cfg.get("peak_widths") or {}
    if isinstance(width_cfg, Mapping):
        width_po214 = width_cfg.get("Po214", cal_cfg.get("peak_width"))
    else:
        width_po214 = cal_cfg.get("peak_width")

    peaks = utils.find_adc_bin_peaks(
        adc_values,
        expected,
        window=window,
        prominence=prominence,
        width=width_po214,
    )
    adc_peak = peaks["Po214"]

    # Build a channel-centered histogram for the fit.
    adc_arr = np.asarray(adc_values, dtype=float)
    edges = utils.adc_hist_edges(adc_arr, channel_width=1.0)
    hist, edges = np.histogram(adc_arr, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    fit_window = cal_cfg.get("fit_window_adc", 50)
    retry_window = _resolve_retry_window(cal_cfg, fit_window)
    use_emg = cal_cfg.get("use_emg", False)

    # Initial guesses
    mu0 = float(adc_peak)

    # Base ADC-space width guess
    sigma_cfg = cal_cfg.get("init_sigma_adc", 10.0)
    if isinstance(sigma_cfg, Mapping):
        sigma0 = sigma_cfg.get("Po214", sigma_cfg.get("default", 10.0))
    else:
        sigma0 = sigma_cfg

    # Optional energy-space width guess -> convert to ADC using fixed slope
    sigma_E_cfg = cal_cfg.get("sigma_e_init", cal_cfg.get("sigma_E_init"))
    if sigma_E_cfg is not None:
        if isinstance(sigma_E_cfg, Mapping):
            sigma_E_guess = sigma_E_cfg.get("Po214", sigma_E_cfg.get("default"))
        else:
            sigma_E_guess = sigma_E_cfg
        if sigma_E_guess is not None:
            # guard against zero slope, though config should never set that
            if a == 0:
                raise ValueError("Fixed slope 'a' must be nonzero for sigma_E_init conversion")
            sigma0 = float(abs(sigma_E_guess) / abs(a))

    tau_cfg = cal_cfg.get("init_tau_adc", 0.0)
    tau0 = max(tau_cfg, _TAU_MIN) if use_emg else 0.0

    # Read configurable bounds
    emg_tau_upper = cal_cfg.get("emg_tau_upper_bound", _EMG_TAU_UPPER_BOUND_DEFAULT)
    emg_sigma_upper = cal_cfg.get("emg_sigma_upper_bound", _EMG_SIGMA_UPPER_BOUND_DEFAULT)
    gaussian_sigma_upper = cal_cfg.get("gaussian_sigma_upper_bound", _GAUSSIAN_SIGMA_UPPER_BOUND_DEFAULT)

    if use_emg:
        popt, pcov = _fit_emg_peak(
            "Po214",
            centers,
            hist,
            mu0,
            fit_window,
            sigma0,
            tau0,
            (_TAU_MIN, emg_tau_upper),
            fit_maxfev,
            retry_window,
            sigma_upper_bound=emg_sigma_upper,
        )
        A_fit, mu_fit, sigma_fit, tau_fit = popt
    else:
        popt, pcov = _fit_gaussian_peak(
            "Po214", centers, hist, mu0, fit_window, sigma0, fit_maxfev, retry_window,
            sigma_upper_bound=gaussian_sigma_upper,
        )
        A_fit, mu_fit, sigma_fit = popt
        tau_fit = 0.0

    centroid_adc = _peak_centroid_from_fit(mu_fit, sigma_fit, tau_fit)
    peak_info = {
        "centroid_adc": float(centroid_adc),
        "loc_adc": float(mu_fit),
        "sigma_adc": float(sigma_fit),
        "tau_adc": float(tau_fit),
        "amplitude": float(A_fit),
        "covariance": pcov.tolist(),
    }

    # Intercept from the known Po214 energy and fixed slope
    energies = {**DEFAULT_KNOWN_ENERGIES, **cal_cfg.get("known_energies", {})}
    E_known = energies["Po214"]
    c = float(E_known - a * centroid_adc)
    peak_info["centroid_mev"] = float(apply_calibration(centroid_adc, a, c))
    peak_info["loc_mev"] = float(apply_calibration(mu_fit, a, c))

    # Convert sigma from ADC to energy using fixed slope and propagate errors
    sigma_adc = float(sigma_fit)
    sigma_E = abs(a) * sigma_adc
    var_sigma_adc = float(pcov[2][2])
    dsigma_E = abs(a) * float(np.sqrt(max(var_sigma_adc, 0.0)))

    # Intercept uncertainty from centroid fit; slope is fixed
    mu_err = float(np.sqrt(pcov[1][1]))
    var_c = (a * mu_err) ** 2
    cov = np.array([[var_c, 0.0], [0.0, 0.0]])

    result = CalibrationResult(
        coeffs=[c, a],
        cov=cov,
        peaks={"Po214": peak_info},
        sigma_E=float(sigma_E),
        sigma_E_error=dsigma_E,
        status=status,
    )
    return result


def apply_calibration(adc_values, slope, intercept, quadratic_coeff=0.0, cubic_coeff=0.0):
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
    cubic_coeff : float, optional
        Cubic coefficient ``a3`` when using
        ``E = a3*ADC**3 + a2*ADC**2 + a*ADC + c``.
        Defaults to ``0.0``.
    """

    adc_arr = np.asarray(adc_values, dtype=float)
    return cubic_coeff * adc_arr**3 + quadratic_coeff * adc_arr**2 + slope * adc_arr + intercept


def _fallback_to_fixed_slope(adc_values, config, exc, warning_message=None):
    if warning_message is not None:
        warnings.warn(warning_message, RuntimeWarning)

    _LOGGER.exception(
        "Calibration failed with configured slope; using fixed-slope fallback (cause: %s)",
        exc,
        exc_info=exc,
    )

    cfg_fallback = deepcopy(config)
    cfg_fallback.setdefault("calibration", {})
    cfg_fallback["calibration"]["float_slope"] = False
    cfg_fallback["calibration"]["use_two_point"] = False

    status = {"fallback": "fixed_slope", "cause": str(exc)}
    return fixed_slope_calibration(adc_values, cfg_fallback, status=status)


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
    edges = utils.adc_hist_edges(adc_values, hist_bins=hist_bins)
    hist, edges = np.histogram(adc_values, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])

    # 2) Peak‐finding with SciPy:
    #    We look for up to 5 peaks (Po-210, Po-218, Po-214, maybe background bumps).
    #    Use config thresholds for prominence & width.
    cal_cfg = config.get("calibration", {})
    fit_maxfev = _resolve_curve_fit_max_evals(cal_cfg)

    prom = cal_cfg["peak_prominence"]
    width_cfg = cal_cfg.get("peak_widths") or {}
    wid_global = cal_cfg.get("peak_width")
    # If per-isotope widths are specified but no global width, use minimal width
    # to ensure find_peaks calculates widths for filtering
    width_for_peaks = wid_global
    if wid_global is None and isinstance(width_cfg, Mapping) and width_cfg:
        width_for_peaks = 0  # Minimal width to trigger width calculation
    peaks, props = find_peaks(hist, prominence=prom, width=width_for_peaks)

    # 3) From the found peaks, pick the three that best match expected energies:
    #    Convert expected energy->ADC guess via last calibration or central ADC->MeV mapping if available.
    #    For the first run, use rough nominal ADC guesses from config:
    # e.g. {"Po210": 800, "Po218": 900, "Po214": 1200}
    nominal_adc = cal_cfg["nominal_adc"]

    # Build a dictionary: { isotope : list of candidate indices within +/- radius }
    # radius (in ADC channels) to match found peaks to nominal guesses
    # 2024-03: renamed from ``peak_search_radius_adc`` to ``peak_search_radius``
    radius = cal_cfg["peak_search_radius"]
    candidates = {iso: [] for iso in nominal_adc}
    peak_widths_found = props.get("widths", np.zeros_like(peaks))
    for iso, adc_guess in nominal_adc.items():
        req_width = None
        if isinstance(width_cfg, Mapping):
            req_width = width_cfg.get(iso)
        for idx, pwidth in zip(peaks, peak_widths_found):
            if req_width is not None and pwidth < req_width:
                continue
            if abs(centers[idx] - adc_guess) <= radius:
                candidates[iso].append(idx)
    # If multiple candidates per isotope, pick the one with highest histogram count:
    # Core isotopes (Po210, Po218, Po214) are required; additional ones are optional.
    _core_isos = {"Po210", "Po218", "Po214"}
    chosen_idx = {}
    for iso in candidates:
        if not candidates[iso]:
            if iso in _core_isos:
                raise RuntimeError(
                    f"No candidate peak found for {iso} around ADC={nominal_adc[iso]}."
                )
            logging.getLogger(__name__).warning(
                "No candidate peak found for %s around ADC=%s -- skipping (optional isotope).",
                iso, nominal_adc[iso],
            )
            continue
        # pick the one with max(hist) among candidates
        best = max(candidates[iso], key=lambda i: hist[i])
        chosen_idx[iso] = best

    # 4) For each chosen peak, perform a local fit with EMG or Gaussian:
    peak_fits = {}
    window = cal_cfg["fit_window_adc"]  # e.g. +/-50 ADC channels
    retry_window = _resolve_retry_window(cal_cfg, window)
    use_emg = cal_cfg["use_emg"]  # True/False

    # Read configurable bounds from config
    emg_sigma_upper = cal_cfg.get("emg_sigma_upper_bound", _EMG_SIGMA_UPPER_BOUND_DEFAULT)
    gaussian_sigma_upper = cal_cfg.get("gaussian_sigma_upper_bound", _GAUSSIAN_SIGMA_UPPER_BOUND_DEFAULT)

    for iso, idx_center in chosen_idx.items():
        x0 = centers[idx_center]
        mu0 = float(x0)
        sigma_cfg = cal_cfg.get("init_sigma_adc", 10.0)
        if isinstance(sigma_cfg, Mapping):
            sigma0 = sigma_cfg.get(iso, sigma_cfg.get("default", 10.0))
        else:
            sigma0 = sigma_cfg

        sigma_E_cfg = cal_cfg.get("sigma_e_init", cal_cfg.get("sigma_E_init"))
        if sigma_E_cfg is not None:
            if isinstance(sigma_E_cfg, Mapping):
                sigma_E_guess = sigma_E_cfg.get(iso, sigma_E_cfg.get("default"))
            else:
                sigma_E_guess = sigma_E_cfg
            slope_guess = cal_cfg.get("slope_mev_per_ch", cal_cfg.get("slope_MeV_per_ch"))
            if slope_guess is not None and sigma_E_guess is not None:
                if abs(slope_guess) < 1e-15:
                    raise ValueError("slope_mev_per_ch must be nonzero for sigma_e_init conversion")
                sigma0 = abs(sigma_E_guess) / abs(slope_guess)

        tau_cfg = cal_cfg.get("init_tau_adc", 0.0)
        # Avoid zero or negative starting tau which can cause numerical issues
        tau0 = max(tau_cfg, _TAU_MIN) if use_emg else 0.0

        if use_emg and iso in ("Po210", "Po218"):
            tau_bounds = _resolve_tau_bounds(cal_cfg, iso)
            popt, pcov = _fit_emg_peak(
                iso,
                centers,
                hist,
                mu0,
                window,
                sigma0,
                tau0,
                tau_bounds,
                fit_maxfev,
                retry_window,
                sigma_upper_bound=emg_sigma_upper,
            )
            A_fit, mu_fit, sigma_fit, tau_fit = popt
        else:
            popt, pcov = _fit_gaussian_peak(
                iso, centers, hist, mu0, window, sigma0, fit_maxfev, retry_window,
                sigma_upper_bound=gaussian_sigma_upper,
            )
            A_fit, mu_fit, sigma_fit = popt
            tau_fit = 0.0
        centroid_adc = _peak_centroid_from_fit(mu_fit, sigma_fit, tau_fit)
        peak_fits[iso] = {
            "centroid_adc": float(centroid_adc),
            "loc_adc": float(mu_fit),
            "sigma_adc": float(sigma_fit),
            "tau_adc": float(tau_fit),
            "amplitude": float(A_fit),
            "covariance": pcov.tolist(),
        }

    # 5) Calibration coefficients
    adc210 = peak_fits["Po210"]["centroid_adc"]
    adc214 = peak_fits["Po214"]["centroid_adc"]
    adc218 = peak_fits["Po218"]["centroid_adc"]

    # Ensure peaks are ordered as Po210 < Po218 < Po214
    if not (adc210 < adc218 < adc214):
        raise RuntimeError(
            "Calibration produced inconsistent peak ordering: "
            f"Po210={adc210:.3f}, Po218={adc218:.3f}, Po214={adc214:.3f}"
        )

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

    quad_opt = config.get("calibration", {}).get("use_quadratic")
    if quad_opt is None:
        quad_old = config.get("calibration", {}).get("quadratic")
        if quad_old is not None:
            quad_opt = "true" if quad_old else "false"
        else:
            quad_opt = "false"
    if isinstance(quad_opt, bool):
        quad_opt = "true" if quad_opt else "false"
    quad_opt = str(quad_opt).lower()

    cubic = quad_opt == "cubic"

    if cubic:
        # Cubic requires Po216 as the 4th calibration point
        if "Po216" not in peak_fits:
            raise RuntimeError(
                "Cubic calibration requires Po216 peak but it was not found. "
                "Ensure Po216 is in nominal_adc and the peak is detectable."
            )
        adc216 = peak_fits["Po216"]["centroid_adc"]
        E216 = energies.get("Po216")
        if E216 is None:
            raise RuntimeError("Cubic calibration requires Po216 in known_energies.")
        if not (adc210 < adc218 < adc216 < adc214):
            raise RuntimeError(
                "Calibration produced inconsistent peak ordering: "
                f"Po210={adc210:.3f}, Po218={adc218:.3f}, "
                f"Po216={adc216:.3f}, Po214={adc214:.3f}"
            )
        quadratic = True  # cubic implies quadratic-level support too

    if quad_opt == "auto":
        a_lin, c_lin = two_point_calibration([adc210, adc214], [E210, E214])
        resid = abs(apply_calibration(adc218, a_lin, c_lin) - E218)
        quad_threshold = cal_cfg.get("quadratic_residual_threshold_mev", _QUADRATIC_RESIDUAL_THRESHOLD_DEFAULT)
        quadratic = resid > quad_threshold
    elif quad_opt == "true":
        quadratic = True
    elif not cubic:
        quadratic = False

    if cubic:
        # Build calibration matrix from available peaks (4+ points for cubic)
        cal_adcs = [adc210, adc218, adc216, adc214]
        cal_Es = [E210, E218, E216, E214]
        # Add Po212 if available for overdetermined least-squares
        if "Po212" in peak_fits:
            adc212 = peak_fits["Po212"]["centroid_adc"]
            E212 = energies.get("Po212")
            if E212 is not None and adc212 > adc214:
                cal_adcs.append(adc212)
                cal_Es.append(E212)
                logger.info(
                    "Cubic calibration: using 5 peaks (including Po212 at ADC=%.1f)",
                    adc212,
                )
        A = np.array(
            [[x**3, x**2, x, 1.0] for x in cal_adcs]
        )
        y = np.array(cal_Es, dtype=float)
        if len(cal_adcs) > 4:
            # Overdetermined: least-squares fit
            result, residuals, _, _ = np.linalg.lstsq(A, y, rcond=None)
            a3, a2, a, c = result
            if len(residuals) > 0:
                rms_resid = float(np.sqrt(residuals[0] / len(cal_adcs)))
                logger.info("Cubic calibration residual RMS: %.6f MeV", rms_resid)
        else:
            # Exact solve with 4 peaks
            a3, a2, a, c = np.linalg.solve(A, y)
    elif quadratic:
        # Solve for quadratic coefficients a2, a, c using all three peaks
        A = np.array(
            [
                [adc210**2, adc210, 1.0],
                [adc218**2, adc218, 1.0],
                [adc214**2, adc214, 1.0],
            ]
        )
        y = np.array([E210, E218, E214], dtype=float)
        a2, a, c = np.linalg.solve(A, y)
        a3 = 0.0
    else:
        a, c = two_point_calibration([adc210, adc214], [E210, E214])
        a2 = 0.0
        a3 = 0.0

    # 6) Convert fitted centroids to energy for sanity checks
    for iso, info in peak_fits.items():
        info["centroid_mev"] = float(
            apply_calibration(info["centroid_adc"], a, c, quadratic_coeff=a2, cubic_coeff=a3)
        )
        if "loc_adc" in info:
            info["loc_mev"] = float(
                apply_calibration(info["loc_adc"], a, c, quadratic_coeff=a2, cubic_coeff=a3)
            )
        info["peak_adc"] = float(info["centroid_adc"])

    # Sanity check that fitted energies match expectations within tolerance
    tol = float(config.get("calibration", {}).get("sanity_tolerance_mev", 0.5))
    for iso in peak_fits:
        if iso not in energies:
            continue
        diff = abs(peak_fits[iso]["centroid_mev"] - energies[iso])
        if diff > tol:
            raise RuntimeError(
                f"{iso} peak at {peak_fits[iso]['centroid_mev']:.3f} MeV "
                f"outside +/-{tol} MeV of expected {energies[iso]}"
            )

    # 7) Propagate centroid uncertainties to calibration coefficients
    def _extract_centroid_error(iso_name):
        cov = peak_fits[iso_name].get("covariance")
        if cov is None or len(cov) < 2 or len(cov[1]) < 2:
            raise ValueError(f"Invalid covariance matrix for {iso_name} peak fit")
        return float(np.sqrt(cov[1][1]))

    mu_err_210 = _extract_centroid_error("Po210")
    mu_err_214 = _extract_centroid_error("Po214")
    mu_err_218 = _extract_centroid_error("Po218")

    if cubic:
        mu_err_216 = _extract_centroid_error("Po216")

        def solve_coeff(m):
            A = np.array(
                [
                    [m[0] ** 3, m[0] ** 2, m[0], 1.0],
                    [m[1] ** 3, m[1] ** 2, m[1], 1.0],
                    [m[2] ** 3, m[2] ** 2, m[2], 1.0],
                    [m[3] ** 3, m[3] ** 2, m[3], 1.0],
                ]
            )
            return np.linalg.solve(A, np.array([E210, E218, E216, E214], dtype=float))

        mus = np.array([adc210, adc218, adc216, adc214], dtype=float)
        coeff0 = np.array([a3, a2, a, c], dtype=float)
        eps_rel = cal_cfg.get("jacobian_epsilon", _JACOBIAN_EPSILON_DEFAULT)
        n_pts = 4
        J = np.zeros((n_pts, n_pts))
        for i in range(n_pts):
            m_step = mus.copy()
            eps = max(abs(mus[i]) * eps_rel, eps_rel)
            m_step[i] += eps
            J[:, i] = (solve_coeff(m_step) - coeff0) / eps

        sigma_vec = np.array([mu_err_210, mu_err_218, mu_err_216, mu_err_214], dtype=float)
        cov_coeff = J @ np.diag(sigma_vec**2) @ J.T
        var_a3, var_a2, var_a, var_c = np.diag(cov_coeff)
        cov_a_a2 = cov_coeff[2, 1]
        cov_ac = cov_coeff[2, 3]
        cov_a2_c = cov_coeff[1, 3]
        var_a3 = cov_coeff[0, 0]

    elif quadratic:

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
        # Use relative epsilon scaled to ADC magnitude for better numerical precision
        eps_rel = cal_cfg.get("jacobian_epsilon", _JACOBIAN_EPSILON_DEFAULT)
        J = np.zeros((3, 3))
        for i in range(3):
            m_step = mus.copy()
            eps = max(abs(mus[i]) * eps_rel, eps_rel)  # Relative epsilon with floor
            m_step[i] += eps
            J[:, i] = (solve_coeff(m_step) - coeff0) / eps

        sigma_vec = np.array([mu_err_210, mu_err_218, mu_err_214], dtype=float)
        cov_coeff = J @ np.diag(sigma_vec**2) @ J.T
        var_a2, var_a, var_c = np.diag(cov_coeff)
        cov_a_a2 = cov_coeff[1, 0]
        cov_ac = cov_coeff[1, 2]
        cov_a2_c = cov_coeff[0, 2]
        var_a3 = 0.0
    else:
        delta = adc214 - adc210
        # Check for sufficient peak separation to avoid numerical instability
        min_peak_sep = cal_cfg.get("min_peak_separation_adc", _MIN_PEAK_SEPARATION_ADC_DEFAULT)
        if abs(delta) < min_peak_sep:
            raise ValueError(
                f"Insufficient peak separation: Po214-Po210 delta = {delta:.1f} ADC channels "
                f"(minimum {min_peak_sep:.1f} required)"
            )
        var_a = (a / delta) ** 2 * (mu_err_210**2 + mu_err_214**2)
        var_c = (a * adc214 / delta) ** 2 * mu_err_210**2 + (
            a * adc210 / delta
        ) ** 2 * mu_err_214**2
        # Covariance from error propagation: second term should be subtracted
        cov_ac = (a**2 / delta**2) * (
            adc214 * mu_err_210**2 - adc210 * mu_err_214**2
        )
        var_a2 = 0.0
        cov_a_a2 = 0.0
        cov_a2_c = 0.0
        var_a3 = 0.0

    # 8) Convert sigma_ADC -> sigma_E (MeV) using local derivative at the Po-214 peak.
    slope_local = 3 * a3 * adc214**2 + 2 * a2 * adc214 + a
    sigma_adc214 = peak_fits["Po214"]["sigma_adc"]
    sigma_E = abs(slope_local) * sigma_adc214

    var_slope_local = (
        var_a
        + (2 * adc214) ** 2 * var_a2
        + 2 * (2 * adc214) * cov_a_a2
        + (3 * adc214**2) ** 2 * var_a3
    )
    var_sigma_adc = peak_fits["Po214"]["covariance"][2][2]
    var_sigma_E = (sigma_adc214**2) * var_slope_local + (
        abs(slope_local)**2
    ) * var_sigma_adc
    dsigma_E = float(np.sqrt(max(var_sigma_E, 0.0)))

    # 8) Build result object:
    if cubic:
        coeffs = [c, a, a2, a3]
        cov = np.zeros((4, 4))
        cov[0, 0] = var_c
        cov[1, 1] = var_a
        cov[2, 2] = var_a2
        cov[3, 3] = var_a3
        cov[0, 1] = cov[1, 0] = cov_ac
        cov[0, 2] = cov[2, 0] = cov_a2_c
        cov[1, 2] = cov[2, 1] = cov_a_a2
        # Fill cubic cross-terms from full covariance
        cov[0, 3] = cov[3, 0] = cov_coeff[3, 0]  # c-a3
        cov[1, 3] = cov[3, 1] = cov_coeff[2, 0]  # a-a3
        cov[2, 3] = cov[3, 2] = cov_coeff[1, 0]  # a2-a3
    elif quadratic:
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
    """Return calibration constants for ``adc_values`` using ``config``."""
    cal_cfg = config.get("calibration", {})
    slope = cal_cfg.get("slope_mev_per_ch", cal_cfg.get("slope_MeV_per_ch"))
    float_slope = cal_cfg.get("float_slope", False)

    if slope is not None and not float_slope:
        if cal_cfg.get("use_two_point", False):
            try:
                from .calibrate import intercept_fit_two_point as _if2p
            except ImportError:  # pragma: no cover - fallback for root imports
                from calibrate import intercept_fit_two_point as _if2p
            try:
                return _if2p(adc_values, config)
            except RuntimeError as exc:
                warning_msg = None
                if "No candidate peak found" in str(exc):
                    warning_msg = (
                        "Two-point calibration failed to find both peaks; "
                        "falling back to one-point intercept-only"
                    )
                return _fallback_to_fixed_slope(adc_values, config, exc, warning_msg)
        return fixed_slope_calibration(adc_values, config)

    cfg = config if slope is None or not float_slope else deepcopy(config)
    if slope is not None and float_slope:
        if abs(slope) < 1e-15:
            raise ValueError("slope_mev_per_ch must be nonzero when float_slope is enabled")
        energies = {**DEFAULT_KNOWN_ENERGIES, **cal_cfg.get("known_energies", {})}
        intercept = cal_cfg.get("intercept_mev", cal_cfg.get("intercept_MeV", 0.0))
        cfg.setdefault("calibration", {})["nominal_adc"] = {
            iso: int(round((energies[iso] - intercept) / slope)) for iso in energies
        }
    try:
        return calibrate_run(adc_values, cfg)
    except RuntimeError as exc:
        if slope is not None:
            warning_msg = None
            if "No candidate peak found" in str(exc):
                warning_msg = (
                    "Two-point calibration failed to find both peaks; "
                    "falling back to one-point intercept-only"
                )
            return _fallback_to_fixed_slope(adc_values, config, exc, warning_msg)
        raise


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
            # A low prominence threshold helps reliably auto-find the
            # secondary anchor peak. Higher values (e.g. 20) proved too
            # strict and caused peak detection failures in some spectra.
            # Restore a more permissive default (matching the original
            # behaviour) by dropping the prominence to 5.
            "peak_prominence": 5,
            "peak_width": 3,
            "nominal_adc": nominal_adc,
            # parameter renamed to ``peak_search_radius`` to match config.json
            "peak_search_radius": peak_search_radius,
            "fit_window_adc": 50,
            "use_emg": False,
            "init_sigma_adc": 10.0,
            "init_tau_adc": 1.0,
            "sanity_tolerance_mev": 0.5,
            # Include defaults for configurable bounds
            "min_peak_separation_adc": _MIN_PEAK_SEPARATION_ADC_DEFAULT,
            "gaussian_sigma_upper_bound": _GAUSSIAN_SIGMA_UPPER_BOUND_DEFAULT,
            "emg_sigma_upper_bound": _EMG_SIGMA_UPPER_BOUND_DEFAULT,
            "emg_tau_upper_bound": _EMG_TAU_UPPER_BOUND_DEFAULT,
            "quadratic_residual_threshold_mev": _QUADRATIC_RESIDUAL_THRESHOLD_DEFAULT,
            "jacobian_epsilon": _JACOBIAN_EPSILON_DEFAULT,
        }
    }

    if nominal_adc is not None:
        config["calibration"]["nominal_adc"] = dict(nominal_adc)

    # Run calibration with custom histogram binning
    return calibrate_run(adc_arr, config, hist_bins=hist_bins)


# Deprecated alias for backward compatibility
def intercept_fit_two_point(*args, **kwargs):
    """Deprecated alias for :func:`calibrate.intercept_fit_two_point`."""
    warnings.warn(
        "rmtest.calibration.intercept_fit_two_point is deprecated; use rmtest.calibrate.intercept_fit_two_point instead.",
        DeprecationWarning,
        stacklevel=2,
    )
    try:
        from .calibrate import intercept_fit_two_point as _if2p
    except ImportError:  # pragma: no cover - fallback for root imports
        from calibrate import intercept_fit_two_point as _if2p
    return _if2p(*args, **kwargs)

__all__ = [
    "CalibrationResult",
    "two_point_calibration",
    "intercept_fit_two_point",
    "fixed_slope_calibration",
    "apply_calibration",
    "calibrate_run",
    "derive_calibration_constants",
    "derive_calibration_constants_auto",
    "configure_emg",
]
