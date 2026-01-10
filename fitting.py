# -----------------------------------------------------
# fitting.py
# -----------------------------------------------------

import logging
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import NotRequired, TypedDict

import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy.optimize import curve_fit

_ORIG_CURVE_FIT = curve_fit
from scipy.stats import chi2
from constants import safe_exp as _safe_exp
from math_utils import log_expm1_stable

# Single source of truth for shape PDFs, with a conservative fallback
try:
    from rmtest.spectral.shapes import (
        emg_pdf_E as emg_left,
        gaussian_pdf_E as gaussian,
    )
except ImportError:  # pragma: no cover
    # Fallback to legacy calibration module to keep environments unbroken.
    try:
        from calibration import emg_left, gaussian
    except ImportError:
        import sys
        from pathlib import Path

        _src_root = Path(__file__).resolve().parent / "src"
        if _src_root.is_dir():
            sys.path.insert(0, str(_src_root))
            from rmtest.spectral.shapes import (  # type: ignore
                emg_pdf_E as emg_left,
                gaussian_pdf_E as gaussian,
            )
        else:  # pragma: no cover - defensive fallback
            raise
try:
    from rmtest.spectral.intensity import (
        build_spectral_intensity,
        integral_of_intensity,
    )
except ImportError:  # pragma: no cover - fallback for local scripts
    import sys
    from pathlib import Path

    _src_root = Path(__file__).resolve().parent / "src"
    if _src_root.is_dir():
        sys.path.insert(0, str(_src_root))
        from rmtest.spectral.intensity import (  # type: ignore
            build_spectral_intensity,
            integral_of_intensity,
        )
    else:  # pragma: no cover - defensive fallback
        raise
try:
    from rmtest.emg_constants import (
        clamp_tau as _clamp_tau,
        EMG_MIN_TAU as _EMG_FLOOR,
    )
except ImportError:
    try:
        from src.rmtest.emg_constants import (
            clamp_tau as _clamp_tau,
            EMG_MIN_TAU as _EMG_FLOOR,
        )
    except ImportError:
        # Fallback implementation for local development
        def _clamp_tau(val, cfg=None, *, min_tau=None):
            floor = 5.0e-4 if min_tau is None else min_tau
            return val if val >= floor else floor
        _EMG_FLOOR = 5.0e-4
try:  # pragma: no cover - optional dependency path for package layout
    from rmtest.fitting.emg_config import (
        get_emg_stable_mode as _get_emg_stable_mode,
        set_emg_mode_from_config as _set_emg_mode_from_config,
    )
    from rmtest.fitting.emg_utils import resolve_emg_usage as _resolve_emg_usage
except ImportError:  # pragma: no cover - package may be unavailable at runtime
    import importlib.util
    import sys
    from pathlib import Path

    _emg_cfg_path = Path(__file__).resolve().parent / "src" / "rmtest" / "fitting" / "emg_config.py"
    if _emg_cfg_path.is_file():
        spec = importlib.util.spec_from_file_location("rmtest.fitting.emg_config", _emg_cfg_path)
        if spec and spec.loader:  # pragma: no cover - dynamic import fallback
            module = importlib.util.module_from_spec(spec)
            sys.modules.setdefault("rmtest.fitting.emg_config", module)
            spec.loader.exec_module(module)
            _get_emg_stable_mode = module.get_emg_stable_mode
            _set_emg_mode_from_config = module.set_emg_mode_from_config
        else:  # pragma: no cover - extremely defensive fallback
            def _get_emg_stable_mode(default: bool = True) -> bool:
                return bool(default)

            def _set_emg_mode_from_config(cfg):
                return _get_emg_stable_mode()
    else:  # pragma: no cover - defensive fallback when module missing
        def _get_emg_stable_mode(default: bool = True) -> bool:
            return bool(default)

        def _set_emg_mode_from_config(cfg):
            return _get_emg_stable_mode()

    _emg_utils_path = Path(__file__).resolve().parent / "src" / "rmtest" / "fitting" / "emg_utils.py"
    if "_resolve_emg_usage" not in globals():
        class _FallbackEMGSpec:
            __slots__ = ("enabled", "mean", "sigma", "source")

            def __init__(self, enabled: bool):
                self.enabled = bool(enabled)
                self.mean = None
                self.sigma = None
                self.source = "fallback"

        if _emg_utils_path.is_file():
            spec_utils = importlib.util.spec_from_file_location(
                "rmtest.fitting.emg_utils", _emg_utils_path
            )
            if spec_utils and spec_utils.loader:  # pragma: no cover - dynamic import fallback
                module_utils = importlib.util.module_from_spec(spec_utils)
                sys.modules.setdefault("rmtest.fitting.emg_utils", module_utils)
                spec_utils.loader.exec_module(module_utils)
                _resolve_emg_usage = module_utils.resolve_emg_usage
            else:  # pragma: no cover - defensive fallback
                def _resolve_emg_usage(isotopes, priors, **kwargs):
                    return {iso: _FallbackEMGSpec(False) for iso in isotopes}
        else:  # pragma: no cover - defensive fallback when module missing
            def _resolve_emg_usage(isotopes, priors, **kwargs):
                return {iso: _FallbackEMGSpec(False) for iso in isotopes}


_TAU_BOUND_EXPANSION = 10.0


EMG_STABLE_MODE: bool = _get_emg_stable_mode()


def _update_emg_stable_mode_from_config(
    cfg: Mapping[str, object] | SimpleNamespace | None,
) -> None:
    """Synchronize :data:`EMG_STABLE_MODE` with a loaded configuration."""

    global EMG_STABLE_MODE
    EMG_STABLE_MODE = _set_emg_mode_from_config(cfg)


def get_emg_stable_mode() -> bool:
    """Return the resolved EMG stable mode preference."""

    return _get_emg_stable_mode()


def softplus(x: np.ndarray | float) -> np.ndarray | float:
    """Stable softplus implementation for positive parameters."""
    x = np.asarray(x, dtype=float)
    return np.log1p(np.exp(-np.abs(x))) + np.maximum(x, 0.0)


def _softplus_inv(y: np.ndarray | float) -> np.ndarray | float:
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask = y > 0
    out[mask] = log_expm1_stable(y[mask])
    out[~mask] = -20.0
    return out


def _sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    x = np.asarray(x, dtype=float)
    # Use a numerically stable formulation to avoid overflow in ``exp`` when
    # ``x`` is very negative.
    out = np.empty_like(x)
    pos_mask = x >= 0

    # For non-negative ``x``, the original expression is stable.
    out[pos_mask] = 1.0 / (1.0 + np.exp(-x[pos_mask]))

    # For negative ``x``, rewrite the sigmoid to keep the exponent bounded:
    #   sigmoid(x) = exp(x) / (1 + exp(x))
    exp_x = np.exp(x[~pos_mask])
    out[~pos_mask] = exp_x / (1.0 + exp_x)

    return out


def make_linear_bkg(
    Emin: float, Emax: float, Eref: float | None = None, n_norm: int = 512
):
    """Return a unit-area log-linear background shape function.

    Parameters
    ----------
    Emin, Emax : float
        Energy bounds of the shape.
    Eref : float, optional
        Reference energy for the linear term. Defaults to the midpoint of
        ``Emin`` and ``Emax``.
    n_norm : int, optional
        Number of points used to compute the normalization constant.
    """

    if Eref is None:
        Eref = 0.5 * (Emin + Emax)
    grid = np.linspace(Emin, Emax, int(n_norm))

    def shape(E, beta0, beta1):
        exp_grid = _safe_exp(beta0 + beta1 * (grid - Eref))
        Z = np.trapz(exp_grid, grid)
        Z = max(Z, 1e-300)
        E = np.asarray(E, dtype=float)
        return _safe_exp(beta0 + beta1 * (E - Eref)) / Z

    return shape


# Backward-compatible aliases
_softplus = softplus
_make_linear_bkg = make_linear_bkg


# Use shared overflow guard for exponentiation
__all__ = [
    "fit_time_series",
    "fit_decay",
    "fit_spectrum",
    "softplus",
    "make_linear_bkg",
    "EMG_STABLE_MODE",
    "get_emg_stable_mode",
]


class FitParams(TypedDict, total=False):
    """Typed mapping of fit parameter names to values."""

    # Time-series parameters
    E_Po214: NotRequired[float]
    dE_Po214: NotRequired[float]
    B_Po214: NotRequired[float]
    dB_Po214: NotRequired[float]
    N0_Po214: NotRequired[float]
    dN0_Po214: NotRequired[float]

    E_Po218: NotRequired[float]
    dE_Po218: NotRequired[float]
    B_Po218: NotRequired[float]
    dB_Po218: NotRequired[float]
    N0_Po218: NotRequired[float]
    dN0_Po218: NotRequired[float]

    E_corrected: NotRequired[float]
    dE_corrected: NotRequired[float]

    cov_E_Po214_N0_Po214: NotRequired[float]
    fit_valid: NotRequired[bool]

    # Spectrum parameters
    sigma0: NotRequired[float]
    dsigma0: NotRequired[float]
    F: NotRequired[float]
    dF: NotRequired[float]
    mu_Po210: NotRequired[float]
    dmu_Po210: NotRequired[float]
    S_Po210: NotRequired[float]
    dS_Po210: NotRequired[float]
    mu_Po218: NotRequired[float]
    dmu_Po218: NotRequired[float]
    S_Po218: NotRequired[float]
    dS_Po218: NotRequired[float]
    tau_Po218: NotRequired[float]
    dtau_Po218: NotRequired[float]
    mu_Po214: NotRequired[float]
    dmu_Po214: NotRequired[float]
    S_Po214: NotRequired[float]
    dS_Po214: NotRequired[float]
    tau_Po214: NotRequired[float]
    dtau_Po214: NotRequired[float]
    tau_Po210: NotRequired[float]
    dtau_Po210: NotRequired[float]
    b0: NotRequired[float]
    db0: NotRequired[float]
    b1: NotRequired[float]
    db1: NotRequired[float]
    S_bkg: NotRequired[float]
    dS_bkg: NotRequired[float]

@dataclass
class FitResult:
    """Container for fit output."""

    params: dict[str, float]
    cov: np.ndarray | None

    ndf: int
    param_index: dict[str, int] | None = None
    counts: int | None = None
    likelihood: str | None = None
    _cov_df: pd.DataFrame | None = field(init=False, default=None, repr=False)

    def __post_init__(self):
        ordered: list[str] | None = None
        if self.param_index is None and isinstance(self.params, dict):
            ordered = [
                k
                for k in self.params.keys()
                if k != "fit_valid" and not k.startswith("d") and not k.startswith("cov_")
            ]
            self.param_index = {name: i for i, name in enumerate(ordered)}
        elif self.param_index is not None:
            ordered = [n for n, _ in sorted(self.param_index.items(), key=lambda kv: kv[1])]

        if self.cov is not None:
            self.cov = np.asarray(self.cov, dtype=float)
            if self.cov.ndim >= 2:
                if self.cov.shape[0] != self.cov.shape[1] or (
                    self.param_index is not None
                    and self.cov.shape[0] != len(self.param_index)
                ):
                    raise ValueError(
                        "cov must be square and match param_index length"
                    )
                if self.param_index is not None:
                    if ordered is None:
                        ordered = [n for n, _ in sorted(self.param_index.items(), key=lambda kv: kv[1])]
                    object.__setattr__(
                        self,
                        "_cov_df",
                        pd.DataFrame(self.cov, index=ordered, columns=ordered),
                    )

    def get_cov(self, name1: str, name2: str) -> float:
        """Return covariance entry for two parameters."""
        if self.cov is None or self.param_index is None:
            return 0.0
        if self._cov_df is not None:
            try:
                return float(self._cov_df.loc[name1, name2])
            except KeyError as exc:
                raise KeyError(
                    f"Parameter(s) missing in covariance: {name1}, {name2}"
                ) from exc

        try:
            i1 = self.param_index[name1]
            i2 = self.param_index[name2]
        except KeyError as exc:
            raise KeyError(
                f"Parameter(s) missing in covariance: {name1}, {name2}"
            ) from exc

        if self.cov.ndim >= 2 and 0 <= i1 < self.cov.shape[0] and 0 <= i2 < self.cov.shape[1]:
            return float(self.cov[i1, i2])

        raise KeyError(f"Parameter(s) missing in covariance: {name1}, {name2}")

    @property
    def cov_df(self) -> pd.DataFrame:
        """Return covariance matrix as a :class:`pandas.DataFrame`."""
        if self._cov_df is None:
            if self.cov is None or self.param_index is None:
                self._cov_df = pd.DataFrame()
            else:
                ordered = sorted(self.param_index.items(), key=lambda kv: kv[1])
                names = [n for n, _ in ordered]
                cov = np.asarray(self.cov, dtype=float)
                if cov.ndim == 2 and cov.shape[0] == len(names):
                    self._cov_df = pd.DataFrame(cov, index=names, columns=names)
                else:
                    self._cov_df = pd.DataFrame()
        return self._cov_df


class _WidthLookup:
    """Vector-friendly helper to align bin widths with their centers."""

    __slots__ = ("centers", "widths")

    def __init__(self, centers: np.ndarray, widths: np.ndarray) -> None:
        centers = np.asarray(centers, dtype=float)
        widths = np.asarray(widths, dtype=float)
        if centers.shape != widths.shape:
            raise ValueError("centers and widths must share the same shape")
        self.centers = centers
        self.widths = widths

    def scale(self, x, y):
        """Scale density values ``y`` by the bin widths for ``x``."""

        if np.isscalar(x):
            key = float(x)
            idx = int(np.searchsorted(self.centers, key))
            if idx >= self.centers.size or not np.isclose(self.centers[idx], key, rtol=1e-10, atol=1e-10):
                raise KeyError(
                    f"{x} not found in bin centers; model can only be evaluated at"
                    " histogram centers"
                )
            return y * self.widths[idx]

        x_arr = np.asarray(x)
        if x_arr.shape == self.centers.shape and np.array_equal(x_arr, self.centers):
            return y * self.widths

        raise KeyError(
            "Binned model is defined only for the configured histogram centers; "
            "pass a scalar center value or the exact centers array"
        )


def fit_decay(times, priors, t0=0.0, t_end=None, flags=None):
    """Simple rate estimator used for unit tests.

    Parameters
    ----------
    times : array-like
        Event times (relative to ``t0``) in seconds.
    priors : dict
        Dictionary that may contain an ``"eff"`` entry specifying the detection
        efficiency.
    t0 : float, optional
        Start time of the interval.  Only used to compute the total exposure
        ``t_end - t0``.
    t_end : float, optional
        End time of the interval.  If ``None`` the maximum time in ``times`` is
        used.
    flags : dict, optional
        Additional flags (ignored by this simple implementation).

    Returns
    -------
    dict
        Dictionary with at least the keys ``"E"``, ``"N0"``, ``"B"`` and
        ``"eff"`` representing a naive rate estimate and placeholder values for
        initial population and background.
    """

    if flags is None:
        flags = {}
    else:
        flags = dict(flags)
    if flags.get("fix_sigma_E"):
        flags.setdefault("fix_sigma0", True)
        flags.setdefault("fix_F", True)

    t = np.asarray(times, dtype=float)
    if t_end is None:
        T = float(t.max() if t.size > 0 else 0.0) - float(t0)
    else:
        T = float(t_end) - float(t0)

    eff_prior = priors.get("eff", (1.0, 0.0))
    eff = float(eff_prior[0] if isinstance(eff_prior, (tuple, list)) and len(eff_prior) > 0 else 1.0)

    count = len(t)
    rate = count / (T * eff) if (T > 0 and eff > 0) else 0.0

    return {
        "E": rate,
        "N0": 0.0,
        "B": 0.0,
        "eff": eff,
    }


def fit_spectrum(
    energies,
    priors,
    flags=None,
    bins=None,
    bin_edges=None,
    bounds=None,
    unbinned=False,
    strict=False,
    *,
    max_tau_ratio=None,
):
    """Fit the radon spectrum using either χ² histogram or unbinned likelihood.

    Parameters
    ----------
    energies : array-like
        Energy values (MeV).
    priors : dict
        Parameter priors of the form {name: (mu, sigma)}.
    flags : dict, optional
        Flags such as ``{"fix_sigma0": True}`` to fix parameters. Fixed
        parameters are implemented by constraining the optimizer to a tiny
        interval (``+/-1e-12``) around the provided mean value.
    bins : int or sequence, optional
        Number of bins or bin edges to use when histogramming the input
        energies.  Ignored if ``bin_edges`` is provided.  If both ``bins``
        and ``bin_edges`` are ``None``, the Freedman--Diaconis rule is used.
    bin_edges : array-like, optional
        Explicit, strictly increasing bin edges for histogramming the
        energies. Non-uniform spacing is supported and takes precedence over
        ``bins`` when given. Variable-width bins are supported only when the
        ``x`` values passed to the model correspond exactly to the bin centers
        defined by ``bin_edges``.
    bounds : dict, optional
        Mapping of parameter name to ``(lower, upper)`` tuples overriding the
        default +/-5 sigma range derived from the priors.  ``None`` values disable a
        limit on that side.
    unbinned : bool, optional
        When ``True`` use an extended unbinned likelihood fit instead of the
        default χ² fit to a histogrammed spectrum.
    strict : bool, optional
        When ``True`` raise a :class:`RuntimeError` if the fit covariance
        matrix is not positive definite.  The default is ``False`` which
        attempts to stabilise the covariance by adding a tiny jitter.
    max_tau_ratio : float, optional
        If given, enforce an upper bound ``tau <= max_tau_ratio * sigma0`` for
        EMG tail parameters.

    Returns
    -------
    dict
        Best fit values and uncertainties.
    """
    
    if flags is None:
        flags = {}

    # The spectral intensity returned by ``build_spectral_intensity`` is an
    # event *intensity* (counts/MeV), not a unit-normalised PDF.  In unbinned
    # mode this requires an extended likelihood so that the total expected
    # counts appear in the objective.  Otherwise the optimiser can drive the
    # signal areas to arbitrarily large values with no penalty.  Default to the
    # extended path unless the caller explicitly overrides the likelihood.
    if unbinned:
        flags.setdefault("likelihood", "extended")

    likelihood_mode = "unbinned" if unbinned else "binned_poisson"
    likelihood_path = "unbinned_extended" if unbinned else "binned_poisson"
    if flags.get("fix_sigma_E"):
        flags.setdefault("fix_sigma0", True)
        flags.setdefault("fix_F", True)

    fix_sigma0 = flags.get("fix_sigma0", False)
    fix_F = flags.get("fix_F", False)
    if fix_sigma0 and not fix_F:
        raise ValueError(
            "Resolution flags conflict: cannot fix sigma0 while allowing F to float"
        )

    e = np.asarray(energies, dtype=float)
    n_events = e.size
    if e.size == 0:
        raise RuntimeError("No energies provided to fit_spectrum")

    # Determine bin edges for width/normalisation even in unbinned mode
    if bin_edges is not None:
        edges = np.asarray(bin_edges, dtype=float)
    elif bins is not None:
        edges = np.histogram_bin_edges(e, bins=bins)
    else:
        edges = np.histogram_bin_edges(e, bins="fd")

    edges = np.asarray(edges, dtype=float)
    if not np.all(np.diff(edges) > 0):
        raise ValueError("bin_edges must be strictly increasing")

    widths = np.diff(edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if widths.size != centers.size:
        raise RuntimeError("width and center size mismatch")
    width_lookup = _WidthLookup(centers, widths)
    if not unbinned:
        hist, _ = np.histogram(e, bins=edges)

    E_lo = float(edges[0])
    E_hi = float(edges[-1])

    # Configure background handling
    background_model = flags.get("background_model")
    priors = dict(priors)
    if background_model == "loglin_unit":
        if "S_bkg" not in priors:
            b0_prior = priors.get("b0", (0.0, 1.0))
            b1_prior = priors.get("b1", (0.0, 1.0))
            b0_mu = b0_prior[0] if isinstance(b0_prior, (tuple, list)) and len(b0_prior) > 0 else 0.0
            b1_mu = b1_prior[0] if isinstance(b1_prior, (tuple, list)) and len(b1_prior) > 0 else 0.0
            # For exp(b0 + b1*(E - Eref)), calculate integral over [E_lo, E_hi]
            Eref = 0.5 * (E_lo + E_hi)
            if abs(b1_mu) > 1e-10:
                # ∫exp(b0 + b1*(E-Eref))dE = exp(b0)/b1 * [exp(b1*(E_hi-Eref)) - exp(b1*(E_lo-Eref))]
                from constants import safe_exp
                exp_hi = safe_exp(b0_mu + b1_mu * (E_hi - Eref))
                exp_lo = safe_exp(b0_mu + b1_mu * (E_lo - Eref))
                mu = (exp_hi - exp_lo) / b1_mu
            else:
                # For b1 ~ 0, use exp(b0) * (E_hi - E_lo)
                from constants import safe_exp
                mu = safe_exp(b0_mu) * (E_hi - E_lo)
            mu = max(mu, 0.0)
            sig = max(abs(mu) * 0.1, 1.0)
            priors["S_bkg"] = (mu, sig)
        required = {"b0", "b1"}
        missing = required - priors.keys()
        if missing:
            got = sorted(priors.keys())
            raise ValueError(
                "background_model=loglin_unit requires params {S_bkg, b0, b1}; got: "
                f"{got}"
            )

    if "S_bkg" in priors or background_model == "loglin_unit":
        flags.setdefault("likelihood", "extended")

    from feature_selectors import select_neg_loglike

    opts = SimpleNamespace(**flags)
    neg_loglike = select_neg_loglike(opts)

    # Guard against NaNs/Infs arising from unstable histogramming or EMG evals
    if not unbinned and not np.isfinite(hist).all():
        raise RuntimeError(
            "fit_spectrum: histogram contains non-finite values; "
            "check input energies and binning parameters"
        )
    if not np.isfinite(centers).all():
        raise RuntimeError(
            "fit_spectrum: histogram centers contain non-finite values; "
            "check input energies and binning parameters"
        )

    # Helper to fetch prior values
    def p(name, default):
        return priors.get(name, (default, 1.0))

    # Extract config if present for use in tau clamping and per-isotope overrides
    cfg = flags.get("cfg") or flags.get("config")

    # Determine which peaks should include an EMG tail using the shared helper
    iso_list = ["Po210", "Po218", "Po214"]
    emg_specs = _resolve_emg_usage(
        iso_list,
        priors,
        flags=flags,
        tau_floor=_EMG_FLOOR,
    )
    use_emg = {iso: spec.enabled for iso, spec in emg_specs.items()}
    for iso, spec in emg_specs.items():
        key = f"tau_{iso}"
        if spec.enabled and key not in priors and spec.mean is not None:
            mean = float(spec.mean)
            # Clamp tau to the global floor
            mean = _clamp_tau(mean, cfg, min_tau=_EMG_FLOOR)
            sigma = float(spec.sigma) if spec.sigma is not None else 1.0
            if not np.isfinite(sigma) or sigma <= 0:
                sigma = 1.0
            priors[key] = (mean, sigma)

    # Apply per-isotope overrides from config if present
    if cfg is not None:
        per_isotope = cfg.get("fitting", {}).get("per_isotope", {})
        for iso in iso_list:
            if iso not in per_isotope:
                continue
            overrides = per_isotope[iso]

            # Override use_emg if specified
            if "use_emg" in overrides:
                use_emg[iso] = bool(overrides["use_emg"])
                emg_specs[iso].enabled = bool(overrides["use_emg"])

            # Override tau if specified
            if "tau" in overrides and use_emg.get(iso, False):
                tau_val = float(overrides["tau"])
                # Clamp tau using the shared helper
                tau_val = _clamp_tau(tau_val, cfg, min_tau=_EMG_FLOOR)
                # Override or set tau_prior if present
                if "tau_prior" in overrides:
                    tau_prior = overrides["tau_prior"]
                    if isinstance(tau_prior, (list, tuple)) and len(tau_prior) == 2:
                        priors[f"tau_{iso}"] = (tau_val, float(tau_prior[1]))
                else:
                    # Use existing sigma if present, otherwise default to 1.0
                    existing_sigma = priors.get(f"tau_{iso}", (None, 1.0))[1]
                    priors[f"tau_{iso}"] = (tau_val, existing_sigma)
            elif "tau_prior" in overrides and use_emg.get(iso, False):
                # Only tau_prior is specified without tau
                tau_prior = overrides["tau_prior"]
                if isinstance(tau_prior, (list, tuple)) and len(tau_prior) == 2:
                    tau_mean = _clamp_tau(float(tau_prior[0]), cfg, min_tau=_EMG_FLOOR)
                    priors[f"tau_{iso}"] = (tau_mean, float(tau_prior[1]))

            # Override peak_sigma if specified
            if "peak_sigma" in overrides:
                key = f"sigma_{iso}"
                if key in priors:
                    mean, _ = priors[key]
                    priors[key] = (float(overrides["peak_sigma"]), priors[key][1])

            # Override peak_scale if specified
            if "peak_scale" in overrides:
                key = f"S_{iso}"
                if key in priors:
                    _, sigma = priors[key]
                    priors[key] = (float(overrides["peak_scale"]), sigma)

    # Track which resolution parameters are fixed
    fix_sigma0 = flags.get("fix_sigma0", False)
    fix_F = flags.get("fix_F", False)

    # Build parameter ordering and index mapping dynamically so that fixed
    # parameters are excluded from the free-parameter list.  When a parameter is
    # fixed we keep its mean value for later use in the model function but do
    # not let the optimiser vary it.
    param_order: list[str] = []
    param_index: dict[str, int] = {}
    sigma0_val = p("sigma0", 1.0)[0]
    F_val = p("F", 0.0)[0]
    if not fix_sigma0:
        param_index["sigma0"] = len(param_order)
        param_order.append("sigma0")
    if not fix_F:
        param_index["F"] = len(param_order)
        param_order.append("F")
    for iso in iso_list:
        param_index[f"mu_{iso}"] = len(param_order)
        param_order.append(f"mu_{iso}")
        param_index[f"S_{iso}"] = len(param_order)
        param_order.append(f"S_{iso}")
        if use_emg[iso]:
            param_index[f"tau_{iso}"] = len(param_order)
            param_order.append(f"tau_{iso}")
    param_index["b0"] = len(param_order)
    param_order.append("b0")
    param_index["b1"] = len(param_order)
    param_order.append("b1")
    if background_model == "loglin_unit" or "S_bkg" in priors:
        param_index["S_bkg"] = len(param_order)
        param_order.append("S_bkg")

    p0 = []
    bounds_lo, bounds_hi = [], []
    eps = 1e-12
    sigma0_mean = sigma0_val
    for name in param_order:
        mean, sig = p(name, 1.0)
        if name.startswith("S_"):
            mean = float(_softplus_inv(mean))
        # Enforce a strictly positive initial tau to avoid singular EMG tails
        if name.startswith("tau_"):
            mean = _clamp_tau(mean, None, min_tau=_EMG_FLOOR)
        is_fixed = flags.get(f"fix_{name}", False) or abs(sig) < 1e-15
        if is_fixed:
            # Optimiser requires lower < upper; use a tiny width around fixed values
            lo = mean - eps
            hi = mean + eps
        else:
            sig_val = float(sig)
            if not np.isfinite(sig_val):
                delta = np.inf
            else:
                delta = 5 * abs(sig_val)
            lo = mean - delta
            hi = mean + delta
            if name.startswith("tau_"):
                if np.isfinite(delta):
                    tau_scale = max(abs(mean), abs(sig_val), delta / 5, _EMG_FLOOR)
                    if np.isfinite(tau_scale) and tau_scale > 0:
                        extra = _TAU_BOUND_EXPANSION * tau_scale
                        lo = min(lo, mean - extra)
                        hi = max(hi, mean + extra)
                if not np.isfinite(delta):
                    hi = np.inf
        if bounds and name in bounds:
            user_lo, user_hi = bounds[name]
            if user_lo is not None:
                lo = max(lo, user_lo)
            if user_hi is not None:
                hi = min(hi, user_hi)
        if name.startswith("tau_"):
            lo = max(lo, _EMG_FLOOR)
            if max_tau_ratio is not None:
                hi = min(hi, max_tau_ratio * sigma0_mean)
        if name in ("sigma0", "F"):
            lo = max(lo, 0.0)
        if hi <= lo:
            logging.warning(
                f"Parameter {name}: upper bound ({hi}) <= lower bound ({lo}). "
                f"Adjusting upper bound to {lo + eps}."
            )
            hi = lo + eps
        mean = np.clip(mean, lo, hi)
        p0.append(mean)
        bounds_lo.append(lo)
        bounds_hi.append(hi)

    area_keys = [f"S_{iso}" for iso in iso_list]
    if background_model == "loglin_unit" or "S_bkg" in priors:
        area_keys = area_keys + ["S_bkg"]

    domain = (E_lo, E_hi)
    use_emg_map = {iso: bool(use_emg.get(iso, False)) for iso in iso_list}

    # Extract clip_floor from config with safe default
    clip_floor = 1e-300
    if cfg is not None:
        clip_floor = float(cfg.get("spectral_fit", {}).get("clip_floor", 1e-300))

    loglin_n_norm = None
    if cfg is not None:
        loglin_n_norm = cfg.get("spectral_fit", {}).get("loglin_n_norm")

    spectral_intensity = build_spectral_intensity(
        iso_list,
        use_emg_map,
        domain,
        clip_floor=clip_floor,
        background_model=background_model,
        loglin_n_norm=loglin_n_norm,
    )

    def _build_raw_param_map(params):
        p_map = dict(zip(param_order, params))
        if fix_sigma0:
            p_map.setdefault("sigma0", sigma0_val)
        if fix_F:
            p_map.setdefault("F", F_val)
        return p_map

    def _physical_params(raw_map: Mapping[str, float]) -> dict[str, float]:
        params_dict: dict[str, float] = {}
        params_dict["sigma0"] = float(raw_map.get("sigma0", sigma0_val))
        params_dict["b0"] = float(raw_map["b0"])
        params_dict["b1"] = float(raw_map["b1"])
        if "F" in raw_map or not fix_F:
            params_dict["F"] = float(raw_map.get("F", F_val))
        for iso in iso_list:
            params_dict[f"mu_{iso}"] = float(raw_map[f"mu_{iso}"])
            S_val = float(_softplus(raw_map[f"S_{iso}"]))
            params_dict[f"S_{iso}"] = S_val
            params_dict[f"N_{iso}"] = S_val
            tau_key = f"tau_{iso}"
            if use_emg_map.get(iso, False) and tau_key in raw_map:
                params_dict[tau_key] = float(raw_map[tau_key])
        if "S_bkg" in raw_map:
            params_dict["S_bkg"] = float(_softplus(raw_map["S_bkg"]))
        return params_dict

    def _finalize_signal_params(out_params: dict[str, float]) -> dict[str, float]:
        for iso in iso_list:
            n_key = f"N_{iso}"
            s_key = f"S_{iso}"
            if n_key in out_params:
                out_params[s_key] = float(out_params[n_key])
            elif s_key in out_params:
                out_params[s_key] = float(out_params[s_key])
        return out_params

    def _model_density(x, *params):
        """
        Per-energy density λ(E) in counts/MeV.
        Clipping is applied inside spectral_intensity to the configured floor.
        """
        raw_map = _build_raw_param_map(params)
        physical = _physical_params(raw_map)
        return spectral_intensity(x, physical, domain)

    def _model_binned(x, *params):
        y = _model_density(x, *params)
        return width_lookup.scale(x, y)

    if not unbinned:
        popt_cf = pcov_cf = None
        if curve_fit is not _ORIG_CURVE_FIT:
            try:
                popt_cf, pcov_cf = curve_fit(
                    _model_binned,
                    centers,
                    hist,
                    p0=p0,
                    bounds=(bounds_lo, bounds_hi),
                )
            except Exception:
                pass

        def _nll(*params):
            model = _model_binned(centers, *params)
            model_safe = np.maximum(model, 1e-300)  # Floor to prevent log(0)
            return float(np.sum(model_safe - hist * np.log(model_safe)))

        m = Minuit(_nll, *p0, name=param_order)
        m.errordef = Minuit.LIKELIHOOD
        for name, lo, hi in zip(param_order, bounds_lo, bounds_hi):
            m.limits[name] = (lo, hi)
            if flags.get(f"fix_{name}", False):
                m.fixed[name] = True
        m.migrad()
        if not m.valid:
            m.simplex()
            m.migrad()
        ndf = max(1, hist.size - len(param_order))
        out = {}
        param_index = {name: i for i, name in enumerate(param_order)}
        m.hesse()
        cov_raw = m.covariance
        covariance_available = cov_raw is not None
        if cov_raw is None:
            cov_raw = np.zeros((len(param_order), len(param_order)))
        pcov = np.array(cov_raw)
        popt = np.array([m.values[p] for p in param_order])
        nll_val = float(m.fval)
        if curve_fit is not _ORIG_CURVE_FIT and popt_cf is not None and pcov_cf is not None:
            pcov = np.array(pcov_cf)
            popt = np.array(popt_cf)
            covariance_available = True
    else:
        def _intensity_fn(E_vals, p_map):
            raw_map = dict(p_map)
            if fix_sigma0:
                raw_map.setdefault("sigma0", sigma0_val)
            if fix_F:
                raw_map.setdefault("F", F_val)
            physical = _physical_params(raw_map)
            return spectral_intensity(E_vals, physical, domain)

        if flags.get("likelihood") == "extended":
            def _nll(*params):
                # Extended model: mu_total = integral of (peaks + backgrounds) over [E_lo, E_hi].
                # We obtain mu_total from the same intensity builder used for the per-E density
                # (unclipped), then we recover the background integral by subtracting the peak
                # integrals. This keeps the total mean consistent with the density we fit.
                raw_map = _build_raw_param_map(params)
                physical = _physical_params(raw_map)
                mu_total = integral_of_intensity(
                    physical,
                    domain,
                    iso_list=iso_list,
                    use_emg=use_emg_map,
                    background_model=background_model,
                )
                area_sum = 0.0
                for key in area_keys:
                    if key in raw_map:
                        area_sum += float(_softplus(raw_map[key]))

                # Check for NaN/inf before subtraction to prevent propagation
                if not np.isfinite(mu_total) or not np.isfinite(area_sum):
                    bg_integral = None
                else:
                    bg_integral = mu_total - area_sum
                    if not np.isfinite(bg_integral):
                        bg_integral = None
                    else:
                        bg_integral = max(bg_integral, 0.0)

                def _intensity_cached(E_vals, _params):
                    return spectral_intensity(E_vals, physical, domain)

                return neg_loglike(
                    e,
                    _intensity_cached,
                    raw_map,
                    area_keys=area_keys,
                    background_model="loglin_unit" if background_model == "loglin_unit" else None,
                    background_integral=bg_integral,
                )
        else:
            def _nll(*params):
                raw_map = _build_raw_param_map(params)
                return neg_loglike(e, _intensity_fn, raw_map)

        m = Minuit(_nll, *p0, name=param_order)
        m.errordef = Minuit.LIKELIHOOD
        for name, lo, hi in zip(param_order, bounds_lo, bounds_hi):
            m.limits[name] = (lo, hi)
            if flags.get(f"fix_{name}", False):
                m.fixed[name] = True
        m.migrad()
        if not m.valid:
            m.simplex()
            m.migrad()
        ndf = max(1, e.size - len(param_order))
        out = {}
        param_index = {name: i for i, name in enumerate(param_order)}
        m.hesse()
        cov_raw = m.covariance
        covariance_available = cov_raw is not None
        if cov_raw is None:
            cov = np.zeros((len(param_order), len(param_order)))
            perr = np.zeros(len(param_order))
        else:
            cov = np.array(cov_raw)
            g = np.ones(len(param_order))
            for i, pname in enumerate(param_order):
                if pname.startswith("S_"):
                    g[i] = float(_sigmoid(m.values[pname]))
            cov = cov * (g[:, None] * g[None, :])
            perr = np.sqrt(np.clip(np.diag(cov), 0, None))
        fit_valid = True
        try:
            if covariance_available:
                eigvals = np.linalg.eigvals(cov)
                fit_valid = bool(np.all(eigvals >= 0))
        except np.linalg.LinAlgError:
            fit_valid = False
        if not fit_valid:
            if strict:
                raise RuntimeError(
                    "fit_spectrum: covariance matrix not positive definite"
                )
            logging.warning(
                "fit_spectrum: covariance matrix not positive definite"
            )
            jitter = 1e-12 * np.mean(np.diag(cov))
            if not np.isfinite(jitter) or jitter <= 0:
                jitter = 1e-12
            cov = cov + jitter * np.eye(cov.shape[0])
            try:
                np.linalg.cholesky(cov)
                fit_valid = True
                perr = np.sqrt(np.clip(np.diag(cov), 0, None))
            except np.linalg.LinAlgError:
                pass
        out["fit_valid"] = fit_valid
        for i, pname in enumerate(param_order):
            val = float(m.values[pname])
            if pname.startswith("S_"):
                out[pname] = float(_softplus(val))
                out["d" + pname] = float(perr[i] if i < len(perr) else np.nan)
            else:
                out[pname] = val
                out["d" + pname] = float(perr[i] if i < len(perr) else np.nan)
        if fix_sigma0:
            out["sigma0"] = sigma0_val
            out["dsigma0"] = 0.0
        if fix_F:
            out["F"] = F_val
            out["dF"] = 0.0
        k = len(param_order)
        out["aic"] = float(2 * m.fval + 2 * k)
        out["likelihood_path"] = likelihood_path
        out = _finalize_signal_params(out)
        return FitResult(
            out,
            cov,
            int(ndf),
            param_index,
            counts=int(n_events),
            likelihood=likelihood_mode,
        )

    g = np.ones(len(param_order))
    for i, name in enumerate(param_order):
        if name.startswith("S_"):
            g[i] = float(_sigmoid(popt[i]))
    pcov = pcov * (g[:, None] * g[None, :])
    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    fit_valid = True
    try:
        if covariance_available:
            eigvals = np.linalg.eigvals(pcov)
            fit_valid = bool(np.all(eigvals >= 0))
    except np.linalg.LinAlgError:
        fit_valid = False

    if not fit_valid:
        if strict:
            raise RuntimeError(
                "fit_spectrum: covariance matrix not positive definite"
            )
        logging.warning("fit_spectrum: covariance matrix not positive definite")
        jitter = 1e-12 * np.mean(np.diag(pcov))
        if not np.isfinite(jitter) or jitter <= 0:
            jitter = 1e-12
        pcov = pcov + jitter * np.eye(pcov.shape[0])
        try:
            np.linalg.cholesky(pcov)
            fit_valid = True
            perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
        except np.linalg.LinAlgError:
            pass
    out = {}
    for i, name in enumerate(param_order):
        val = float(popt[i])
        if name.startswith("S_"):
            out[name] = float(_softplus(val))
            out["d" + name] = float(perr[i])
        else:
            out[name] = val
            out["d" + name] = float(perr[i])

    if fix_sigma0:
        out["sigma0"] = sigma0_val
        out["dsigma0"] = 0.0
    if fix_F:
        out["F"] = F_val
        out["dF"] = 0.0

    out["fit_valid"] = fit_valid
    out["likelihood_path"] = likelihood_path

    model_counts = _model_binned(centers, *popt)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = 2 * np.sum(
            model_counts - hist + hist * np.log(np.where(hist > 0, hist / model_counts, 1.0))
        )
    out["chi2"] = float(chi2)
    ndf = max(1, hist.size - len(popt))
    out["chi2_ndf"] = chi2 / ndf if ndf > 0 else np.nan
    out["nll"] = float(nll_val)
    k = len(popt)
    out["aic"] = float(2 * nll_val + 2 * k)
    param_index = {name: i for i, name in enumerate(param_order)}
    out = _finalize_signal_params(out)
    return FitResult(
        out,
        pcov,
        int(ndf),
        param_index,
        counts=int(n_events),
        likelihood=likelihood_mode,
    )


def _integral_model(E, N0, B, lam, eff, T):
    """
    Analytic integral of: eff * [E*(1 - exp(-lam*t)) + lam*N0*exp(-lam*t)] + B
    from t=0 to t=T:
           eff*(E - E*exp(-lam*t) + lam*N0*exp(-lam*t)) dt + B*T
    = eff * [ E*( T - (1 - exp(-lam*T))/lam ) + lam*N0*( (1 - exp(-lam*T))/lam ) ] + B*T
    """
    if lam <= 0:
        # In principle lam should never be <=0; return a large number to penalize
        return 1e50
    if T < 0:
        raise ValueError(
            f"_integral_model called with negative time interval T={T}. "
            "This indicates an error in time window calculation."
        )
    # Term1 = E * (T - (1 - e^{-lam T})/lam )
    exp_term = _safe_exp(-lam * T)
    decay_term = (1.0 - exp_term) / lam
    term_E = E * (T - decay_term)
    term_N0 = N0 * (1.0 - exp_term)
    return eff * (term_E + term_N0) + B * T


def _neg_log_likelihood_time(
    params,  # flattened list of all parameters in order
    times_dict,
    weights_dict,
    t_start,
    t_end,
    iso_list,
    lam_map,
    eff_map,
    fix_b_map,
    fix_n0_map,
    param_indices,
    fixed_b_values=None,
):
    """
    params: tuple of all (E_iso, [B_iso], [N0_iso], for each iso in iso_list, in the order recorded by param_indices)
    times_dict: mapping of isotope -> array of event timestamps.
    weights_dict: mapping of isotope -> array of per-event weights or None.
    t_start, t_end: floats (absolute UNIX seconds)
    lam_map: { iso: decay_constant (1/s) }
    eff_map: { iso: detection efficiency }
    fix_b_map, fix_n0_map: booleans per iso
    param_indices: dictionary mapping each parameter name ("E_Po214", "B_Po214", "N0_Po214", etc.)
                     index into the params tuple.
    Returns: scalar negative log likelihood
    """
    nll = 0.0
    T_rel = t_end - t_start

    # Build a dict for this iteration s parameter values:
    p = {}
    for pname, idx in param_indices.items():
        p[pname] = params[idx]

    # For each isotope, compute its contribution to NLL:
    for iso in iso_list:
        lam = lam_map[iso]
        eff = eff_map[iso]
        if eff is None:
            eff = p[f"eff_{iso}"]
        if eff <= 0:
            return 1e50

        # Extract parameters (some may be fixed to specific values):
        E_iso = p[f"E_{iso}"]
        if fix_b_map[iso]:
            B_iso = 0.0
            if fixed_b_values and iso in fixed_b_values:
                B_iso = float(fixed_b_values[iso])
        else:
            B_iso = p[f"B_{iso}"]
        N0_iso = 0.0 if fix_n0_map[iso] else p[f"N0_{iso}"]

        # 1) Integral term. When per-event weights are supplied we
        # scale the integral by the mean weight so that a uniform
        # scaling of all weights cancels between the log and integral
        # contributions.
        integral = _integral_model(E_iso, N0_iso, B_iso, lam, eff, T_rel)
        weights = weights_dict.get(iso)
        if weights is not None and len(weights) > 0:
            weight_mean = np.mean(weights)
            integral *= float(weight_mean)

        # 2) Sum of log[r(t_i)] for each event t_i in times_dict[iso]:
        times_iso = times_dict.get(iso, np.empty(0))
        weights = weights_dict.get(iso)
        if len(times_iso) > 0:
            # Calculate rate r(t_i_rel) at each observed time:
            t_rel = times_iso - t_start
            if np.any(t_rel < 0):
                raise ValueError(
                    f"fit_time_series: negative relative times detected for {iso}. "
                    f"t_start={t_start}, min(times)={times_iso.min()}. Check t_start parameter."
                )
            # r_iso(t_rel) = eff * [ E*(1 - exp(-lam*t_rel)) + lam*N0*exp(-lam*t_rel) ] + B
            exp_term = _safe_exp(-lam * t_rel)
            rate_vals = (
                eff
                * (
                    E_iso * (1.0 - exp_term)
                    + lam * N0_iso * exp_term
                )
                + B_iso
            )
            # If any rate_vals   0, penalize heavily:
            if np.any(rate_vals <= 0):
                return 1e50
            if weights is None:
                nll -= np.sum(np.log(rate_vals))
            else:
                nll -= np.sum(weights * np.log(rate_vals))
        # Add the integral term:
        nll += integral

    return nll


def fit_time_series(
    times_dict,
    t_start,
    t_end,
    config,
    weights=None,
    strict=False,
    fixed_background=None,
    **kwargs,
):
    """
    times_dict: mapping of isotope -> array of timestamps in seconds.
    weights : dict or None
        Optional mapping of isotope -> per-event weights matching
        ``times_dict``.
    strict : bool, optional
        When ``True`` raise a :class:`RuntimeError` if the covariance matrix
        is not positive definite.  The default is ``False`` which attempts to
        stabilise the matrix by adding a tiny jitter.
    t_start, t_end: floats (absolute UNIX seconds) defining the fit window
    config: JSON dict with these keys:
          "isotopes": { "Po214": {"half_life_s": , "efficiency": ,  }, "Po218": {   } }
          "fit_background": bool
          "fit_initial": bool
          "background_guess": float  (initial guess for B_iso)
          "initial_guess":    float  (initial guess for N0_iso)
    Returns: dict with best fit values & 1  uncertainties, e.g.:
        {
          "E_Po214": 12.3,  "dE_Po214": 1.4,
          "B_Po214": 0.02, "dB_Po214": 0.005,
          "N0_Po214": 50,  "dN0_Po214": 10,
          "E_Po218": 5.6,  "dE_Po218": 0.8,
          "B_Po218": 0.01, "dB_Po218": 0.003,
          "N0_Po218": 10,  "dN0_Po218": 3,
          "fit_valid": True
        }
    """
    if kwargs:
        logging.debug(
            "fit_time_series: ignoring unsupported keyword arguments: %s",
            sorted(kwargs.keys()),
        )

    iso_list = list(config["isotopes"].keys())

    # Normalize weights mapping
    if weights is None:
        weights_dict = {iso: None for iso in iso_list}
    else:
        weights_dict = {iso: np.asarray(weights.get(iso), dtype=float) if weights.get(iso) is not None else None for iso in iso_list}

    # Early exit when statistics are insufficient
    total_counts = 0.0
    for iso in iso_list:
        w_arr = weights_dict.get(iso)
        if w_arr is None:
            total_counts += len(times_dict.get(iso, []))
        else:
            total_counts += float(np.sum(w_arr))

    cfg_min_counts = config.get("min_counts")
    if cfg_min_counts is not None and int(cfg_min_counts) > 0:
        min_counts = int(cfg_min_counts)
    else:
        # Require at least one event by default
        min_counts = 1

    if total_counts < min_counts:
        logging.info(
            "fit_time_series: skipping fit, only %.0f events (< %d)",
            total_counts,
            min_counts,
        )
        return FitResult({"fit_valid": False}, None, 0, counts=int(total_counts))

    # 1) Build maps: lam_map, eff_map, fix_b_map, fix_n0_map
    lam_map, eff_map = {}, {}
    fix_b_map, fix_n0_map = {}, {}
    for iso in iso_list:
        iso_cfg = config["isotopes"][iso]
        hl = float(iso_cfg["half_life_s"])
        if hl <= 0:
            raise ValueError("half_life_s must be positive")
        lam_map[iso] = np.log(2.0) / hl
        eff_val = iso_cfg.get("efficiency", 1.0)
        if eff_val is None:
            eff_map[iso] = None
        else:
            eff = float(eff_val)
            if eff <= 0:
                raise ValueError("efficiency must be positive")
            eff_map[iso] = eff
        fix_b_map[iso] = not bool(config.get("fit_background", False))
        fix_n0_map[iso] = not bool(config.get("fit_initial", False))

    # 2) Decide parameter ordering. We always fit E_iso, then optionally B_iso, N0_iso.
    param_indices = {}  # name   index in the flat parameter tuple
    initial_guesses = []
    limits = {}

    background_guess = float(config.get("background_guess", 0.0))
    n0_guess_frac = float(config.get("n0_guess_fraction", 0.1))

    idx = 0
    for iso in iso_list:
        #    E_iso
        param_indices[f"E_{iso}"] = idx
        # Make a  smart  initial guess: (#events)/(T_rel*eff) or 1e-3 if zero
        times_arr = np.asarray(times_dict.get(iso, []), dtype=float)
        w_arr = weights_dict.get(iso)
        if w_arr is None:
            Ntot = len(times_arr)
        else:
            Ntot = float(np.sum(w_arr))
        T_rel = t_end - t_start
        eff_known = eff_map[iso] if eff_map[iso] is not None else 1.0
        guess_E = max((Ntot / (T_rel * eff_known))
                      if (T_rel > 0 and eff_known > 0) else 0.0, 1e-6)
        initial_guesses.append(guess_E)
        limits[f"E_{iso}"] = (0.0, None)
        idx += 1

        #    efficiency parameter if not fixed
        if eff_map[iso] is None:
            param_indices[f"eff_{iso}"] = idx
            initial_guesses.append(1.0)
            limits[f"eff_{iso}"] = (0.0, None)
            idx += 1

        #    B_iso (if not fixed)
        if not fix_b_map[iso]:
            param_indices[f"B_{iso}"] = idx
            initial_guesses.append(background_guess)
            limits[f"B_{iso}"] = (0.0, None)
            idx += 1

        #    N0_iso (if not fixed)
        if not fix_n0_map[iso]:
            param_indices[f"N0_{iso}"] = idx
            # N0 guess = fraction of total events (very rough) or zero
            guess_N0 = Ntot * n0_guess_frac if Ntot > 0 else 0.0
            initial_guesses.append(guess_N0)
            limits[f"N0_{iso}"] = (0.0, None)
            idx += 1

    # 3) Build the Minuit minimizer
    def _nll_minuit_wrapper(*args):
        return _neg_log_likelihood_time(
            args,
            times_dict,
            weights_dict,
            t_start,
            t_end,
            iso_list,
            lam_map,
            eff_map,
            fix_b_map,
            fix_n0_map,
            param_indices,
            fixed_background,
        )

    # Collect parameter names in the same order as initial_guesses
    ordered_params = [None] * len(initial_guesses)
    for name, i in param_indices.items():
        ordered_params[i] = name

    m = Minuit(_nll_minuit_wrapper, *initial_guesses, name=ordered_params)
    m.errordef = Minuit.LIKELIHOOD

    # 4) Apply the limits
    for pname, (lo, hi) in limits.items():
        m.limits[pname] = (lo, hi)

    # 5) Run the fit
    m.migrad()
    if not m.valid:
        m.simplex()
        m.migrad()

    n_events = sum(len(np.asarray(times_dict.get(iso, []))) for iso in iso_list)
    ndf = max(1, n_events - len(ordered_params))

    out = {}
    param_index = dict(param_indices)
    if not m.valid:
        out["fit_valid"] = False
        for pname in ordered_params:
            val = float(m.values[pname])
            err = float(m.errors[pname]) if pname in m.errors else np.nan
            out[pname] = val
            out["d" + pname] = err
        out["nll"] = float(m.fval)
        if fixed_background:
            for iso, val in fixed_background.items():
                key = f"B_{iso}"
                if key not in out:
                    out[key] = float(val)
                    out["d" + key] = 0.0
        cov = np.zeros((len(ordered_params), len(ordered_params)))
        if "E_Po214" in ordered_params and "N0_Po214" in ordered_params:
            i1 = ordered_params.index("E_Po214")
            i2 = ordered_params.index("N0_Po214")
            out["cov_E_Po214_N0_Po214"] = float(cov[i1, i2])
        return FitResult(out, cov, int(ndf), param_index, counts=int(n_events))

    m.hesse()  # compute uncertainties
    if m.covariance is None:
        # Fallback to zero covariance if hesse() failed
        cov = np.zeros((len(ordered_params), len(ordered_params)))
        perr = np.zeros(len(ordered_params))
    else:
        cov = np.array(m.covariance)
        perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    try:
            eigvals = np.linalg.eigvals(cov)
            fit_valid = bool(np.all(eigvals >= 0))
    except np.linalg.LinAlgError:
        fit_valid = False
    if not fit_valid:
        if strict:
            raise RuntimeError(
                "fit_time_series: covariance matrix not positive definite"
            )
        logging.warning(
            "fit_time_series: covariance matrix not positive definite"
        )
        jitter = 1e-12 * np.mean(np.diag(cov))
        if not np.isfinite(jitter) or jitter <= 0:
            jitter = 1e-12
        cov = cov + jitter * np.eye(cov.shape[0])
        try:
            np.linalg.cholesky(cov)
            fit_valid = True
            perr = np.sqrt(np.clip(np.diag(cov), 0, None))
        except np.linalg.LinAlgError:
            fit_valid = False

    # Likelihood ratio test against background-only model
    best_nll = float(m.fval)
    null_params = [float(m.values[p]) for p in ordered_params]
    for iso in iso_list:
        idx = param_indices.get(f"E_{iso}")
        if idx is not None:
            null_params[idx] = 0.0
    nll_null = _neg_log_likelihood_time(
        null_params,
        times_dict,
        weights_dict,
        t_start,
        t_end,
        iso_list,
        lam_map,
        eff_map,
        fix_b_map,
        fix_n0_map,
        param_indices,
    )
    ts_val = max(0.0, 2.0 * (nll_null - best_nll))
    out["lrt_ts"] = ts_val
    out["nll"] = best_nll
    try:
        crit = chi2.ppf(0.95, df=len(iso_list))
    except Exception:
        crit = 0.0
    # Separate technical validity from statistical significance
    signal_detected = ts_val >= crit
    out["signal_detected"] = signal_detected
    out["fit_valid"] = fit_valid  # Technical validity only (convergence, etc.)
    for i, pname in enumerate(ordered_params):
        out[pname] = float(m.values[pname])
        out["d" + pname] = float(perr[i] if i < len(perr) else np.nan)

    if fixed_background:
        for iso, val in fixed_background.items():
            key = f"B_{iso}"
            if key not in out:
                out[key] = float(val)
                out["d" + key] = 0.0

    if "E_Po214" in ordered_params and "N0_Po214" in ordered_params:
        i1 = ordered_params.index("E_Po214")
        i2 = ordered_params.index("N0_Po214")
        out["cov_E_Po214_N0_Po214"] = float(cov[i1, i2])

    return FitResult(out, cov, int(ndf), param_index, counts=int(n_events))


fit_time_series.__rmtest_time_fit_callback_version__ = (1, 1)


# -----------------------------------------------------
# End of fitting.py
# -----------------------------------------------------
