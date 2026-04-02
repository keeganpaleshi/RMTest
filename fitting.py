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
            floor = _EMG_TAU_FLOOR_DEFAULT if min_tau is None else min_tau
            return val if val >= floor else floor
        _EMG_FLOOR = _EMG_TAU_FLOOR_DEFAULT
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


# Default fitting constants (can be overridden via config)
_TAU_BOUND_EXPANSION_DEFAULT = 10.0
_BACKGROUND_NORM_POINTS_DEFAULT = 512
_EMG_TAU_FLOOR_DEFAULT = 5.0e-4

# Module-level values (can be updated from config)
_TAU_BOUND_EXPANSION = _TAU_BOUND_EXPANSION_DEFAULT


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
    was_scalar = np.isscalar(y)
    y = np.asarray(y, dtype=float)
    out = np.empty_like(y)
    mask = y > 0
    out[mask] = log_expm1_stable(y[mask])
    out[~mask] = -20.0
    return float(out) if was_scalar else out


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
    Emin: float, Emax: float, Eref: float | None = None, n_norm: int | None = None
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
        Defaults to ``_BACKGROUND_NORM_POINTS_DEFAULT`` (512).
    """
    if n_norm is None:
        n_norm = _BACKGROUND_NORM_POINTS_DEFAULT

    if Eref is None:
        Eref = 0.5 * (Emin + Emax)
    grid = np.linspace(Emin, Emax, int(n_norm))

    def shape(E, beta0, beta1):
        exp_grid = _safe_exp(beta0 + beta1 * (grid - Eref))
        # Use np.trapezoid (NumPy 2.0+) or fall back to np.trapz
        trapz_func = getattr(np, 'trapezoid', None) or np.trapz
        Z = trapz_func(exp_grid, grid)
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
    minos_errors: dict[str, tuple[float, float]] | None = None
    """Per-parameter MINOS asymmetric errors ``{name: (lower, upper)}``."""
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
    pre_binned_hist=None,
    pre_dnl_meta=None,
    skip_minos=False,
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
    widths_base = widths.copy()  # Preserve original widths for DNL iteration
    if not unbinned:
        if pre_binned_hist is not None:
            hist = np.asarray(pre_binned_hist, dtype=float)
            if hist.size != centers.size:
                raise ValueError(
                    f"pre_binned_hist length ({hist.size}) does not match "
                    f"bin_edges ({centers.size} bins)"
                )
        else:
            hist, _ = np.histogram(e, bins=edges)

    E_lo = float(edges[0])
    E_hi = float(edges[-1])

    # Configure background handling
    background_model = flags.get("background_model")
    priors = dict(priors)
    if background_model in ("loglin_unit", "sigmoid_unit", "exp_unit", "double_logit_unit", "none"):
        if "S_bkg" not in priors:
            signal_guess = 0.0
            # Dynamically discover isotopes from priors (any key matching mu_*)
            _discovered_isos = sorted(
                k[3:] for k in priors if k.startswith("mu_") and f"S_{k[3:]}" in priors
            )
            for iso_name in _discovered_isos:
                prior_val = priors.get(f"S_{iso_name}")
                if isinstance(prior_val, (tuple, list)) and len(prior_val) > 0:
                    try:
                        signal_guess += max(float(prior_val[0]), 0.0)
                    except (TypeError, ValueError):
                        continue

            remaining = float(n_events) - signal_guess
            if not np.isfinite(remaining) or remaining <= 0.0:
                remaining = max(0.1 * float(n_events), 1.0)

            mu = max(float(remaining), 1.0)
            sig = max(np.sqrt(mu), 0.5 * float(n_events), 1.0)
            priors["S_bkg"] = (mu, sig)
        if str(background_model).lower() != "none":
            required = {"b0", "b1"}
            missing = required - priors.keys()
            if missing:
                got = sorted(priors.keys())
                raise ValueError(
                    "background_model=loglin_unit requires params {S_bkg, b0, b1}; got: "
                    f"{got}"
                )

    if "S_bkg" in priors or background_model in ("loglin_unit", "sigmoid_unit", "exp_unit", "double_logit_unit", "none"):
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
    # Dynamically discover isotopes from priors: any key matching mu_* with S_*
    iso_list = sorted(
        k[3:] for k in priors if k.startswith("mu_") and f"S_{k[3:]}" in priors
    )
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
            key = f"sigma_{iso}"
            peak_sigma_prior = overrides.get("peak_sigma_prior")
            if isinstance(peak_sigma_prior, (list, tuple)) and len(peak_sigma_prior) == 2:
                priors[key] = (float(peak_sigma_prior[0]), float(peak_sigma_prior[1]))
            elif "peak_sigma" in overrides:
                sigma_mean = float(overrides["peak_sigma"])
                existing = priors.get(key)
                if existing is None:
                    sigma_prior = max(abs(sigma_mean) * 0.1, 1.0e-3)
                else:
                    sigma_prior = float(existing[1])
                priors[key] = (sigma_mean, sigma_prior)

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
    # Determine which peaks include a low-energy shelf component
    use_shelf = {}
    if cfg is not None:
        shelf_cfg = cfg.get("spectral_fit", {}).get("use_shelf", {})
        if isinstance(shelf_cfg, bool):
            use_shelf = {iso: shelf_cfg for iso in iso_list}
        elif isinstance(shelf_cfg, dict):
            use_shelf = {iso: bool(shelf_cfg.get(iso, False)) for iso in iso_list}
    # Also enable shelf if f_shelf priors are present
    for iso in iso_list:
        if f"f_shelf_{iso}" in priors:
            use_shelf[iso] = True

    # Determine which peaks include a broad "halo" component (double-peak)
    use_halo = {}
    if cfg is not None:
        halo_cfg = cfg.get("spectral_fit", {}).get("use_halo", {})
        if isinstance(halo_cfg, bool):
            use_halo = {iso: halo_cfg for iso in iso_list}
        elif isinstance(halo_cfg, dict):
            use_halo = {iso: bool(halo_cfg.get(iso, False)) for iso in iso_list}
    # Also enable halo if f_halo priors are present
    for iso in iso_list:
        if f"f_halo_{iso}" in priors:
            use_halo[iso] = True

    # --- Shared shape parameters ---
    # When shared_shape_params.<type> is True, a single shared parameter
    # replaces per-isotope entries for all participating isotopes.
    # The broadcast from shared → per-isotope happens in _physical_params.
    shared_participants: dict[str, list[str]] = {}
    # --- Tau linear energy model ---
    # When tau_energy_model: "linear", tau for each alpha peak is computed as:
    #   tau(E) = tau_0 + tau_slope * (E - E_ref)
    # This replaces both per-isotope tau and shared tau for standard alpha peaks.
    # Extra peaks (e.g. Unknown1) keep their own independent tau.
    _tau_linear_model = False
    _tau_linear_E_ref = 7.0
    _tau_linear_participants: list[str] = []
    if cfg is not None:
        _shared_cfg = cfg.get("spectral_fit", {}).get("shared_shape_params", {})
        sp_cfg = cfg.get("spectral_fit", {})
        # Extra peaks (e.g. Unknown1) must NOT participate in shared shape params —
        # they have different physics from the standard alpha peaks.
        _extra_peak_names = set(
            sp_cfg.get("extra_peaks", {}).keys() if isinstance(sp_cfg.get("extra_peaks"), dict) else []
        )
        # Check for tau linear energy model
        _tau_em = sp_cfg.get("tau_energy_model", "").lower()
        if _tau_em == "linear":
            _tau_linear_model = True
            _tau_linear_E_ref = float(sp_cfg.get("tau_energy_model_E_ref", 7.0))
            _tau_linear_participants = [iso for iso in iso_list if use_emg.get(iso, False) and iso not in _extra_peak_names]
        _shared_mappings = {
            "tau": lambda iso: use_emg.get(iso, False),
            "f_shelf": lambda iso: use_shelf.get(iso, False),
            "sigma_shelf": lambda iso: use_shelf.get(iso, False),
            "f_halo": lambda iso: use_halo.get(iso, False),
            "sigma_halo": lambda iso: use_halo.get(iso, False),
            "tau_halo": lambda iso: use_halo.get(iso, False),
        }
        # Passive recipients: receive shared values but don't influence them.
        # Useful for weak peaks (Po216) or extra peaks (Unknown1) that should
        # inherit detector-response params without distorting the shared fit.
        _shared_passive_isos: set[str] = set(
            sp_cfg.get("shared_shape_passive", [])
        )
        for ptype, eligibility_fn in _shared_mappings.items():
            # Skip tau sharing if using linear energy model instead
            if ptype == "tau" and _tau_linear_model:
                continue
            if _shared_cfg.get(ptype, False):
                participants = [
                    iso for iso in iso_list
                    if eligibility_fn(iso)
                    and iso not in _extra_peak_names
                    and iso not in _shared_passive_isos
                ]
                if participants:
                    shared_participants[ptype] = participants
                    # Load shared prior into priors dict
                    prior_key = f"{ptype}_shared_prior"
                    if prior_key in sp_cfg:
                        val = sp_cfg[prior_key]
                        if isinstance(val, (list, tuple)) and len(val) == 2:
                            priors[f"{ptype}_shared"] = (float(val[0]), float(val[1]))

    def _uses_shared(ptype: str, iso: str) -> bool:
        """True if iso uses the shared value for ptype (participant OR passive)."""
        if ptype in shared_participants and iso in shared_participants[ptype]:
            return True
        # Passive isotopes also use shared value (if that ptype is shared at all)
        if iso in _shared_passive_isos and ptype in shared_participants:
            return True
        return False

    for iso in iso_list:
        param_index[f"mu_{iso}"] = len(param_order)
        param_order.append(f"mu_{iso}")
        sigma_key = f"sigma_{iso}"
        if sigma_key in priors:
            param_index[sigma_key] = len(param_order)
            param_order.append(sigma_key)
        param_index[f"S_{iso}"] = len(param_order)
        param_order.append(f"S_{iso}")
        # tau: skip per-isotope if shared OR if using linear energy model
        _iso_in_tau_linear = _tau_linear_model and iso in _tau_linear_participants
        if use_emg[iso] and not _iso_in_tau_linear and not _uses_shared("tau", iso):
            param_index[f"tau_{iso}"] = len(param_order)
            param_order.append(f"tau_{iso}")
        # Low-energy shelf fraction
        if use_shelf.get(iso, False):
            # f_shelf: skip per-isotope if shared or passive
            if not _uses_shared("f_shelf", iso):
                shelf_key = f"f_shelf_{iso}"
                param_index[shelf_key] = len(param_order)
                param_order.append(shelf_key)
            # sigma_shelf: skip per-isotope if shared or passive
            shelf_sigma_key = f"sigma_shelf_{iso}"
            if shelf_sigma_key in priors and not _uses_shared("sigma_shelf", iso):
                param_index[shelf_sigma_key] = len(param_order)
                param_order.append(shelf_sigma_key)
        # Halo (broad) component fraction and width
        if use_halo.get(iso, False):
            # f_halo: skip per-isotope if shared or passive
            if not _uses_shared("f_halo", iso):
                halo_key = f"f_halo_{iso}"
                param_index[halo_key] = len(param_order)
                param_order.append(halo_key)
            # sigma_halo: skip per-isotope if shared or passive
            halo_sigma_key = f"sigma_halo_{iso}"
            if halo_sigma_key in priors and not _uses_shared("sigma_halo", iso):
                param_index[halo_sigma_key] = len(param_order)
                param_order.append(halo_sigma_key)
            # tau_halo: skip per-isotope if shared or passive
            halo_tau_key = f"tau_halo_{iso}"
            if halo_tau_key in priors and not _uses_shared("tau_halo", iso):
                param_index[halo_tau_key] = len(param_order)
                param_order.append(halo_tau_key)
        # Beta coincidence: high-energy tail from alpha+beta pile-up
        # Enabled per-isotope when f_beta_{iso}_prior is in spectral_fit config
        _fb_prior_key = f"f_beta_{iso}_prior"
        if sp_cfg.get(_fb_prior_key) is not None:
            _fb_prior = sp_cfg[_fb_prior_key]
            priors[f"f_beta_{iso}"] = (float(_fb_prior[0]), float(_fb_prior[1]))
            param_index[f"f_beta_{iso}"] = len(param_order)
            param_order.append(f"f_beta_{iso}")
            _lb_prior_key = f"lambda_beta_{iso}_prior"
            _lb_prior = sp_cfg.get(_lb_prior_key, [0.5, 0.5])
            priors[f"lambda_beta_{iso}"] = (float(_lb_prior[0]), float(_lb_prior[1]))
            param_index[f"lambda_beta_{iso}"] = len(param_order)
            param_order.append(f"lambda_beta_{iso}")
    # Add shared shape parameters (one entry per shared type)
    for shared_name in shared_participants:
        key = f"{shared_name}_shared"
        if key not in param_index:
            param_index[key] = len(param_order)
            param_order.append(key)
    # Tau linear energy model parameters: tau_0 and tau_slope
    if _tau_linear_model and _tau_linear_participants:
        sp_cfg_t = cfg.get("spectral_fit", {}) if cfg else {}
        _tau0_prior = sp_cfg_t.get("tau_0_prior", [0.12, 0.08])
        priors["tau_0"] = (float(_tau0_prior[0]), float(_tau0_prior[1]))
        param_index["tau_0"] = len(param_order)
        param_order.append("tau_0")
        _tslope_prior = sp_cfg_t.get("tau_slope_prior", [0.0, 0.05])
        priors["tau_slope"] = (float(_tslope_prior[0]), float(_tslope_prior[1]))
        param_index["tau_slope"] = len(param_order)
        param_order.append("tau_slope")
    # Right-side broadening: sigma_right = sigma * (1 + sigma_asym).
    # Modes:
    #   1. use_sigma_asym: true  → single shared sigma_asym (constant across peaks)
    #   2. asym_energy_model: linear/quadratic/cubic
    #      asym(E) = asym_0 + asym_1*(E-Eref) [+ asym_2*(E-Eref)^2 [+ asym_3*(E-Eref)^3]]
    #      Broadcasts per-isotope sigma_asym_{iso} values in _physical_params.
    _use_sigma_asym = False
    _asym_poly_model = False
    _asym_poly_order = 0  # 1=linear, 2=quadratic, 3=cubic
    _asym_poly_E_ref = 7.0
    _asym_poly_participants = []
    if cfg is not None:
        _use_sigma_asym = bool(cfg.get("spectral_fit", {}).get("use_sigma_asym", False))
        _asym_em = sp_cfg.get("asym_energy_model", "").lower()
        if _asym_em in ("linear", "quadratic", "cubic"):
            _asym_poly_model = True
            _asym_poly_order = {"linear": 1, "quadratic": 2, "cubic": 3}[_asym_em]
            _use_sigma_asym = False  # polynomial model supersedes single-param
            _asym_poly_E_ref = float(sp_cfg.get("asym_energy_model_E_ref", 7.0))
            # Isotopes with free (independent) sigma_asym — excluded from polynomial
            _asym_free_isos = set(sp_cfg.get("asym_free_isotopes", []))
            _asym_poly_participants = [
                iso for iso in iso_list
                if iso not in _extra_peak_names and iso not in _asym_free_isos
            ]
    if _asym_poly_model:
        # asym_0 is always present (constant term)
        _a0_prior = sp_cfg.get("asym_0_prior", [0.2, 0.15])
        priors["asym_0"] = (float(_a0_prior[0]), float(_a0_prior[1]))
        param_index["asym_0"] = len(param_order)
        param_order.append("asym_0")
        # Higher-order coefficients: asym_1, asym_2, asym_3
        for _ord in range(1, _asym_poly_order + 1):
            _key = f"asym_{_ord}"
            _default = [0.0, 0.05]
            _prior = sp_cfg.get(f"{_key}_prior", _default)
            priors[_key] = (float(_prior[0]), float(_prior[1]))
            param_index[_key] = len(param_order)
            param_order.append(_key)
        # Register free sigma_asym for isotopes excluded from polynomial
        for _free_iso in sorted(_asym_free_isos):
            if _free_iso in iso_list:
                _sa_key = f"sigma_asym_{_free_iso}"
                _sa_default = [0.5, 0.3]  # broad default prior
                _sa_prior = sp_cfg.get(f"sigma_asym_{_free_iso}_prior", _sa_default)
                priors[_sa_key] = (float(_sa_prior[0]), float(_sa_prior[1]))
                param_index[_sa_key] = len(param_order)
                param_order.append(_sa_key)
    # Free sigma_right: independent right-side width (not tied to sigma * (1+asym))
    _sigma_right_free_isos: set[str] = set()
    if cfg is not None:
        _sigma_right_free_isos = set(
            sp_cfg.get("sigma_right_free_isotopes", [])
        )
        for _sr_iso in sorted(_sigma_right_free_isos):
            if _sr_iso in iso_list:
                _sr_key = f"sigma_right_{_sr_iso}"
                _sr_default = [0.20, 0.10]  # ~2x typical sigma
                _sr_prior = sp_cfg.get(f"sigma_right_{_sr_iso}_prior", _sr_default)
                priors[_sr_key] = (float(_sr_prior[0]), float(_sr_prior[1]))
                param_index[_sr_key] = len(param_order)
                param_order.append(_sr_key)
    elif _use_sigma_asym:
        sp_cfg_sa = cfg.get("spectral_fit", {}) if cfg else {}
        _sa_prior = sp_cfg_sa.get("sigma_asym_prior", [0.1, 0.1])
        priors["sigma_asym"] = (float(_sa_prior[0]), float(_sa_prior[1]))
        param_index["sigma_asym"] = len(param_order)
        param_order.append("sigma_asym")
    # Additive right-side exponential tail (shared, on top of sigma_asym)
    _use_tail_right = False
    if cfg is not None:
        _use_tail_right = bool(cfg.get("spectral_fit", {}).get("use_tail_right", False))
    if _use_tail_right:
        sp_cfg_tr = cfg.get("spectral_fit", {}) if cfg else {}
        _ftr_prior = sp_cfg_tr.get("f_tail_right_prior", [0.05, 0.05])
        priors["f_tail_right"] = (float(_ftr_prior[0]), float(_ftr_prior[1]))
        param_index["f_tail_right"] = len(param_order)
        param_order.append("f_tail_right")
        _ttr_prior = sp_cfg_tr.get("tau_tail_right_prior", [0.15, 0.10])
        priors["tau_tail_right"] = (float(_ttr_prior[0]), float(_ttr_prior[1]))
        param_index["tau_tail_right"] = len(param_order)
        param_order.append("tau_tail_right")
    # For "none" background model, skip b0/b1/b2/b3 entirely —
    # all spectral structure comes from peaks + halos + shelves.
    if str(background_model).lower() != "none":
        param_index["b0"] = len(param_order)
        param_order.append("b0")
        param_index["b1"] = len(param_order)
        param_order.append("b1")
        # Log-quadratic background term (b2)
        if "b2" in priors:
            param_index["b2"] = len(param_order)
            param_order.append("b2")
        # Log-cubic background term (b3)
        if "b3" in priors:
            param_index["b3"] = len(param_order)
            param_order.append("b3")
    if background_model in ("loglin_unit", "sigmoid_unit", "exp_unit", "double_logit_unit", "none") or "S_bkg" in priors:
        param_index["S_bkg"] = len(param_order)
        param_order.append("S_bkg")

    # ADC edge components: sigmoid terms at low/high energy boundaries
    _adc_edge = False
    if cfg is not None:
        _adc_edge = bool(
            cfg.get("spectral_fit", {}).get("adc_edge_components", False)
        )
    if _adc_edge:
        _adc_lo_prior = sp_cfg.get("S_adc_lo_prior", [50.0, 100.0])
        priors["S_adc_lo"] = (float(_adc_lo_prior[0]), float(_adc_lo_prior[1]))
        _w_lo_prior = sp_cfg.get("w_adc_lo_prior", [0.3, 0.3])
        priors["w_adc_lo"] = (float(_w_lo_prior[0]), float(_w_lo_prior[1]))

        _adc_hi_prior = sp_cfg.get("S_adc_hi_prior", [50.0, 100.0])
        priors["S_adc_hi"] = (float(_adc_hi_prior[0]), float(_adc_hi_prior[1]))
        _w_hi_prior = sp_cfg.get("w_adc_hi_prior", [0.3, 0.3])
        priors["w_adc_hi"] = (float(_w_hi_prior[0]), float(_w_hi_prior[1]))

    # ADC edge component parameters (S and w for each edge — no mu, anchored at boundaries)
    if _adc_edge:
        for _adc_key in ("S_adc_lo", "w_adc_lo",
                         "S_adc_hi", "w_adc_hi"):
            param_index[_adc_key] = len(param_order)
            param_order.append(_adc_key)

    p0 = []
    bounds_lo, bounds_hi = [], []
    eps = 1e-12
    sigma0_mean = sigma0_val
    for name in param_order:
        mean, sig = p(name, 1.0)
        if name.startswith("S_"):
            mean = float(_softplus_inv(mean))
        # Enforce a strictly positive initial tau to avoid singular EMG tails
        if name.startswith("tau_") and name not in ("tau_0", "tau_slope"):
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
            if name.startswith("tau_") and name not in ("tau_0", "tau_slope"):
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
        if name.startswith("tau_") and name not in ("tau_0", "tau_slope"):
            lo = max(lo, _EMG_FLOOR)
            if max_tau_ratio is not None:
                hi = min(hi, max_tau_ratio * sigma0_mean)
        if name in ("sigma0", "F") or name.startswith("sigma_"):
            lo = max(lo, 0.0)
        if name.startswith("f_shelf_"):
            lo = max(lo, 0.0)
            _f_shelf_cap = float(flags.get("max_f_shelf", 0.30))
            hi = min(hi, _f_shelf_cap)  # shelf fraction cap (configurable)
        if name.startswith("f_halo_"):
            lo = max(lo, 0.0)
            _f_halo_cap = float(flags.get("max_f_halo", 0.40))
            hi = min(hi, _f_halo_cap)  # halo fraction cap (configurable)
        if name.startswith("f_beta_"):
            lo = max(lo, 0.0)
            hi = min(hi, 0.30)   # beta coincidence fraction cap
        if name.startswith("lambda_beta_"):
            lo = max(lo, 0.05)   # minimum 50 keV decay length
            hi = min(hi, 5.0)    # maximum 5 MeV decay length
        # Tau linear model: tau_0 must be positive; tau_slope is free
        if name == "tau_0":
            lo = max(lo, _EMG_FLOOR)
            if max_tau_ratio is not None:
                hi = min(hi, max_tau_ratio * sigma0_mean)
        if name == "tau_slope":
            pass  # allow negative slope (tau decreasing with energy)
        # sigma_asym: fractional right-side broadening, must be >= 0
        if name == "sigma_asym" or name.startswith("sigma_asym_"):
            lo = max(lo, 0.0)
            hi = min(hi, 2.0)  # at most 3× wider on right (1+2.0)
        # sigma_right: independent right-side width in MeV, must be > 0
        if name.startswith("sigma_right_"):
            lo = max(lo, 0.01)  # minimum 10 keV
            hi = min(hi, 1.0)   # maximum 1 MeV
        # Asym polynomial energy model: asym_0 >= 0, higher-order terms free
        if name == "asym_0":
            lo = max(lo, 0.0)
            hi = min(hi, 2.0)  # same cap as sigma_asym
        if name in ("asym_1", "asym_2", "asym_3"):
            pass  # free — allow any sign for polynomial coefficients
        # Right-tail fraction and decay constant
        if name == "f_tail_right":
            lo = max(lo, 0.0)
            hi = min(hi, 0.20)  # at most 20% — tail should be small
        if name == "tau_tail_right":
            # Floor at 0.02 MeV — must be a genuine tail, not a near-Gaussian
            lo = max(lo, 0.02)
            # Configurable ceiling to keep tail short-range
            _tau_tr_max = float(flags.get("tau_tail_right_max", 1.0))
            hi = min(hi, _tau_tr_max)
        # sigma_shelf must be physically meaningful (configurable floor, default 0.03 MeV)
        if name.startswith("sigma_shelf_"):
            _sigma_shelf_min = float(flags.get("sigma_shelf_min", 0.03))
            lo = max(lo, _sigma_shelf_min)
        # tau_halo: cap at N× core tau to prevent unrealistically long
        # right-side EMG tails that bleed into neighbouring peaks.
        # (brentq fallback in _right_emg_mode_offset handles edge cases.)
        if name.startswith("tau_halo_"):
            iso_name = name.replace("tau_halo_", "")
            tau_core_key = f"tau_{iso_name}"
            if tau_core_key in priors:
                core_tau_mean = float(priors[tau_core_key][0])
                _tau_halo_max_mult = float(flags.get("tau_halo_max_mult", 5.0))
                hi = min(hi, core_tau_mean * _tau_halo_max_mult)
        # sigma_halo: enforce a reasonable range relative to core sigma.
        # Min 1.5× core sigma (must be wider than core).  Max 8× allows
        # broad halos for dead-layer scattering that can extend 1-2 MeV.
        if name.startswith("sigma_halo_"):
            iso_name = name.replace("sigma_halo_", "")
            sigma_core_key = f"sigma_{iso_name}"
            if sigma_core_key in priors:
                core_sigma_mean = float(priors[sigma_core_key][0])
            else:
                core_sigma_mean = sigma0_mean
            _sigma_halo_min_mult = float(flags.get("sigma_halo_min_mult", 1.5))
            _sigma_halo_max_mult = float(flags.get("sigma_halo_max_mult", 8.0))
            lo = max(lo, core_sigma_mean * _sigma_halo_min_mult)
            hi = min(hi, core_sigma_mean * _sigma_halo_max_mult)
        # --- Shared shape parameter bounds ---
        if name == "f_shelf_shared":
            lo = max(lo, 0.0)
            hi = min(hi, float(flags.get("max_f_shelf", 0.30)))
        if name == "f_halo_shared":
            lo = max(lo, 0.0)
            hi = min(hi, float(flags.get("max_f_halo", 0.40)))
        if name == "sigma_shelf_shared":
            lo = max(lo, 0.03)
        if name == "tau_halo_shared":
            # Cap at 5× the shared tau prior mean (or average core tau)
            if "tau_shared" in priors:
                core_tau_mean = float(priors["tau_shared"][0])
            elif "tau_shared_prior" in priors:
                core_tau_mean = float(priors["tau_shared_prior"][0])
            else:
                # Average of per-isotope tau priors
                _tau_vals = [float(priors[f"tau_{iso}"][0]) for iso in iso_list if f"tau_{iso}" in priors]
                core_tau_mean = np.mean(_tau_vals) if _tau_vals else 0.08
            hi = min(hi, core_tau_mean * 5.0)
        if name == "sigma_halo_shared":
            # Use average core sigma across isotopes
            _sigma_vals = [float(priors.get(f"sigma_{iso}", (sigma0_mean,))[0]) for iso in iso_list]
            avg_core_sigma = np.mean(_sigma_vals) if _sigma_vals else sigma0_mean
            lo = max(lo, avg_core_sigma * float(flags.get("sigma_halo_min_mult", 1.5)))
            hi = min(hi, avg_core_sigma * float(flags.get("sigma_halo_max_mult", 8.0)))
        if name == "tau_shared":
            lo = max(lo, _EMG_FLOOR)
        # ADC edge component bounds
        if name.startswith("S_adc_"):
            lo = max(lo, 0.0)  # non-negative counts
        if name.startswith("w_adc_"):
            lo = max(lo, 0.05)   # minimum width 50 keV
            hi = min(hi, 3.0)    # maximum width 3 MeV
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

    # area_keys contains only peak integrals, not background
    # (background integral is recovered by subtraction in extended likelihood)
    area_keys = [f"S_{iso}" for iso in iso_list]

    domain = (E_lo, E_hi)
    use_emg_map = {iso: bool(use_emg.get(iso, False)) for iso in iso_list}

    # Extract clip_floor from config with safe default
    clip_floor = 1e-300
    if cfg is not None:
        clip_floor = float(cfg.get("spectral_fit", {}).get("clip_floor", 1e-300))

    loglin_n_norm = None
    _bkg_range = None
    if cfg is not None:
        loglin_n_norm = cfg.get("spectral_fit", {}).get("loglin_n_norm")
        _br = cfg.get("spectral_fit", {}).get("bkg_energy_range")
        if _br is not None and len(_br) == 2:
            _bkg_range = (float(_br[0]), float(_br[1]))

    use_shelf_map = {iso: bool(use_shelf.get(iso, False)) for iso in iso_list}
    use_halo_map = {iso: bool(use_halo.get(iso, False)) for iso in iso_list}

    # shelf_range: how far (MeV) the shelf extends below the peak (Gaussian taper)
    # Can be a scalar (applied to all isotopes) or a dict with per-isotope values
    _shelf_range = None
    if cfg is not None:
        _sr = cfg.get("spectral_fit", {}).get("shelf_range")
        if _sr is not None:
            if isinstance(_sr, dict):
                # Per-isotope shelf range: {"Po210": 1.5, "Po218": 0.4, "default": 1.0}
                _shelf_range = {k: float(v) for k, v in _sr.items()}
            else:
                _shelf_range = float(_sr)

    _shelf_cutoff_delta = None
    if cfg is not None:
        _scd = cfg.get("spectral_fit", {}).get("shelf_cutoff_delta")
        if _scd is not None:
            _shelf_cutoff_delta = float(_scd)

    spectral_intensity = build_spectral_intensity(
        iso_list,
        use_emg_map,
        domain,
        clip_floor=clip_floor,
        background_model=background_model,
        loglin_n_norm=loglin_n_norm,
        use_shelf=use_shelf_map,
        use_halo=use_halo_map,
        shelf_range=_shelf_range,
        shelf_cutoff_delta=_shelf_cutoff_delta,
        adc_edge_components=_adc_edge,
        bkg_range=_bkg_range,
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
        # For "none" background model, b0/b1 are not in the parameter space
        if "b0" in raw_map:
            params_dict["b0"] = float(raw_map["b0"])
        if "b1" in raw_map:
            params_dict["b1"] = float(raw_map["b1"])
        if "b2" in raw_map:
            params_dict["b2"] = float(raw_map["b2"])
        if "b3" in raw_map:
            params_dict["b3"] = float(raw_map["b3"])
        if "F" in raw_map or not fix_F:
            params_dict["F"] = float(raw_map.get("F", F_val))
        for iso in iso_list:
            params_dict[f"mu_{iso}"] = float(raw_map[f"mu_{iso}"])
            sigma_key = f"sigma_{iso}"
            if sigma_key in raw_map:
                params_dict[sigma_key] = float(raw_map[sigma_key])
            S_val = float(_softplus(raw_map[f"S_{iso}"]))
            params_dict[f"S_{iso}"] = S_val
            params_dict[f"N_{iso}"] = S_val
            tau_key = f"tau_{iso}"
            if use_emg_map.get(iso, False) and tau_key in raw_map:
                params_dict[tau_key] = float(raw_map[tau_key])
            shelf_key = f"f_shelf_{iso}"
            if use_shelf_map.get(iso, False) and shelf_key in raw_map:
                params_dict[shelf_key] = float(raw_map[shelf_key])
            shelf_sigma_key = f"sigma_shelf_{iso}"
            if shelf_sigma_key in raw_map:
                params_dict[shelf_sigma_key] = float(raw_map[shelf_sigma_key])
            halo_key = f"f_halo_{iso}"
            if use_halo_map.get(iso, False) and halo_key in raw_map:
                params_dict[halo_key] = float(raw_map[halo_key])
            halo_sigma_key = f"sigma_halo_{iso}"
            if halo_sigma_key in raw_map:
                params_dict[halo_sigma_key] = float(raw_map[halo_sigma_key])
            halo_tau_key = f"tau_halo_{iso}"
            if halo_tau_key in raw_map:
                params_dict[halo_tau_key] = float(raw_map[halo_tau_key])
            # Beta coincidence parameters
            for _bk in (f"f_beta_{iso}", f"lambda_beta_{iso}"):
                if _bk in raw_map:
                    params_dict[_bk] = float(raw_map[_bk])
        # Broadcast shared shape parameters → per-isotope keys
        for shared_name, participants in shared_participants.items():
            shared_key = f"{shared_name}_shared"
            if shared_key in raw_map:
                _shared_val = float(raw_map[shared_key])
                for iso in participants:
                    per_iso_key = f"{shared_name}_{iso}"
                    if per_iso_key not in params_dict:
                        params_dict[per_iso_key] = _shared_val
                # Also broadcast to passive recipients
                for iso in _shared_passive_isos:
                    per_iso_key = f"{shared_name}_{iso}"
                    if per_iso_key not in params_dict:
                        params_dict[per_iso_key] = _shared_val
        # Tau linear energy model: tau(E) = tau_0 + tau_slope * (E - E_ref)
        if _tau_linear_model and "tau_0" in raw_map and "tau_slope" in raw_map:
            _t0 = float(raw_map["tau_0"])
            _ts = float(raw_map["tau_slope"])
            params_dict["tau_0"] = _t0
            params_dict["tau_slope"] = _ts
            for iso in _tau_linear_participants:
                mu_iso = float(raw_map.get(f"mu_{iso}", 0.0))
                tau_val = _t0 + _ts * (mu_iso - _tau_linear_E_ref)
                # Enforce EMG floor
                tau_val = max(tau_val, _EMG_FLOOR)
                params_dict[f"tau_{iso}"] = tau_val
        # Asym polynomial energy model: asym(E) = sum_k asym_k * (E - E_ref)^k
        # Broadcasts per-isotope sigma_asym_{iso} values.
        if _asym_poly_model and "asym_0" in raw_map:
            _coeffs = [float(raw_map["asym_0"])]
            params_dict["asym_0"] = _coeffs[0]
            for _ord in range(1, _asym_poly_order + 1):
                _key = f"asym_{_ord}"
                _c = float(raw_map.get(_key, 0.0))
                _coeffs.append(_c)
                params_dict[_key] = _c
            for iso in _asym_poly_participants:
                mu_iso = float(raw_map.get(f"mu_{iso}", 0.0))
                _dE = mu_iso - _asym_poly_E_ref
                asym_val = sum(c * _dE**k for k, c in enumerate(_coeffs))
                asym_val = max(asym_val, 0.0)  # can't have negative asymmetry
                params_dict[f"sigma_asym_{iso}"] = asym_val
            # Free isotopes: pass through their independent sigma_asym directly
            for _free_iso in (_asym_free_isos if _asym_poly_model else ()):
                _sa_key = f"sigma_asym_{_free_iso}"
                if _sa_key in raw_map:
                    params_dict[_sa_key] = max(float(raw_map[_sa_key]), 0.0)
        # Free sigma_right: pass through directly
        for _sr_iso in _sigma_right_free_isos:
            _sr_key = f"sigma_right_{_sr_iso}"
            if _sr_key in raw_map:
                params_dict[_sr_key] = max(float(raw_map[_sr_key]), 0.0)
        # Shared right-side broadening (single-parameter mode)
        if "sigma_asym" in raw_map:
            params_dict["sigma_asym"] = float(raw_map["sigma_asym"])
        # Additive right-side tail (legacy — kept for backward compat)
        if "f_tail_right" in raw_map:
            params_dict["f_tail_right"] = float(raw_map["f_tail_right"])
        if "tau_tail_right" in raw_map:
            params_dict["tau_tail_right"] = float(raw_map["tau_tail_right"])
        if "S_bkg" in raw_map:
            params_dict["S_bkg"] = float(_softplus(raw_map["S_bkg"]))
        # ADC edge components — pass through (S_ uses softplus)
        for _adc_key in ("S_adc_lo", "w_adc_lo",
                         "S_adc_hi", "w_adc_hi"):
            if _adc_key in raw_map:
                if _adc_key.startswith("S_"):
                    params_dict[_adc_key] = float(_softplus(raw_map[_adc_key]))
                else:
                    params_dict[_adc_key] = float(raw_map[_adc_key])
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

    _dnl_iters = 0   # default: no DNL correction
    _dnl_meta = {}   # populated if DNL correction is applied

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

        # Build Gaussian prior penalty map from flags["penalty_priors"].
        # Each entry is {param_name: [mean, sigma]}.  These add soft
        # constraints to the NLL so the fitter can't drive params to
        # boundary values when the likelihood is insensitive.
        _penalty_map = {}
        if flags:
            _pp_cfg = flags.get("penalty_priors", {})
            if isinstance(_pp_cfg, dict):
                for _pp_name, _pp_val in _pp_cfg.items():
                    if (
                        _pp_name in param_index
                        and isinstance(_pp_val, (list, tuple))
                        and len(_pp_val) == 2
                    ):
                        _pp_mu = float(_pp_val[0])
                        _pp_sig = float(_pp_val[1])
                        if _pp_sig > 0:
                            _penalty_map[param_index[_pp_name]] = (_pp_mu, _pp_sig)
        if _penalty_map:
            logging.info(
                "Gaussian penalty priors on %d parameters: %s",
                len(_penalty_map),
                ", ".join(
                    f"{param_order[i]}=N({mu:.3g},{sig:.3g})"
                    for i, (mu, sig) in sorted(_penalty_map.items())
                ),
            )

        import time as _time_mod
        _fit_timers = {}

        _nll_call_count = [0]

        def _nll(*params):
            _nll_call_count[0] += 1
            model = _model_binned(centers, *params)
            model_safe = np.maximum(model, 1e-300)  # Floor to prevent log(0)
            nll_poisson = float(np.sum(model_safe - hist * np.log(model_safe)))
            # Gaussian prior penalties (0.5 * ((p - mu)/sigma)^2)
            nll_prior = 0.0
            for _idx, (_mu, _sig) in _penalty_map.items():
                nll_prior += 0.5 * ((params[_idx] - _mu) / _sig) ** 2
            return nll_poisson + nll_prior

        _t0_initial = _time_mod.perf_counter()
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
        _fit_timers["initial_migrad_s"] = _time_mod.perf_counter() - _t0_initial
        _fit_timers["initial_migrad_nll_calls"] = _nll_call_count[0]

        # ── Fourier DNL helper ─────────────────────────────────────
        def _fit_fourier_dnl(residuals, bin_indices, adc_bin_width,
                             period_codes, valid_mask):
            """Fit Fourier model to DNL residuals at ADC bit-cycling periods.

            Parameters
            ----------
            residuals : array
                DNL residuals (ratio - 1.0) for each bin.
            bin_indices : array
                Integer bin indices (0-based).
            adc_bin_width : int
                Number of ADC codes per spectral bin.
            period_codes : list[int]
                ADC bit-cycling periods in code units (e.g. [4,8,...,512]).
            valid_mask : array of bool
                Which bins have reliable residual estimates.

            Returns
            -------
            dnl_factors : array
                Reconstructed DNL factors (1.0 + Fourier model) for all bins.
            coeffs : dict
                {period_codes: (a_k, b_k)} Fourier coefficients.
            """
            # Convert code-space periods to bin-space
            periods_bin = [p / adc_bin_width for p in period_codes]
            # Keep only resolvable periods (Nyquist: P_bin >= 2.0)
            resolvable = [(pc, pb) for pc, pb in zip(period_codes, periods_bin)
                          if pb >= 2.0]
            if not resolvable:
                logging.warning("No resolvable Fourier DNL periods for "
                                "adc_bin_width=%d", adc_bin_width)
                return np.ones_like(residuals, dtype=float), {}

            valid_idx = np.flatnonzero(valid_mask)
            if valid_idx.size < 2 * len(resolvable):
                logging.warning("Too few valid bins (%d) for %d Fourier terms",
                                valid_idx.size, len(resolvable))
                return np.ones_like(residuals, dtype=float), {}

            # Build design matrix: cos/sin for each resolvable period
            n_terms = len(resolvable)
            A = np.zeros((valid_idx.size, 2 * n_terms))
            for k, (pc, pb) in enumerate(resolvable):
                phase = 2.0 * np.pi * bin_indices[valid_idx] / pb
                A[:, 2*k] = np.cos(phase)
                A[:, 2*k+1] = np.sin(phase)

            # Least-squares fit
            b = residuals[valid_idx]
            result, _res, _rank, _sv = np.linalg.lstsq(A, b, rcond=None)

            # Reconstruct over all bins
            dnl_model = np.zeros(len(bin_indices), dtype=float)
            coeffs = {}
            for k, (pc, pb) in enumerate(resolvable):
                a_k = float(result[2*k])
                b_k = float(result[2*k+1])
                phase = 2.0 * np.pi * bin_indices / pb
                dnl_model += a_k * np.cos(phase) + b_k * np.sin(phase)
                coeffs[pc] = (a_k, b_k)

            dnl_factors = 1.0 + dnl_model
            return dnl_factors, coeffs

        # ── ADC DNL (differential non-linearity) correction ─────────
        # Estimate per-bin correction factors from the data/model ratio
        # and refit with corrected bin widths.  The DNL is a hardware
        # property of the ADC: some channels have slightly wider or
        # narrower effective voltage ranges, causing systematic count
        # excesses/deficits at the ±10-20 % level.
        #
        # We use a band-pass approach to isolate DNL from model errors:
        # 1. Compute raw ratio = data / model
        # 2. Smooth the ratio with a wide window to capture the trend
        #    (smooth model errors + baseline)
        # 3. DNL = ratio / trend → extracts only short-range
        #    channel-to-channel variations
        _dnl_cfg = {}
        if cfg is not None:
            _dnl_cfg = cfg.get("spectral_fit", {}).get("dnl_correction", {})
        # When a pre-corrected histogram was supplied (two-stage pipeline),
        # skip all DNL iterations -- correction was already applied.
        if pre_binned_hist is not None:
            _dnl_iters = 0
        else:
            _dnl_iters = (
                int(_dnl_cfg.get("iterations", 1))
                if _dnl_cfg.get("enabled", False)
                else 0
            )
        _fourier_coeffs = {}  # populated by self-estimation; must exist for metadata block
        _dnl_accum = np.ones_like(hist, dtype=float)  # identity until DNL applied
        _pre_dnl_errs = {p: float(m.errors[p]) for p in param_order}  # baseline errors
        # B2: Load external DNL map from file if configured.  This takes
        # precedence over self-estimation and provides independent Poisson
        # statistics (no self-reference covariance).
        _external_dnl = flags.get("external_dnl_factors") if flags else None
        _ext_map_path = _dnl_cfg.get("external_map_path")
        if _external_dnl is None and _ext_map_path and _dnl_iters > 0:
            try:
                _ext_map = np.load(_ext_map_path)
                if _ext_map.shape == hist.shape:
                    _external_dnl = _ext_map
                    logging.info(
                        "Loaded external DNL map from %s (%d bins)",
                        _ext_map_path, _ext_map.size,
                    )
                else:
                    logging.warning(
                        "External DNL map shape %s != hist shape %s; ignoring",
                        _ext_map.shape, hist.shape,
                    )
            except Exception as _load_err:
                logging.warning(
                    "Failed to load external DNL map from %s: %s",
                    _ext_map_path, _load_err,
                )
        if _external_dnl is not None:
            _external_dnl = np.asarray(_external_dnl, dtype=float)
            if _external_dnl.shape == hist.shape:
                _dnl_accum = _external_dnl.copy()
                _dnl_window = int(_dnl_cfg.get("smooth_window", 21))
                width_lookup = _WidthLookup(
                    centers, widths_base * _dnl_accum
                )
                # Refit with external DNL widths
                m_ext = Minuit(_nll, *[float(m.values[p]) for p in param_order],
                               name=param_order)
                m_ext.errordef = Minuit.LIKELIHOOD
                for name, lo, hi in zip(param_order, bounds_lo, bounds_hi):
                    m_ext.limits[name] = (lo, hi)
                    if flags.get(f"fix_{name}", False):
                        m_ext.fixed[name] = True
                    prev_err = float(m.errors[name])
                    if prev_err > 0:
                        m_ext.errors[name] = prev_err
                    else:
                        m_ext.errors[name] = (hi - lo) * 0.01
                m_ext.migrad()
                if not m_ext.valid:
                    m_ext.simplex()
                    m_ext.migrad()
                m = m_ext
                _dnl_iters = 1  # flag that DNL was applied
                _dnl_meta = {
                    "dnl_applied": True,
                    "dnl_iterations": 0,
                    "dnl_smooth_window": _dnl_window,
                    "dnl_factors": _dnl_accum.tolist(),
                    "operator_class": "external_fixed",
                    "calibration_source": "external",
                    "n_corrected_bins": int(np.sum(_dnl_accum != 1.0)),
                    "statistical_model": "rescaled_independent_poisson",
                    "covariance_note": (
                        "External fixed DNL map: bins are rescaled but "
                        "conditionally independent Poisson. Standard "
                        "per-bin residuals are valid."
                    ),
                }
            else:
                logging.warning(
                    "External DNL factors shape %s != hist shape %s; ignoring",
                    _external_dnl.shape, hist.shape,
                )
                _external_dnl = None
        _t0_dnl = _time_mod.perf_counter()
        _nll_calls_before_dnl = _nll_call_count[0]
        if _external_dnl is None and _dnl_iters > 0:
            from scipy.ndimage import uniform_filter1d as _uf1d

            _dnl_min = float(_dnl_cfg.get("min_counts", 50.0))
            _dnl_max = float(_dnl_cfg.get("max_correction", 0.5))
            _dnl_window = int(_dnl_cfg.get("smooth_window", 21))
            if _dnl_window % 2 == 0:
                _dnl_window += 1  # ensure odd

            # Save pre-DNL MIGRAD errors as a reliable baseline for
            # covariance recovery.  Post-DNL refits may produce zero
            # errors when starting at a converged point.
            _pre_dnl_errs = {p: float(m.errors[p]) for p in param_order}

            # Accumulate DNL across iterations (each iteration
            # estimates the *residual* DNL on top of the current
            # correction, then multiplies it in).
            _dnl_accum = np.ones_like(hist, dtype=float)

            for _di in range(_dnl_iters):
                _popt_cur = np.array([m.values[p] for p in param_order])
                _model_cur = _model_binned(centers, *_popt_cur)

                # Step 1: raw ratio of observed to predicted
                _dnl_mask = _model_cur > _dnl_min
                _ratio = np.ones_like(hist, dtype=float)
                _ratio[_dnl_mask] = (
                    hist[_dnl_mask].astype(float) / _model_cur[_dnl_mask]
                )

                # Step 2: smooth the ratio to capture long-range trends
                # (model shape errors vary on 10-100 channel scales)
                _ratio_smooth = _uf1d(
                    _ratio, size=_dnl_window, mode="nearest"
                )

                # Step 3: residual DNL = ratio / trend
                _dnl_resid = np.ones_like(hist, dtype=float)
                _valid = _dnl_mask & (_ratio_smooth > 0.1)
                _dnl_resid[_valid] = (
                    _ratio[_valid] / _ratio_smooth[_valid]
                )

                # Fill interior gaps in the DNL via linear interpolation.
                # DNL is a hardware property of the ADC channel, so bins
                # that fall below the min_counts threshold *between* valid
                # regions should be interpolated.  Bins beyond the last
                # valid index (edge extrapolation) stay at 1.0  - we cannot
                # reliably estimate DNL where the model has negligible
                # counts, and flat extrapolation of a noisy boundary value
                # introduces artefacts.
                _invalid = ~_valid
                if np.any(_valid) and np.any(_invalid):
                    _valid_idx = np.flatnonzero(_valid)
                    _invalid_idx = np.flatnonzero(_invalid)
                    # Only interpolate *interior* gaps (between first and
                    # last valid bins).  Edge bins stay at 1.0.
                    _lo_valid = _valid_idx[0]
                    _hi_valid = _valid_idx[-1]
                    _interior = _invalid_idx[
                        (_invalid_idx > _lo_valid) & (_invalid_idx < _hi_valid)
                    ]
                    if _interior.size > 0:
                        _dnl_resid[_interior] = np.interp(
                            _interior, _valid_idx,
                            _dnl_resid[_valid_idx],
                        )

                # Optional Fourier parameterization: replace per-bin
                # residual with a low-DOF Fourier model at ADC bit-cycling
                # periods.  This prevents the DNL from absorbing model
                # misfit and gives ~14 effective DOF instead of ~182.
                #
                # Modes:
                #   parameterized=True:  ALL iterations use Fourier only
                #   parameterized="hybrid": iteration 0 uses Fourier,
                #       subsequent iterations use per-bin bandpass to
                #       capture individual bin spikes not modeled by
                #       the smooth Fourier basis.
                #   parameterized=False: ALL iterations use bandpass
                _parameterized_dnl = _dnl_cfg.get("parameterized", False)
                _hybrid_dnl = (
                    str(_parameterized_dnl).lower() == "hybrid"
                )
                _use_fourier_this_iter = (
                    _parameterized_dnl is True
                    or (_hybrid_dnl and _di == 0)
                )
                _fourier_coeffs_iter = {}
                if _use_fourier_this_iter:
                    _fourier_periods = _dnl_cfg.get(
                        "fourier_periods_codes",
                        [4, 8, 16, 32, 64, 128, 256, 512],
                    )
                    _adc_bw = 3  # default
                    if cfg is not None:
                        _adc_bw = int(cfg.get("spectral_fit", {}).get(
                            "adc_bin_width", 3))
                    _dnl_resid, _fourier_coeffs_iter = _fit_fourier_dnl(
                        _dnl_resid - 1.0,
                        np.arange(len(centers)),
                        _adc_bw,
                        _fourier_periods,
                        _valid,
                    )
                    # _fit_fourier_dnl returns factors (1+model); _dnl_resid
                    # is already in the right form for accumulation.
                if _fourier_coeffs_iter:
                    _fourier_coeffs = _fourier_coeffs_iter

                # Accumulate and clip
                _dnl_accum = _dnl_accum * _dnl_resid
                _dnl_accum = np.clip(
                    _dnl_accum, 1.0 - _dnl_max, 1.0 + _dnl_max
                )

                # Update bin widths with accumulated DNL correction
                width_lookup = _WidthLookup(
                    centers, widths_base * _dnl_accum
                )

                # Refit with DNL-corrected widths
                _dnl_prev = _dnl_accum_before = _dnl_accum.copy()
                try:
                    m_dnl = Minuit(_nll, *_popt_cur, name=param_order)
                    m_dnl.errordef = Minuit.LIKELIHOOD
                    for name, lo, hi in zip(
                        param_order, bounds_lo, bounds_hi
                    ):
                        m_dnl.limits[name] = (lo, hi)
                        if flags.get(f"fix_{name}", False):
                            m_dnl.fixed[name] = True
                        # Seed step sizes from previous fit so MIGRAD errors
                        # are non-zero even when starting at the minimum.
                        prev_err = float(m.errors[name])
                        if prev_err > 0:
                            m_dnl.errors[name] = prev_err
                        else:
                            m_dnl.errors[name] = (hi - lo) * 0.01
                    m_dnl.migrad()
                    if not m_dnl.valid:
                        m_dnl.simplex()
                        m_dnl.migrad()
                    m = m_dnl  # Accept the successful fit
                except Exception as _dnl_exc:
                    logging.warning(
                        "DNL iteration %d/%d failed (%s); "
                        "reverting to previous correction",
                        _di + 1,
                        _dnl_iters,
                        _dnl_exc,
                    )
                    _dnl_accum = _dnl_prev
                    width_lookup = _WidthLookup(
                        centers, widths_base * _dnl_accum
                    )
                    break

                _n_corr = int(np.sum(_valid))
                logging.info(
                    "DNL iteration %d/%d: NLL=%.1f, "
                    "%d/%d bins corrected, "
                    "accum dnl range [%.4f, %.4f], mean %.4f",
                    _di + 1,
                    _dnl_iters,
                    m.fval,
                    _n_corr,
                    len(_dnl_accum),
                    float(_dnl_accum.min()),
                    float(_dnl_accum.max()),
                    float(_dnl_accum.mean()),
                )

        # Store DNL metadata in the result (if correction was applied)
        _dnl_meta = {}
        if pre_dnl_meta is not None:
            _dnl_meta = dict(pre_dnl_meta)
            logging.info(
                "Using pre_dnl_meta: %d keys, dnl_applied=%s",
                len(_dnl_meta),
                _dnl_meta.get("dnl_applied"),
            )
        elif _dnl_iters > 0:
            _dnl_meta["dnl_applied"] = True
            _dnl_meta["dnl_iterations"] = _dnl_iters
            _dnl_meta["dnl_smooth_window"] = _dnl_window
            _dnl_meta["dnl_factors"] = _dnl_accum.tolist()
            # Statistical classification of the DNL operator
            _parameterized_dnl_final = _dnl_cfg.get("parameterized", False)
            if str(_parameterized_dnl_final).lower() == "hybrid" and _fourier_coeffs:
                _dnl_meta["operator_class"] = "self_estimated_hybrid"
                _dnl_meta["fourier_coefficients"] = {
                    str(k): list(v) for k, v in _fourier_coeffs.items()
                }
                # Hybrid: Fourier DOF + bandpass DOF from subsequent iters
                _fourier_dof = 2 * len(_fourier_coeffs)
                _bandpass_iters = max(0, _dnl_iters - 1)
                _dnl_meta["effective_dnl_params"] = (
                    _fourier_dof + _bandpass_iters * _n_corr / _dnl_window
                    if _dnl_window > 0 else _fourier_dof
                )
            elif _parameterized_dnl_final is True and _fourier_coeffs:
                _dnl_meta["operator_class"] = "self_estimated_fourier"
                _dnl_meta["fourier_coefficients"] = {
                    str(k): list(v) for k, v in _fourier_coeffs.items()
                }
                _dnl_meta["effective_dnl_params"] = 2 * len(_fourier_coeffs)
            else:
                _dnl_meta["operator_class"] = "self_estimated_bandpass"
            _dnl_meta["calibration_source"] = "self"
            _corr_mask = _dnl_accum != 1.0
            _n_corr = int(np.sum(_corr_mask))
            _dnl_meta["n_corrected_bins"] = _n_corr
            if not (_parameterized_dnl_final and _fourier_coeffs):
                _dnl_meta["effective_dnl_params"] = (
                    float(_dnl_iters * _n_corr / _dnl_window) if _dnl_window > 0 else 0.0
                )
            _dnl_dev = _dnl_accum[_corr_mask] - 1.0 if _n_corr > 0 else np.array([])
            _dnl_meta["dnl_amplitude_rms"] = (
                float(np.sqrt(np.mean(_dnl_dev ** 2))) if _dnl_dev.size > 0 else 0.0
            )
            _dnl_meta["statistical_model"] = (
                "approximate_independent_poisson_with_smoothing_covariance"
            )
            _dnl_meta["covariance_note"] = (
                "Self-estimated band-pass DNL introduces smoothing covariance "
                "between bins via uniform_filter1d(W=%d). Naive independent-bin "
                "residuals are approximate; covariance-aware (whitened) "
                "residuals are needed for calibrated diagnostics."
                % _dnl_window
            )
            _dnl_meta["poisson_nll_note"] = (
                "independent Poisson NLL is approximate under self-estimated DNL"
            )
            # Warn if DNL amplitude is large enough to invalidate Poisson approx
            if _dnl_meta["dnl_amplitude_rms"] > 0.05:
                logging.warning(
                    "DNL amplitude RMS = %.3f (>5%%): independent Poisson NLL "
                    "approximation may be poor; consider pseudoexperiment "
                    "calibration of residual statistics",
                    _dnl_meta["dnl_amplitude_rms"],
                )

        _fit_timers["dnl_s"] = _time_mod.perf_counter() - _t0_dnl
        _fit_timers["dnl_nll_calls"] = _nll_call_count[0] - _nll_calls_before_dnl

        _n_free = sum(1 for p in param_order if not m.fixed[p])
        ndf = max(1, hist.size - _n_free)
        out = {}
        if _dnl_meta:
            out["_dnl"] = _dnl_meta
            logging.info("Storing _dnl in output: %d keys", len(_dnl_meta))
        else:
            logging.info("No _dnl metadata to store (pre_dnl_meta=%s, _dnl_iters=%d)",
                        pre_dnl_meta is not None, _dnl_iters)
        param_index = {name: i for i, name in enumerate(param_order)}

        # Freeze the best-fit parameter values from the DNL-converged minimum.
        # Subsequent covariance attempts (Hesse, MINOS) may internally move
        # parameters  - we always return popt from the original minimum.
        popt = np.array([float(m.values[p]) for p in param_order])
        nll_val = float(m.fval)

        # Save MIGRAD approximate errors *before* hesse()  - hesse resets
        # m.errors to zero when the Hessian computation fails.
        # Prefer pre-DNL MIGRAD errors when the post-DNL refit left them
        # at zero (common when starting at a converged point).
        _migrad_errs = {}
        _migrad_is_fallback = {}  # track which errors are fallbacks
        for _pi, _pn in enumerate(param_order):
            _me = float(m.errors[_pn])
            if _me <= 0 and _dnl_iters > 0:
                _me = _pre_dnl_errs.get(_pn, 0.0)
            if _me <= 0:
                # Last resort: 1% of the parameter range
                _me = (bounds_hi[_pi] - bounds_lo[_pi]) * 0.01
                _migrad_is_fallback[_pn] = True
            _migrad_errs[_pn] = _me

        # ----------------------------------------------------------
        # Covariance recovery strategy:
        #   1. Try Hesse (exact second derivatives)
        #   2. If Hesse fails (params at bounds), relax bounds and retry
        #   3. If still fails, compute numerical Hessian via finite diffs
        #   4. Run MINOS for asymmetric errors (per-parameter)
        # ----------------------------------------------------------
        _t0_cov = _time_mod.perf_counter()
        _nll_calls_before_cov = _nll_call_count[0]
        _covariance_method = "none"
        m.hesse()
        cov_raw = m.covariance
        covariance_available = cov_raw is not None
        if covariance_available:
            _covariance_method = "hesse"

        # --- A2: Simplex→MIGRAD→HESSE recovery (Tier 1.5) ---
        # Official iminuit guidance notes that Simplex before MIGRAD
        # can rescue pathological curvature by finding a better
        # starting region for the gradient-based minimiser.
        if cov_raw is None:
            logging.info(
                "Hesse failed  - trying Simplex→MIGRAD→HESSE recovery"
            )
            _saved_vals = popt.copy()
            m.simplex()
            m.migrad()
            if m.valid:
                m.hesse()
                cov_raw = m.covariance
                if cov_raw is not None:
                    _covariance_method = "hesse_after_simplex"
                    covariance_available = True
                    # If Simplex+MIGRAD found the same or better minimum,
                    # update popt; otherwise restore.
                    _new_nll = float(m.fval)
                    if _new_nll <= nll_val + 0.01:
                        popt = np.array(
                            [float(m.values[p]) for p in param_order]
                        )
                        nll_val = _new_nll
                    else:
                        for _pi, _pn in enumerate(param_order):
                            m.values[_pn] = float(_saved_vals[_pi])
                else:
                    # Simplex+MIGRAD converged but HESSE still fails
                    for _pi, _pn in enumerate(param_order):
                        m.values[_pn] = float(_saved_vals[_pi])
            else:
                for _pi, _pn in enumerate(param_order):
                    m.values[_pn] = float(_saved_vals[_pi])

        # --- Strategy 2: relax bounds for params sitting at limits ---
        if cov_raw is None:
            logging.info(
                "Hesse failed  - attempting with relaxed bounds for "
                "parameters at their limits"
            )
            # Reset parameter values to the saved minimum before retrying
            for _pi, _pn in enumerate(param_order):
                m.values[_pn] = float(popt[_pi])
            _at_bound = []
            _relax_frac = 0.05  # expand bound by 5% of range
            for _pi, _pn in enumerate(param_order):
                if m.fixed[_pn]:
                    continue
                _val = float(popt[_pi])
                _lo, _hi = bounds_lo[_pi], bounds_hi[_pi]
                _rng = _hi - _lo
                _tol = _rng * 1e-4
                if _val <= _lo + _tol or _val >= _hi - _tol:
                    _at_bound.append((_pn, _pi, _val, _lo, _hi))
                    _new_lo = _lo - _relax_frac * _rng if _val <= _lo + _tol else _lo
                    _new_hi = _hi + _relax_frac * _rng if _val >= _hi - _tol else _hi
                    m.limits[_pn] = (_new_lo, _new_hi)
            if _at_bound:
                logging.info(
                    "Relaxed bounds for %d params at limits: %s",
                    len(_at_bound),
                    [t[0] for t in _at_bound],
                )
                m.hesse()
                cov_raw = m.covariance
                covariance_available = cov_raw is not None
                # Restore original bounds
                for _pn, _pi, _val, _lo, _hi in _at_bound:
                    m.limits[_pn] = (_lo, _hi)
                if cov_raw is not None:
                    _covariance_method = "hesse_relaxed"
                    logging.info("Hesse succeeded after relaxing bounds")

        # --- Strategy 3: numerical Hessian via finite differences ---
        # Work on the FREE-parameter submatrix only to avoid fixed
        # parameters (with ±1e-12 bounds) distorting eigenvalues.
        if cov_raw is None:
            logging.info(
                "Hesse still failed  - computing numerical Hessian "
                "via finite differences at the converged minimum"
            )
            try:
                _n_par = len(param_order)
                _pvals = popt.copy()  # use saved minimum, not m.values

                # Identify free parameters
                _free_idx = [i for i, p in enumerate(param_order) if not m.fixed[p]]
                _n_free_h = len(_free_idx)

                _steps = np.array([_migrad_errs[param_order[i]] * 0.1 for i in _free_idx])
                # Clamp steps to stay within bounds
                for _si, _pi in enumerate(_free_idx):
                    _max_step = min(
                        _pvals[_pi] - bounds_lo[_pi],
                        bounds_hi[_pi] - _pvals[_pi],
                    ) * 0.5
                    if _max_step > 0:
                        _steps[_si] = min(_steps[_si], _max_step)
                    if _steps[_si] <= 0:
                        _steps[_si] = (bounds_hi[_pi] - bounds_lo[_pi]) * 1e-4

                _hess_free = np.zeros((_n_free_h, _n_free_h))
                _f0 = float(_nll(*_pvals))
                # Diagonal elements (free params only)
                for _si, _pi in enumerate(_free_idx):
                    _pp = _pvals.copy(); _pp[_pi] += _steps[_si]
                    _pm = _pvals.copy(); _pm[_pi] -= _steps[_si]
                    _fp = float(_nll(*_pp))
                    _fm = float(_nll(*_pm))
                    _hess_free[_si, _si] = (_fp - 2 * _f0 + _fm) / (_steps[_si] ** 2)
                # Off-diagonal elements (free params only)
                for _si in range(_n_free_h):
                    _pi = _free_idx[_si]
                    for _sj in range(_si + 1, _n_free_h):
                        _pj = _free_idx[_sj]
                        _ppp = _pvals.copy(); _ppp[_pi] += _steps[_si]; _ppp[_pj] += _steps[_sj]
                        _ppm = _pvals.copy(); _ppm[_pi] += _steps[_si]; _ppm[_pj] -= _steps[_sj]
                        _pmp = _pvals.copy(); _pmp[_pi] -= _steps[_si]; _pmp[_pj] += _steps[_sj]
                        _pmm = _pvals.copy(); _pmm[_pi] -= _steps[_si]; _pmm[_pj] -= _steps[_sj]
                        _fpp = float(_nll(*_ppp))
                        _fpm = float(_nll(*_ppm))
                        _fmp = float(_nll(*_pmp))
                        _fmm = float(_nll(*_pmm))
                        _hess_free[_si, _sj] = (_fpp - _fpm - _fmp + _fmm) / (
                            4.0 * _steps[_si] * _steps[_sj]
                        )
                        _hess_free[_sj, _si] = _hess_free[_si, _sj]

                # Invert the free-parameter submatrix via eigenvalue
                # regularisation and embed back into the full matrix.
                try:
                    _eigvals, _eigvecs = np.linalg.eigh(_hess_free)
                    _eig_thresh = max(1e-10, 1e-6 * np.max(np.abs(_eigvals)))
                    _n_clamped = int(np.sum(_eigvals < _eig_thresh))
                    _eigvals_reg = np.where(
                        _eigvals < _eig_thresh, _eig_thresh, _eigvals
                    )
                    # Regularised inverse of the free submatrix
                    _cov_free = (_eigvecs / _eigvals_reg) @ _eigvecs.T
                    _cov_free = 0.5 * (_cov_free + _cov_free.T)

                    # For free params where numerical Hessian gives a
                    # smaller variance than MIGRAD, use MIGRAD as a floor.
                    for _si, _pi in enumerate(_free_idx):
                        _pn = param_order[_pi]
                        _migrad_var = _migrad_errs[_pn] ** 2
                        if _cov_free[_si, _si] < _migrad_var:
                            if _migrad_is_fallback.get(_pn, False):
                                _cov_free[_si, _si] = _migrad_var
                            else:
                                _scale = np.sqrt(
                                    _migrad_var / max(_cov_free[_si, _si], 1e-30)
                                )
                                _cov_free[_si, :] *= _scale
                                _cov_free[:, _si] *= _scale

                    # Embed into full covariance matrix (fixed params
                    # get near-zero variance on the diagonal)
                    cov_raw = np.zeros((_n_par, _n_par))
                    for _si, _pi in enumerate(_free_idx):
                        for _sj, _pj in enumerate(_free_idx):
                            cov_raw[_pi, _pj] = _cov_free[_si, _sj]
                    # Set fixed params to tiny variance
                    for _pi in range(_n_par):
                        if _pi not in _free_idx:
                            cov_raw[_pi, _pi] = _migrad_errs[param_order[_pi]] ** 2

                    covariance_available = True
                    _covariance_method = "numerical_hessian"
                    if _n_clamped > 0:
                        logging.info(
                            "Numerical Hessian: %d/%d free-param eigenvalues "
                            "clamped (params at bounds). Off-diagonal "
                            "correlations available for well-determined "
                            "parameters.",
                            _n_clamped, _n_free_h,
                        )
                    else:
                        logging.info(
                            "Numerical Hessian inversion succeeded  - "
                            "full covariance with off-diagonal correlations"
                        )
                except np.linalg.LinAlgError:
                    logging.warning("Numerical Hessian eigendecomposition failed")
                    cov_raw = None
            except Exception as _num_exc:
                logging.warning(
                    "Numerical Hessian computation failed: %s", _num_exc
                )
                cov_raw = None

        # --- Final fallback: MIGRAD diagonal-only ---
        if cov_raw is None:
            _covariance_method = "migrad_diagonal"
            n_nonzero = sum(1 for v in _migrad_errs.values() if v > 0)
            logging.warning(
                "All covariance strategies failed  - using MIGRAD approximate "
                "errors as diagonal fallback (no correlations). "
                "%d / %d params have non-zero MIGRAD errors.",
                n_nonzero, len(param_order),
            )
            diag = np.array(
                [_migrad_errs[p] ** 2 for p in param_order]
            )
            cov_raw = np.diag(diag)

        # --- Asymmetric errors via profile likelihood ---
        # Strategy:
        #   A. Try iminuit's native MINOS (proper re-profiling)
        #   B. Manual profile scan with re-minimisation at each point
        #   C. Projection scan (freeze other params  - last resort)
        _minos_errors = None
        _minos_method = "none"
        _free_params = [p for p in param_order if not m.fixed[p]]
        if skip_minos:
            logging.info("Skipping profile likelihood errors (skip_minos=True)")
            _minos_errors = {}
            _minos_method = "skipped"
        else:
            logging.info(
                "Computing profile likelihood errors for %d free parameters...",
                len(_free_params),
            )

        # --- Strategy A: iminuit native MINOS ---
        # Requires valid minimum + covariance.  Re-profiles over all
        # other parameters at each scan point (true MINOS).
        if not skip_minos and covariance_available and _covariance_method in ("hesse", "hesse_relaxed"):
            try:
                # Restore minimum before MINOS
                for _pi, _pn in enumerate(param_order):
                    m.values[_pn] = float(popt[_pi])
                m.minos()
                _minos_errors = {}
                for _pn in _free_params:
                    _me = m.merrors[_pn]
                    _lo, _hi = float(_me.lower), float(_me.upper)
                    if _me.is_valid:
                        _minos_errors[_pn] = (_lo, _hi)
                if _minos_errors:
                    _minos_method = "iminuit_minos"
                    logging.info(
                        "iminuit MINOS: %d / %d parameters have "
                        "asymmetric errors (true profile likelihood)",
                        len(_minos_errors), len(_free_params),
                    )
                else:
                    _minos_errors = None
                    logging.info(
                        "iminuit MINOS returned no valid intervals "
                        " - falling back to manual profile scan"
                    )
                # Restore minimum (MINOS may have moved params)
                for _pi, _pn in enumerate(param_order):
                    m.values[_pn] = float(popt[_pi])
            except Exception as _minos_exc:
                logging.info(
                    "iminuit MINOS failed (%s)  - falling back to "
                    "manual profile scan", _minos_exc
                )
                _minos_errors = None
                # Restore minimum
                for _pi, _pn in enumerate(param_order):
                    m.values[_pn] = float(popt[_pi])

        # --- Strategy B: manual profile scan with re-minimisation ---
        # Fix one parameter at a time, re-minimise over all others,
        # and find the delta-NLL = 0.5 crossing.  This is equivalent
        # to MINOS but works even when iminuit's MINOS fails.
        # A1: Track the best NLL seen during profiling. If any
        # conditional minimum has lower NLL than the stored popt,
        # a full unconstrained refit is triggered to confirm a
        # better basin (profile points are conditional optima, not
        # automatically the global best-fit parameter vector).
        if _minos_errors is None:
            try:
                _nll_min = float(_nll(*popt))
                _minos_errors = {}
                _target = 0.5  # delta-NLL for 1-sigma
                _profile_best_nll = _nll_min
                _profile_best_vals = None

                # Save fixed-state so we can restore
                _orig_fixed = {p: m.fixed[p] for p in param_order}

                for _pn in _free_params:
                    _pi = param_order.index(_pn)
                    _val = float(popt[_pi])
                    _err = _migrad_errs.get(_pn, 0)
                    if _err <= 0:
                        continue
                    _lo_bnd, _hi_bnd = bounds_lo[_pi], bounds_hi[_pi]

                    def _profile_nll_at(_trial_val):
                        """Fix _pn at _trial_val, re-minimise others, return NLL."""
                        nonlocal _profile_best_nll, _profile_best_vals
                        # Restore all params to minimum
                        for _ri, _rn in enumerate(param_order):
                            m.values[_rn] = float(popt[_ri])
                            m.fixed[_rn] = _orig_fixed[_rn]
                        # Fix the parameter under test
                        m.values[_pn] = _trial_val
                        m.fixed[_pn] = True
                        # Re-minimise over remaining free params
                        m.migrad()
                        _fv = float(m.fval)
                        # A1: Track if this conditional optimum found
                        # a lower NLL than the stored global minimum
                        if _fv < _profile_best_nll - 0.01:
                            _profile_best_nll = _fv
                            _profile_best_vals = np.array(
                                [float(m.values[p]) for p in param_order]
                            )
                        return _fv

                    # Scan downward
                    _lo_cross = None
                    _prev_trial = _val
                    _prev_dnll = 0.0
                    for _mult in np.linspace(0.1, 5.0, 25):
                        _trial = _val - _mult * _err
                        if _trial < _lo_bnd:
                            _trial = _lo_bnd
                        _dnll = _profile_nll_at(_trial) - _nll_min
                        if _dnll >= _target:
                            if _prev_dnll < _target and _dnll > _prev_dnll:
                                _frac = (_target - _prev_dnll) / (_dnll - _prev_dnll)
                                _cross = _prev_trial + _frac * (_trial - _prev_trial)
                            else:
                                _cross = _trial
                            _lo_cross = _cross - _val
                            break
                        if _trial <= _lo_bnd:
                            _lo_cross = _lo_bnd - _val
                            break
                        _prev_trial = _trial
                        _prev_dnll = _dnll

                    # Scan upward
                    _hi_cross = None
                    _prev_trial = _val
                    _prev_dnll = 0.0
                    for _mult in np.linspace(0.1, 5.0, 25):
                        _trial = _val + _mult * _err
                        if _trial > _hi_bnd:
                            _trial = _hi_bnd
                        _dnll = _profile_nll_at(_trial) - _nll_min
                        if _dnll >= _target:
                            if _prev_dnll < _target and _dnll > _prev_dnll:
                                _frac = (_target - _prev_dnll) / (_dnll - _prev_dnll)
                                _cross = _prev_trial + _frac * (_trial - _prev_trial)
                            else:
                                _cross = _trial
                            _hi_cross = _cross - _val
                            break
                        if _trial >= _hi_bnd:
                            _hi_cross = _hi_bnd - _val
                            break
                        _prev_trial = _trial
                        _prev_dnll = _dnll

                    if _lo_cross is not None and _hi_cross is not None:
                        _minos_errors[_pn] = (float(_lo_cross), float(_hi_cross))

                # Restore original state
                for _pi, _pn_r in enumerate(param_order):
                    m.values[_pn_r] = float(popt[_pi])
                    m.fixed[_pn_r] = _orig_fixed[_pn_r]

                # A1: If profile scan found a conditional minimum with
                # lower NLL, attempt a full unconstrained refit from
                # that seed. Only accept if the refit confirms a
                # genuinely lower global minimum.
                if (
                    _profile_best_vals is not None
                    and _profile_best_nll < _nll_min - 0.5
                ):
                    logging.info(
                        "Profile scan found a potentially better basin "
                        "(NLL %.4f vs %.4f, delta=%.4f). Attempting "
                        "full unconstrained refit from profile seed.",
                        _profile_best_nll, _nll_min,
                        _nll_min - _profile_best_nll,
                    )
                    for _pi, _pn_r in enumerate(param_order):
                        m.values[_pn_r] = float(_profile_best_vals[_pi])
                        m.fixed[_pn_r] = _orig_fixed[_pn_r]
                    m.migrad()
                    _refit_nll = float(m.fval)
                    if _refit_nll < _nll_min - 0.01 and m.valid:
                        logging.warning(
                            "Profile-seeded refit confirmed lower minimum: "
                            "NLL %.4f -> %.4f. Updating popt.",
                            _nll_min, _refit_nll,
                        )
                        popt = np.array(
                            [float(m.values[p]) for p in param_order]
                        )
                        nll_val = _refit_nll
                        _nll_min = _refit_nll
                        # Recompute HESSE at the new minimum
                        m.hesse()
                        if m.covariance is not None:
                            pcov = np.array(m.covariance)
                            covariance_available = True
                            _covariance_method = "hesse_after_profile_refit"
                    else:
                        logging.info(
                            "Profile-seeded refit did not confirm a "
                            "lower minimum (NLL=%.4f). Keeping original.",
                            _refit_nll,
                        )
                        # Restore original state
                        for _pi, _pn_r in enumerate(param_order):
                            m.values[_pn_r] = float(popt[_pi])
                            m.fixed[_pn_r] = _orig_fixed[_pn_r]

                if _minos_errors:
                    _minos_method = "profile_scan_reprofiled"
                    logging.info(
                        "Profile scan (re-profiled): %d / %d parameters "
                        "have asymmetric errors",
                        len(_minos_errors), len(_free_params),
                    )
                else:
                    logging.warning(
                        "Profile scan (re-profiled) failed for all parameters"
                    )
                    _minos_errors = None
            except Exception as _prof_exc:
                logging.warning("Profile scan failed: %s", _prof_exc)
                _minos_errors = None
                # Restore state on failure
                for _pi, _pn_r in enumerate(param_order):
                    m.values[_pn_r] = float(popt[_pi])
                    m.fixed[_pn_r] = _orig_fixed.get(_pn_r, False)

        _fit_timers["covariance_minos_s"] = _time_mod.perf_counter() - _t0_cov
        _fit_timers["covariance_minos_nll_calls"] = _nll_call_count[0] - _nll_calls_before_cov
        _fit_timers["total_nll_calls"] = _nll_call_count[0]
        _fit_timers["minos_method"] = _minos_method

        pcov = np.array(cov_raw)
        # popt and nll_val already saved at the top of this block
        if (
            curve_fit is not _ORIG_CURVE_FIT
            and popt_cf is not None
            and pcov_cf is not None
            and _dnl_iters == 0  # Don't use pre-DNL curve_fit results
        ):
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
        _n_free_ub = sum(1 for p in param_order if not m.fixed[p])
        ndf = max(1, e.size - _n_free_ub)
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
        # Validate that perr and param_order have matching sizes
        if len(perr) != len(param_order):
            raise RuntimeError(
                f"Internal error: perr length ({len(perr)}) does not match "
                f"param_order length ({len(param_order)})"
            )
        out["fit_valid"] = fit_valid
        for i, pname in enumerate(param_order):
            val = float(m.values[pname])
            if pname.startswith("S_"):
                out[pname] = float(_softplus(val))
                out["d" + pname] = float(perr[i])
            else:
                out[pname] = val
                out["d" + pname] = float(perr[i])
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
    # Validate that perr and param_order have matching sizes
    if len(perr) != len(param_order):
        raise RuntimeError(
            f"Internal error: perr length ({len(perr)}) does not match "
            f"param_order length ({len(param_order)})"
        )
    out = {}
    for i, name in enumerate(param_order):
        val = float(popt[i])
        if name.startswith("S_"):
            out[name] = float(_softplus(val))
            out["d" + name] = float(perr[i])
        else:
            out[name] = val
            out["d" + name] = float(perr[i])

    # Store MINOS profile-likelihood errors alongside (not replacing) the
    # Hessian/covariance errors.  HESSE gives the local curvature estimate
    # at the unconstrained minimum; MINOS gives profile-likelihood intervals
    # by scanning each parameter while re-minimising the others.  When they
    # disagree, MINOS is preferred for that parameter's interval, but the
    # Hessian covariance matrix (and its diagonal) remains the correct basis
    # for symmetric error propagation and correlations.
    if _minos_errors:
        for _pn, (_mlo, _mhi) in _minos_errors.items():
            out["d" + _pn + "_minos_lo"] = float(_mlo)  # negative by convention
            out["d" + _pn + "_minos_hi"] = float(_mhi)  # positive by convention
        logging.info(
            "MINOS profile-likelihood errors stored for %d parameters "
            "(HESSE errors retained in d<param> keys)",
            len(_minos_errors),
        )

    if fix_sigma0:
        out["sigma0"] = sigma0_val
        out["dsigma0"] = 0.0
    if fix_F:
        out["F"] = F_val
        out["dF"] = 0.0

    out["fit_valid"] = fit_valid
    out["likelihood_path"] = likelihood_path
    out["covariance_method"] = _covariance_method
    out["minos_method"] = _minos_method

    # Attach DNL metadata from the binned path (if correction was applied)
    if not unbinned and _dnl_iters > 0:
        out["_dnl"] = _dnl_meta

    model_counts = _model_binned(centers, *popt)
    # Protect against division by zero: ensure model_counts is always positive
    safe_model_counts = np.maximum(model_counts, 1e-10)
    with np.errstate(divide="ignore", invalid="ignore"):
        chi2 = 2 * np.sum(
            model_counts - hist + hist * np.log(np.where(hist > 0, hist / safe_model_counts, 1.0))
        )
    out["chi2"] = float(chi2)
    n_free = sum(1 for p in param_order if not m.fixed[p])
    ndf = max(1, hist.size - n_free)
    out["chi2_ndf"] = chi2 / ndf if ndf > 0 else np.nan
    out["nll"] = float(nll_val)
    out["n_free_params"] = n_free
    k = n_free
    n_bins = hist.size
    out["aic"] = float(2 * nll_val + 2 * k)
    # BIC penalises complexity more aggressively than AIC for large samples
    out["bic"] = float(2 * nll_val + k * np.log(n_bins))
    # AICc: small-sample corrected AIC (converges to AIC for n >> k)
    if n_bins > k + 1:
        out["aicc"] = float(2 * nll_val + 2 * k + 2 * k * (k + 1) / (n_bins - k - 1))
    else:
        out["aicc"] = float("nan")

    # Effective NDF accounting for DNL correction DOF consumption
    if not unbinned and _dnl_iters > 0 and _dnl_meta:
        _dnl_factors = np.array(_dnl_meta.get("dnl_factors", []))
        _n_corr_bins = int(np.sum(_dnl_factors != 1.0)) if _dnl_factors.size > 0 else 0
        if _n_corr_bins > 0 and _dnl_window > 0:
            _dnl_eff_params = float(_dnl_iters * _n_corr_bins / _dnl_window)
            _ndf_eff = max(1, hist.size - n_free - _dnl_eff_params)
            out["dnl_effective_params"] = _dnl_eff_params
            out["ndf_effective"] = float(_ndf_eff)
            out["chi2_ndf_effective"] = chi2 / _ndf_eff if _ndf_eff > 0 else np.nan

    # ------------------------------------------------------------------
    # Store full model prediction arrays for accurate plotting.
    # The plot_spectrum function previously reconstructed the model from
    # parameters but missed shelf, halo, b2, b3, and DNL corrections.
    # ------------------------------------------------------------------
    raw_map = _build_raw_param_map(popt)
    physical = _physical_params(raw_map)

    # Per-component contributions
    # Note: _physical_params sets BOTH N_{iso} and S_{iso}, and the intensity
    # function checks N_{iso} first, so we must zero both to suppress a peak.
    _plot_components = {}
    for _iso in iso_list:
        _iso_params = dict(physical)
        for _other in iso_list:
            if _other != _iso:
                _iso_params[f"S_{_other}"] = 0.0
                _iso_params[f"N_{_other}"] = 0.0
        _iso_params["S_bkg"] = 0.0
        _iso_density = spectral_intensity(centers, _iso_params, domain)
        _plot_components[_iso] = width_lookup.scale(centers, _iso_density).copy()

    # Background only
    _bkg_params = dict(physical)
    for _iso in iso_list:
        _bkg_params[f"S_{_iso}"] = 0.0
        _bkg_params[f"N_{_iso}"] = 0.0
    _bkg_density = spectral_intensity(centers, _bkg_params, domain)
    _plot_components["Background"] = width_lookup.scale(centers, _bkg_density).copy()

    out["_plot_model_total"] = model_counts
    out["_plot_components"] = _plot_components
    out["_plot_centers"] = centers.copy()
    out["_plot_hist"] = hist.copy()
    out["_plot_edges"] = edges.copy()

    # Store spectral fit timing breakdown
    _fit_timers["total_s"] = (
        _fit_timers.get("initial_migrad_s", 0)
        + _fit_timers.get("dnl_s", 0)
        + _fit_timers.get("covariance_minos_s", 0)
    )
    out["_fit_timers"] = _fit_timers
    logging.info(
        "Spectral fit timing: initial_migrad=%.1fs (%d calls), "
        "dnl=%.1fs (%d calls), cov+minos=%.1fs (%d calls, method=%s), "
        "total=%.1fs (%d NLL evals)",
        _fit_timers.get("initial_migrad_s", 0),
        _fit_timers.get("initial_migrad_nll_calls", 0),
        _fit_timers.get("dnl_s", 0),
        _fit_timers.get("dnl_nll_calls", 0),
        _fit_timers.get("covariance_minos_s", 0),
        _fit_timers.get("covariance_minos_nll_calls", 0),
        _fit_timers.get("minos_method", "n/a"),
        _fit_timers.get("total_s", 0),
        _fit_timers.get("total_nll_calls", 0),
    )

    param_index = {name: i for i, name in enumerate(param_order)}
    out = _finalize_signal_params(out)
    return FitResult(
        out,
        pcov,
        int(ndf),
        param_index,
        counts=int(n_events),
        likelihood=likelihood_mode,
        minos_errors=_minos_errors,
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
        return 1e10  # Large penalty to avoid numerical overflow
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
        if idx >= len(params):
            raise ValueError(f"Parameter index {idx} for '{pname}' is out of range (params length: {len(params)})")
        p[pname] = params[idx]

    # For each isotope, compute its contribution to NLL:
    for iso in iso_list:
        lam = lam_map[iso]
        eff = eff_map[iso]
        if eff is None:
            eff = p[f"eff_{iso}"]
        if eff <= 0:
            return 1e10  # Large penalty to avoid numerical overflow

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
                return 1e10  # Large penalty to avoid numerical overflow
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
    """Fit time-series decay data to extract activity parameters.

    The decay model is: r(t) = eff*[E*(1-exp(-λt)) + λ*N0*exp(-λt)] + B
    where E is steady-state activity (Bq), N0 is initial atom count, B is background
    rate, λ = ln(2)/half_life_s, and eff is detection efficiency. Note that N0 is
    in units of atom count, not activity; the initial activity is λ*N0.

    Parameters
    ----------
    times_dict : mapping of isotope -> array of timestamps in seconds
    weights : dict or None
        Optional mapping of isotope -> per-event weights matching
        ``times_dict``.
    strict : bool, optional
        When ``True`` raise a :class:`RuntimeError` if the covariance matrix
        is not positive definite.  The default is ``False`` which attempts to
        stabilise the matrix by adding a tiny jitter.
    t_start, t_end : float
        Absolute UNIX timestamps (seconds) defining the fit window
    config : dict
        Configuration dict with these keys:
          "isotopes": { "Po214": {"half_life_s": , "efficiency": ,  }, "Po218": {   } }
          "fit_background": bool
          "fit_initial": bool
          "background_guess": float  (initial guess for B_iso)
          "initial_guess":    float  (initial guess for N0_iso atom count)

    Returns
    -------
    dict
        Best fit values & 1σ uncertainties, e.g.:
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

    # Early exit when statistics are insufficient. Use the raw event count here:
    # probability weights change the effective information content but should not
    # make a populated window look empty.
    total_counts = 0
    for iso in iso_list:
        total_counts += len(np.asarray(times_dict.get(iso, [])))

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
        if "E_Po214" in param_index and "N0_Po214" in param_index:
            i1 = param_index["E_Po214"]
            i2 = param_index["N0_Po214"]
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

    if "E_Po214" in param_index and "N0_Po214" in param_index:
        i1 = param_index["E_Po214"]
        i2 = param_index["N0_Po214"]
        out["cov_E_Po214_N0_Po214"] = float(cov[i1, i2])

    return FitResult(out, cov, int(ndf), param_index, counts=int(n_events))


fit_time_series.__rmtest_time_fit_callback_version__ = (1, 1)


# -----------------------------------------------------
# End of fitting.py
# -----------------------------------------------------
