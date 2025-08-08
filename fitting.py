# -----------------------------------------------------
# fitting.py
# -----------------------------------------------------

import logging
import warnings
from dataclasses import dataclass, field
from typing import TypedDict, NotRequired

import numpy as np
import pandas as pd
from iminuit import Minuit
from scipy.optimize import curve_fit, OptimizeWarning
from scipy.stats import chi2
from calibration import emg_left, gaussian
from constants import _TAU_MIN, CURVE_FIT_MAX_EVALS, safe_exp as _safe_exp

# Use shared overflow guard for exponentiation
__all__ = ["fit_time_series", "fit_decay", "fit_spectrum"]


def make_linear_bkg(Emin, Emax, Eref=None):
    """Return a positive, unit-normalised linear background shape."""
    if Eref is None:
        Eref = 0.5 * (Emin + Emax)

    def shape(E, beta0, beta1):
        lin = (E - Eref)
        log_b = beta0 + beta1 * lin
        b = np.exp(log_b)
        grid = np.linspace(Emin, Emax, 512)
        area = np.trapz(np.exp(beta0 + beta1 * (grid - Eref)), grid)
        return b / max(area, 1e-300)

    return shape


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
    S_bkg: NotRequired[float]
    dS_bkg: NotRequired[float]
    beta0: NotRequired[float]
    dbeta0: NotRequired[float]
    beta1: NotRequired[float]
    dbeta1: NotRequired[float]

@dataclass
class FitResult:
    """Container for fit output."""

    params: dict[str, float]
    cov: np.ndarray | None

    ndf: int
    param_index: dict[str, int] | None = None
    counts: int | None = None
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

        if self.cov.ndim >= 2 and i1 < self.cov.shape[0] and i2 < self.cov.shape[1]:
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
    if flags.get("fix_sigma_E"):
        flags.setdefault("fix_sigma0", True)
        flags.setdefault("fix_F", True)

    if flags.get("fix_sigma_E"):
        flags.setdefault("fix_sigma0", True)
        flags.setdefault("fix_F", True)

    t = np.asarray(times, dtype=float)
    if t_end is None:
        T = float(t.max() if t.size > 0 else 0.0) - float(t0)
    else:
        T = float(t_end) - float(t0)

    eff = float(priors.get("eff", (1.0, 0.0))[0])

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
    if flags.get("fix_sigma_E"):
        flags.setdefault("fix_sigma0", True)
        flags.setdefault("fix_F", True)

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
    width_map = dict(zip(centers, widths))
    if not unbinned:
        hist, _ = np.histogram(e, bins=edges)

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

    # Determine which peaks should include an EMG tail based on provided priors
    use_emg = {
        "Po210": "tau_Po210" in priors,
        "Po218": "tau_Po218" in priors,
        "Po214": "tau_Po214" in priors,
    }

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
    for iso in ("Po210", "Po218", "Po214"):
        param_index[f"mu_{iso}"] = len(param_order)
        param_order.append(f"mu_{iso}")
        param_index[f"S_{iso}"] = len(param_order)
        param_order.append(f"S_{iso}")
        if use_emg[iso]:
            param_index[f"tau_{iso}"] = len(param_order)
            param_order.append(f"tau_{iso}")
    param_index["S_bkg"] = len(param_order)
    param_order.append("S_bkg")
    param_index["beta0"] = len(param_order)
    param_order.append("beta0")
    param_index["beta1"] = len(param_order)
    param_order.append("beta1")

    p0 = []
    bounds_lo, bounds_hi = [], []
    eps = 1e-12
    sigma0_mean = sigma0_val
    for name in param_order:
        mean, sig = p(name, 1.0)
        # Enforce a strictly positive initial tau to avoid singular EMG tails
        if name.startswith("tau_"):
            mean = max(mean, _TAU_MIN)
        if flags.get(f"fix_{name}", False) or sig == 0:
            # curve_fit requires lower < upper; use a tiny width around fixed values
            lo = mean - eps
            hi = mean + eps
        else:
            delta = 5 * sig if np.isfinite(sig) else np.inf
            lo = mean - delta
            hi = mean + delta
        if bounds and name in bounds:
            user_lo, user_hi = bounds[name]
            if user_lo is not None:
                lo = max(lo, user_lo)
            if user_hi is not None:
                hi = min(hi, user_hi)
        if name.startswith("tau_"):
            lo = max(lo, _TAU_MIN)
            if max_tau_ratio is not None:
                hi = min(hi, max_tau_ratio * sigma0_mean)
        if name in ("sigma0", "F"):
            lo = max(lo, 0.0)
        if name.startswith("S_"):
            lo = max(lo, 0.0)
        if hi <= lo:
            hi = lo + eps
        mean = np.clip(mean, lo, hi)
        p0.append(mean)
        bounds_lo.append(lo)
        bounds_hi.append(hi)

    iso_list = ["Po210", "Po218", "Po214"]

    bkg_shape_fn = make_linear_bkg(edges[0], edges[-1])

    def _model_density(x, *params):
        if fix_sigma0:
            sigma0 = sigma0_val
        else:
            sigma0 = params[param_index["sigma0"]]
        if fix_F:
            F_current = F_val
        else:
            F_current = params[param_index["F"]]
        y = np.zeros_like(x)
        for iso in iso_list:
            mu = params[param_index[f"mu_{iso}"]]
            S = params[param_index[f"S_{iso}"]]
            if use_emg[iso]:
                tau = params[param_index[f"tau_{iso}"]]
                sigma = np.sqrt(sigma0 ** 2 + F_current * x)
                with np.errstate(over="ignore", divide="ignore", invalid="ignore"):
                    y_emg = emg_left(x, mu, sigma, tau)
                y_emg = np.nan_to_num(y_emg, nan=0.0, posinf=0.0, neginf=0.0)
                y += S * y_emg
            else:
                sigma = np.sqrt(sigma0 ** 2 + F_current * x)
                y += S * gaussian(x, mu, sigma)
        B = params[param_index["S_bkg"]]
        beta0 = params[param_index["beta0"]]
        beta1 = params[param_index["beta1"]]
        return y + B * bkg_shape_fn(x, beta0, beta1)

    def _model_binned(x, *params):
        y = _model_density(x, *params)
        if np.isscalar(x):
            try:
                return y * width_map[x]
            except KeyError as exc:
                raise KeyError(f"{x} not found in width_map") from exc
        x = np.asarray(x)
        out = np.empty_like(x, dtype=float)
        for i, xi in enumerate(x):
            try:
                out[i] = y[i] * width_map[xi]
            except KeyError as exc:
                raise KeyError(f"{xi} not found in width_map") from exc
        return out

    if not unbinned:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message="Covariance of the parameters could not be estimated",
                category=OptimizeWarning,
            )
            popt, pcov = curve_fit(
                _model_binned,
                centers,
                hist,
                p0=p0,
                bounds=(bounds_lo, bounds_hi),
                maxfev=CURVE_FIT_MAX_EVALS,
            )
    else:
        def _nll(*params):
            # Per-event rate must be positive and finite
            rate = _model_density(e, *params)
            if np.any(rate <= 0) or not np.isfinite(rate).all():
                return 1e50

            # Sum of signal yields (non-negative by construction)
            S_sum = 0.0
            for iso in iso_list:
                S_sum += params[param_index[f"S_{iso}"]]
            B = params[param_index["S_bkg"]]
            expected = S_sum + B

            # Guard against pathological negative expectations
            if expected <= 0 or not np.isfinite(expected):
                return 1e50

            return expected - np.sum(np.log(rate))

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
        ndf = e.size - len(param_order)
        out = {}
        param_index = {name: i for i, name in enumerate(param_order)}
        if not m.valid:
            out["fit_valid"] = False
            for pname in param_order:
                out[pname] = float(m.values[pname])
                err = float(m.errors[pname]) if pname in m.errors else np.nan
                out["d" + pname] = err
            cov = np.zeros((len(param_order), len(param_order)))
            k = len(param_order)
            out["aic"] = float(2 * m.fval + 2 * k)
            return FitResult(out, cov, int(ndf), param_index, counts=int(n_events))

        m.hesse()
        cov_obj = m.covariance
        if cov_obj is None:
            cov = np.zeros((len(param_order), len(param_order)))
            perr = np.full(len(param_order), np.nan)
            fit_valid = False
        else:
            cov = np.array(cov_obj)
            perr = np.sqrt(np.clip(np.diag(cov), 0, None))
            try:
                eigvals = np.linalg.eigvals(cov)
                fit_valid = bool(np.all(eigvals > 0))
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
            out[pname] = float(m.values[pname])
            out["d" + pname] = float(perr[i] if i < len(perr) else np.nan)
        if fix_sigma0:
            out["sigma0"] = sigma0_val
            out["dsigma0"] = 0.0
        if fix_F:
            out["F"] = F_val
            out["dF"] = 0.0
        k = len(param_order)
        out["aic"] = float(2 * m.fval + 2 * k)
        return FitResult(out, cov, int(ndf), param_index, counts=int(n_events))

    perr = np.sqrt(np.clip(np.diag(pcov), 0, None))
    try:
        eigvals = np.linalg.eigvals(pcov)
        fit_valid = bool(np.all(eigvals > 0))
    except np.linalg.LinAlgError:
        fit_valid = False

    if not fit_valid:
        if strict:
            raise RuntimeError(
                "fit_spectrum: covariance matrix not positive definite"
            )
        logging.warning("fit_spectrum: covariance matrix not positive definite")
        # Add a small diagonal jitter to attempt stabilising the matrix
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
        out[name] = float(popt[i])
        out["d" + name] = float(perr[i])

    if fix_sigma0:
        out["sigma0"] = sigma0_val
        out["dsigma0"] = 0.0
    if fix_F:
        out["F"] = F_val
        out["dF"] = 0.0

    out["fit_valid"] = fit_valid

    ndf = hist.size - len(popt)
    model_counts = _model_binned(centers, *popt)
    chi2 = float(np.sum(((hist - model_counts) ** 2) / np.clip(hist, 1, None)))
    out["chi2"] = chi2
    out["chi2_ndf"] = chi2 / ndf if ndf != 0 else np.nan
    k = len(popt)
    out["aic"] = float(chi2 + 2 * k)
    param_index = {name: i for i, name in enumerate(param_order)}
    return FitResult(out, pcov, int(ndf), param_index, counts=int(n_events))


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
        logging.debug("_integral_model called with negative T; using |T|")
        T = abs(T)
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

        # Extract parameters (some may be fixed to zero):
        E_iso = p[f"E_{iso}"]
        B_iso = 0.0 if fix_b_map[iso] else p[f"B_{iso}"]
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
                logging.debug(
                    "fit_time_series: negative relative times detected; check t_start"
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


def fit_time_series(times_dict, t_start, t_end, config, weights=None, strict=False):
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
        # Require at least one event by default; otherwise use the observed count
        min_counts = max(1, int(total_counts))

    if total_counts < min_counts:
        logger.info(
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
    ndf = n_events - len(ordered_params)

    out = {}
    param_index = dict(param_indices)
    if not m.valid:
        out["fit_valid"] = False
        for pname in ordered_params:
            val = float(m.values[pname])
            err = float(m.errors[pname]) if pname in m.errors else np.nan
            out[pname] = val
            out["d" + pname] = err
        cov = np.zeros((len(ordered_params), len(ordered_params)))
        if "E_Po214" in ordered_params and "N0_Po214" in ordered_params:
            i1 = ordered_params.index("E_Po214")
            i2 = ordered_params.index("N0_Po214")
            out["cov_E_Po214_N0_Po214"] = float(cov[i1, i2])
        return FitResult(out, cov, int(ndf), param_index, counts=int(n_events))

    m.hesse()  # compute uncertainties
    cov = np.array(m.covariance)
    perr = np.sqrt(np.clip(np.diag(cov), 0, None))
    try:
        eigvals = np.linalg.eigvals(cov)
        fit_valid = bool(np.all(eigvals > 0))
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
    try:
        crit = chi2.ppf(0.95, df=len(iso_list))
    except Exception:
        crit = 0.0
    fit_valid = fit_valid and ts_val >= crit
    out["fit_valid"] = fit_valid
    for i, pname in enumerate(ordered_params):
        out[pname] = float(m.values[pname])
        out["d" + pname] = float(perr[i] if i < len(perr) else np.nan)

    if "E_Po214" in ordered_params and "N0_Po214" in ordered_params:
        i1 = ordered_params.index("E_Po214")
        i2 = ordered_params.index("N0_Po214")
        out["cov_E_Po214_N0_Po214"] = float(cov[i1, i2])

    return FitResult(out, cov, int(ndf), param_index, counts=int(n_events))


# -----------------------------------------------------
# End of fitting.py
# -----------------------------------------------------
