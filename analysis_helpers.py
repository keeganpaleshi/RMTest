import logging
import math
import time
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping, Sequence, cast

import numpy as np
import pandas as pd
from scipy.stats import norm

import radon_activity
from calibration import apply_calibration
from constants import (
    DEFAULT_KNOWN_ENERGIES,
    PO210,
    PO214,
    PO218,
    RN222,
)
from fitting import FitParams, FitResult, fit_spectrum
from io_utils import apply_burst_filter
from plot_utils import _build_time_segments, _resolve_run_periods
from utils import to_utc_datetime
from utils.time_utils import parse_timestamp, tz_convert_utc

logger = logging.getLogger(__name__)

NUCLIDES = {
    "Po210": PO210,
    "Po214": PO214,
    "Po218": PO218,
    "Rn222": RN222,
}


class PipelineTimer:
    """Simple helper to time major sections of the analysis pipeline."""

    def __init__(self, logger: logging.Logger | None = None):
        self.logger = logger or logging.getLogger(__name__)
        self._start = time.perf_counter()
        self._sections: list[tuple[str, float]] = []

    @contextmanager
    def section(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            duration = time.perf_counter() - start
            self._sections.append((name, duration))
            self.logger.info("⏱️ %s took %.2f s", name, duration)

    def report(self):
        if not self._sections:
            return
        total = time.perf_counter() - self._start
        lines = [f"Pipeline timing summary (total {total:.2f} s):"]
        lines.extend(f"  • {name}: {duration:.2f} s" for name, duration in self._sections)
        self.logger.info("\n".join(lines))


def _hl_value(cfg: Mapping[str, Any], iso: str) -> float:
    """Return the half-life in seconds for ``iso`` using configuration ``cfg``.

    When the configuration does not specify a value or it is ``None`` the
    constant from :mod:`constants` is used.
    """
    val = cfg.get("time_fit", {}).get(f"hl_{iso.lower()}")
    if isinstance(val, list):
        val = val[0] if val else None
    if val is None:
        consts = cfg.get("nuclide_constants", {})
        val = consts.get(iso, NUCLIDES[iso]).half_life_s
    return float(val)


def _radon_background_mode(
    cfg: Mapping[str, Any],
    time_fit_results: Mapping[str, Any],
) -> str | None:
    """Infer the background treatment used by the time-series fit."""

    tf_cfg = cfg.get("time_fit")
    if not isinstance(tf_cfg, Mapping):
        tf_cfg = {}

    flags = tf_cfg.get("flags")
    if not isinstance(flags, Mapping):
        flags = {}

    if flags.get("fix_background_b"):
        return "fixed_from_baseline"

    iso_candidates = ("Po214", "Po218")
    for iso in iso_candidates:
        fit_obj = time_fit_results.get(iso)
        if fit_obj is None:
            continue

        param_index = None
        params: Mapping[str, Any] | None = None

        if isinstance(fit_obj, FitResult):
            param_index = getattr(fit_obj, "param_index", None)
            params = getattr(fit_obj, "params", None)
        elif isinstance(fit_obj, Mapping):
            param_index = fit_obj.get("param_index")
            params = fit_obj

        key = f"B_{iso}"
        if isinstance(param_index, Mapping) and key in param_index:
            return "floated"

        if isinstance(params, Mapping):
            err_key = f"d{key}"
            val = params.get(err_key)
            if val is not None:
                try:
                    if not np.isclose(float(val), 0.0):
                        return "floated"
                except (TypeError, ValueError):
                    return "floated"
            if key in params:
                return "fixed_from_baseline"

    return "floated"


def _eff_prior(eff_cfg: Any) -> tuple[float, float]:
    """Return efficiency prior ``(mean, sigma)`` from configuration.

    ``None`` or the string ``"null"`` yields a flat prior ``(1.0, 1e6)``.
    Lists or tuples are returned as-is. Numeric values get a 5 % width.
    """
    if eff_cfg in (None, "null"):
        return (1.0, 1e6)
    if isinstance(eff_cfg, (list, tuple)):
        if len(eff_cfg) != 2:
            raise ValueError(
                f"Efficiency prior must be a 2-tuple (mean, sigma), got {len(eff_cfg)} elements"
            )
        return tuple(eff_cfg)
    val = float(eff_cfg)
    return (val, 0.05 * val)


def _roi_diff(pre: np.ndarray, post: np.ndarray, cfg: Mapping[str, Any]) -> dict:
    """Return counts difference per ROI between post and pre arrays."""
    diff = {}
    for iso in ("Po210", "Po218", "Po214"):
        win = cfg.get("time_fit", {}).get(f"window_{iso.lower()}")
        if win is None:
            continue
        if not isinstance(win, (list, tuple)) or len(win) != 2:
            continue
        lo, hi = win
        c_pre = int(((pre >= lo) & (pre <= hi)).sum())
        c_post = int(((post >= lo) & (post <= hi)).sum())
        diff[iso] = c_post - c_pre
    return diff


def _burst_sensitivity_scan(
    events: pd.DataFrame, cfg: Mapping[str, Any], cal_result
) -> tuple[dict, tuple[int, int]]:
    """Evaluate radon activity over a grid of burst parameters."""
    from radon_joint_estimator import estimate_radon_activity

    mult0 = int(cfg.get("burst_filter", {}).get("burst_multiplier", 5))
    win0 = int(cfg.get("burst_filter", {}).get("burst_window_size_s", 60))
    mult_values = [max(1, mult0 - 2), mult0, mult0 + 2]
    win_values = [max(1, win0 // 2), win0, win0 * 2]

    results = {}
    for m in mult_values:
        for w in win_values:
            local_cfg = {
                "burst_filter": {"burst_window_size_s": w, "burst_multiplier": m}
            }
            filtered, _ = apply_burst_filter(events, local_cfg, mode="rate")
            if filtered.empty:
                results[(m, w)] = 0.0
                continue
            timestamps = pd.to_datetime(filtered["timestamp"], utc=True, errors="coerce")
            if timestamps.isna().all():
                results[(m, w)] = 0.0
                continue
            t_min = timestamps.min()
            t_max = timestamps.max()
            if pd.isna(t_min) or pd.isna(t_max):
                results[(m, w)] = 0.0
                continue
            live_time_s = (t_max - t_min).total_seconds()
            if not np.isfinite(live_time_s) or live_time_s <= 0:
                results[(m, w)] = 0.0
                continue
            energies = cal_result.predict(filtered["adc"])
            counts = {}
            for iso in ("Po218", "Po214"):
                win = cfg.get("time_fit", {}).get(f"window_{iso.lower()}")
                if win is None:
                    counts[iso] = 0
                else:
                    counts[iso] = int(
                        ((energies >= win[0]) & (energies <= win[1])).sum()
                    )
            eff214 = cfg.get("time_fit", {}).get("eff_po214")
            eff214 = (
                eff214[0]
                if isinstance(eff214, list) and len(eff214) > 0
                else (eff214 if eff214 is not None else 1.0)
            )
            eff218 = cfg.get("time_fit", {}).get("eff_po218")
            eff218 = (
                eff218[0]
                if isinstance(eff218, list) and len(eff218) > 0
                else (eff218 if eff218 is not None else 1.0)
            )
            est = estimate_radon_activity(
                N218=counts.get("Po218"),
                epsilon218=eff218,
                f218=1.0,
                N214=counts.get("Po214"),
                epsilon214=eff214,
                f214=1.0,
                live_time218_s=live_time_s,
                live_time214_s=live_time_s,
            )
            results[(m, w)] = float(
                est.get("Rn_activity_Bq", 0.0) if isinstance(est, dict) else 0.0
            )

    mean_val = np.nanmean(list(results.values())) if results else 0.0
    best = (
        min(results.items(), key=lambda kv: abs(kv[1] - mean_val))[0]
        if results
        else (mult0, win0)
    )
    return results, best


def _save_stub_spectrum_plot(
    energies: Sequence[float] | np.ndarray,
    out_png: Path,
    *,
    bins: int | None = None,
    bin_edges: Sequence[float] | np.ndarray | None = None,
    config: Mapping[str, Any] | None = None,
) -> Path:
    """Write a fallback spectrum plot when the spectral fit is unavailable."""

    import matplotlib.pyplot as plt

    energies_arr = np.asarray(energies, dtype=float)
    if energies_arr.size == 0:
        # ``np.histogram`` handles empty arrays but benefits from a finite range.
        energies_arr = np.asarray([0.0], dtype=float)

    if bin_edges is not None:
        hist, edges = np.histogram(energies_arr, bins=np.asarray(bin_edges, dtype=float))
    else:
        hist, edges = np.histogram(
            energies_arr,
            bins=bins if bins is not None else 400,
        )

    width = np.diff(edges)
    centers = edges[:-1] + width / 2.0

    hist_color = "#808080"
    if isinstance(config, Mapping):
        from color_schemes import COLOR_SCHEMES

        palette_name = str(config.get("palette", "default"))
        palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
        hist_color = palette.get("hist", hist_color)

    fig, ax = plt.subplots(figsize=(8, 6))
    if hist.size:
        draw_width = width if width.size else 1.0
        ax.bar(
            centers,
            hist,
            width=draw_width,
            color=hist_color,
            alpha=0.7,
            label="Data",
        )

    ax.set_title("Energy Spectrum")
    ax.set_xlabel("Energy [MeV]")
    ax.set_ylabel("Counts per bin")
    ax.text(
        0.5,
        0.85,
        "Spectral fit unavailable",
        transform=ax.transAxes,
        ha="center",
        va="center",
        fontsize=14,
        fontweight="bold",
        color="#aa0000",
        bbox={"facecolor": "white", "alpha": 0.85, "edgecolor": "#aa0000"},
    )

    fig.tight_layout()

    out_path = Path(out_png)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=300)

    plt.close(fig)

    return out_path


def _total_radon_series(activity, errors, monitor_volume, sample_volume):
    """Return total radon Bq and uncertainties for a time series."""

    activity_arr = np.asarray(activity, dtype=float)
    err_arr = None if errors is None else np.asarray(errors, dtype=float)

    total = np.empty_like(activity_arr, dtype=float)
    total_err = None if err_arr is None else np.empty_like(err_arr, dtype=float)

    for idx, value in enumerate(activity_arr):
        err_val = 0.0 if err_arr is None else float(err_arr[idx])
        try:
            _, _, total_bq, sigma_total = radon_activity.compute_total_radon(
                float(value),
                float(err_val),
                float(monitor_volume),
                float(sample_volume),
                allow_negative_activity=True,
            )
        except Exception:
            total_bq = float(value)
            sigma_total = float(err_val)

        total[idx] = total_bq
        if total_err is not None:
            total_err[idx] = sigma_total

    return total, total_err


def _as_timestamp(value: Any) -> float:
    """Return ``value`` as a UTC timestamp in seconds."""

    if isinstance(value, (int, float)):
        return float(value)
    if isinstance(value, np.generic):  # NumPy scalar
        return float(value)
    return to_utc_datetime(value).timestamp()


def _radon_time_window(
    start, end, radon_interval: Sequence[Any] | None
) -> tuple[float, float]:
    """Determine the plotting window for radon time-series outputs."""

    start_ts = _as_timestamp(start)
    end_ts = _as_timestamp(end)

    if radon_interval and len(radon_interval) == 2:
        try:
            interval_start = max(start_ts, _as_timestamp(radon_interval[0]))
            interval_end = min(end_ts, _as_timestamp(radon_interval[1]))
        except Exception:
            interval_start = interval_end = float("nan")
        else:
            if math.isfinite(interval_start) and math.isfinite(interval_end):
                if interval_end > interval_start:
                    return interval_start, interval_end
                if interval_end == interval_start:
                    return interval_start, interval_end

    return start_ts, end_ts


def _regrid_series(
    source_times: np.ndarray,
    source_values: np.ndarray | None,
    target_times: np.ndarray,
    fill_value: float,
) -> np.ndarray:
    """Project ``source_values`` sampled at ``source_times`` onto ``target_times``."""

    if source_values is None or source_values.size == 0 or source_times.size == 0:
        return np.full_like(target_times, float(fill_value), dtype=float)

    if source_values.size != source_times.size:
        return np.full_like(target_times, float(fill_value), dtype=float)

    times = np.asarray(source_times, dtype=float)
    values = np.asarray(source_values, dtype=float)
    mask = np.isfinite(times) & np.isfinite(values)
    if not np.any(mask):
        return np.full_like(target_times, float(fill_value), dtype=float)

    times = times[mask]
    values = values[mask]
    if times.size == 1:
        return np.full_like(target_times, float(values[0]), dtype=float)

    order = np.argsort(times, kind="mergesort")
    times = times[order]
    values = values[order]
    first = float(values[0])
    last = float(values[-1])
    return np.interp(target_times, times, values, left=first, right=last)


def _fit_params(obj: FitResult | Mapping[str, float] | None) -> FitParams:
    """Return fit parameters mapping from a ``FitResult`` or dictionary."""
    if isinstance(obj, FitResult):
        return cast(FitParams, obj.params)
    if isinstance(obj, Mapping):
        return obj  # type: ignore[return-value]
    return {}


def _config_efficiency(cfg: Mapping[str, Any], iso: str) -> float:
    """Return the prior efficiency for ``iso`` from ``cfg``."""

    eff_cfg = cfg.get("time_fit", {}).get(f"eff_{iso.lower()}")
    if isinstance(eff_cfg, (list, tuple)):
        return float(eff_cfg[0]) if eff_cfg else 1.0
    if eff_cfg is None or eff_cfg == "null":
        return 1.0
    try:
        return float(eff_cfg)
    except (TypeError, ValueError):
        return 1.0


def _fit_efficiency(params: Mapping[str, Any] | None, iso: str) -> float | None:
    """Return fitted efficiency for ``iso`` if present in ``params``."""

    if not params:
        return None

    keys = ("eff", f"eff_{iso}", f"eff_{iso.lower()}")
    for key in keys:
        val = params.get(key)
        if val is None:
            continue
        try:
            return float(val)
        except (TypeError, ValueError):
            continue
    return None


def _resolved_efficiency(
    cfg: Mapping[str, Any], iso: str, params: Mapping[str, Any] | None
) -> float:
    """Return efficiency for ``iso`` preferring fitted values over priors."""

    fitted = _fit_efficiency(params, iso)
    if fitted is not None and fitted > 0:
        return fitted
    return _config_efficiency(cfg, iso)


def _safe_float(value: Any) -> float | None:
    """Return ``value`` coerced to ``float`` when it is finite."""

    try:
        if value is None:
            return None
        coerced = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(coerced):
        return None
    return coerced


def _float_with_default(value: Any, default: float) -> float:
    """Return ``value`` as ``float`` or ``default`` when coercion fails."""

    coerced = _safe_float(value)
    return default if coerced is None else coerced


def _radon_activity_curve_from_fit(
    iso: str,
    fit_result: FitResult | Mapping[str, Any] | None,
    fit_params: Mapping[str, Any],
    t_rel: Sequence[float] | np.ndarray,
    cfg: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    """Return radon activity curve using sanitized fit parameters."""

    raw_E = fit_params.get("E_corrected")
    if _safe_float(raw_E) is None:
        raw_E = fit_params.get(f"E_{iso}")
    E = _float_with_default(raw_E, 0.0)

    raw_dE = fit_params.get("dE_corrected")
    if _safe_float(raw_dE) is None:
        raw_dE = fit_params.get(f"dE_{iso}", 0.0)
    dE = _float_with_default(raw_dE, 0.0)
    N0 = _float_with_default(fit_params.get(f"N0_{iso}", 0.0), 0.0)
    dN0 = _float_with_default(fit_params.get(f"dN0_{iso}", 0.0), 0.0)
    hl = _hl_value(cfg, iso)
    cov = _cov_lookup(fit_result, f"E_{iso}", f"N0_{iso}")
    return radon_activity.radon_activity_curve(t_rel, E, dE, N0, dN0, hl, cov)


def _cov_lookup(
    fit_result: FitResult | Mapping[str, float] | None, name1: str, name2: str
) -> float:
    """Return covariance between two parameters if present."""
    if isinstance(fit_result, FitResult):
        try:
            return float(fit_result.cov_df.loc[name1, name2])
        except KeyError:
            try:
                return float(fit_result.get_cov(name1, name2))
            except KeyError:
                return 0.0
    if isinstance(fit_result, Mapping):
        return float(fit_result.get(f"cov_{name1}_{name2}", 0.0))
    return 0.0


def _fallback_uncertainty(
    rate: float | None, fit_result: FitResult | Mapping[str, float] | None, param: str
) -> float:
    """Return uncertainty from covariance or a Poisson estimate."""

    def _try_var(value: Any) -> float | None:
        try:
            var_val = float(value)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(var_val) or var_val <= 0:
            return None
        return var_val

    candidates: list[Any] = []
    if isinstance(fit_result, FitResult):
        if fit_result.cov is not None and fit_result.param_index is not None:
            idx = fit_result.param_index.get(param)
            if idx is not None and idx < fit_result.cov.shape[0]:
                candidates.append(fit_result.cov[idx, idx])
        candidates.append(fit_result.params.get(f"cov_{param}_{param}"))
    elif isinstance(fit_result, Mapping):
        candidates.append(fit_result.get(f"cov_{param}_{param}"))

    for cand in candidates:
        var = _try_var(cand)
        if var is not None:
            return math.sqrt(var)

    try:
        rate_val = float(rate) if rate is not None else None
    except (TypeError, ValueError):
        rate_val = None

    if rate_val is None or not math.isfinite(rate_val):
        return 0.0

    return math.sqrt(abs(rate_val))


def _ensure_events(events: pd.DataFrame, stage: str) -> None:
    """Exit if ``events`` is empty, printing a helpful message."""
    if len(events) == 0:
        logger.error("No events remaining after %s. Exiting.", stage)
        raise SystemExit(1)


def _centroid_deviation(
    params: Mapping[str, float], known: Mapping[str, float]
) -> dict[str, float]:
    """Return |mu_fit - E_known| for each isotope present in ``params``."""
    dev: dict[str, float] = {}
    for iso, e_known in known.items():
        key = f"mu_{iso}"
        if key in params:
            dev[iso] = abs(float(params[key]) - float(e_known))
    return dev


def _normalise_mu_bounds(
    bounds_cfg: Mapping[str, Sequence[float] | None] | None,
    *,
    units: str,
    slope: float,
    intercept: float,
    quadratic_coeff: float,
) -> dict[str, tuple[float, float]]:
    """Return spectral centroid bounds expressed in MeV.

    ``bounds_cfg`` maps isotope names to lower/upper limits.  The
    ``units`` flag specifies whether those values are already in MeV or
    given in raw ADC channels.  When ADC bounds are provided they are
    converted to MeV using the supplied calibration coefficients so that
    downstream spectral fits, which operate in MeV, use consistent
    limits.
    """

    if not bounds_cfg:
        return {}

    units_norm = str(units).lower()
    if units_norm not in {"mev", "adc"}:
        raise ValueError("mu_bounds_units must be either 'mev' or 'adc'")

    normalised: dict[str, tuple[float, float]] = {}
    for iso, bounds in bounds_cfg.items():
        if bounds is None:
            continue
        if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence):
            raise ValueError(f"mu_bounds for {iso} must be a sequence of two numbers")
        if len(bounds) != 2:
            raise ValueError(f"mu_bounds for {iso} must contain exactly two elements")
        try:
            lo_raw = float(bounds[0])
            hi_raw = float(bounds[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"mu_bounds for {iso} must be numeric; got {bounds}"
            ) from exc
        if not lo_raw < hi_raw:
            raise ValueError(f"mu_bounds for {iso} require lower < upper")

        if units_norm == "adc":
            energies = apply_calibration(
                np.asarray([lo_raw, hi_raw], dtype=float),
                slope,
                intercept,
                quadratic_coeff=quadratic_coeff,
            )
            lo_val = float(np.min(energies))
            hi_val = float(np.max(energies))
        else:
            lo_val = float(lo_raw)
            hi_val = float(hi_raw)

        normalised[iso] = (lo_val, hi_val)

    return normalised


def _spectral_fit_with_check(
    energies: np.ndarray,
    priors: Mapping[str, tuple[float, float]],
    flags: Mapping[str, bool],
    cfg: Mapping[str, Any],
    *,
    bins: int | None = None,
    bin_edges: np.ndarray | None = None,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    unbinned: bool = False,
    strict: bool = False,
) -> tuple[FitResult | dict[str, float], dict[str, float]]:
    """Run :func:`fit_spectrum` and apply centroid consistency checks."""

    priors_mapped = dict(priors)
    if "sigma_E" in priors_mapped:
        mean, sig = priors_mapped.pop("sigma_E")
        priors_mapped.setdefault("sigma0", (mean, sig))
        priors_mapped.setdefault("F", (0.0, sig))

    # If F is fixed but no explicit prior is supplied, keep it near zero to
    # avoid unphysical broadening from a large default.
    if flags.get("fix_F", False) and "F" not in priors:
        priors_mapped["F"] = (0.0, 0.01)

    fit_kwargs = {
        "energies": energies,
        "priors": priors_mapped,
        "flags": flags,
    }
    max_tau_ratio = cfg.get("spectral_fit", {}).get("max_tau_ratio")
    if max_tau_ratio is not None:
        fit_kwargs["max_tau_ratio"] = max_tau_ratio
    if bins is not None or bin_edges is not None:
        fit_kwargs.update({"bins": bins, "bin_edges": bin_edges})
    if bounds:
        fit_kwargs["bounds"] = bounds
    if unbinned:
        fit_kwargs["unbinned"] = True
    if strict:
        fit_kwargs["strict"] = True

    result = fit_spectrum(**fit_kwargs)
    params = result.params if isinstance(result, FitResult) else result
    known = cfg.get("calibration", {}).get("known_energies", DEFAULT_KNOWN_ENERGIES)
    if isinstance(result, FitResult) and "sigma0" in params and "F" in params:
        e_ref = float(known.get("Po214", 0.0))
        sigma0 = float(params["sigma0"])
        F_val = float(params["F"])
        sigma_E_val = math.sqrt(max(sigma0**2 + F_val * e_ref, 0.0))
        result.params["sigma_E"] = sigma_E_val
        if result.cov is not None and sigma_E_val > 0.0:
            param_index = getattr(result, "param_index", None) or {}
            has_sigma0 = "sigma0" in param_index
            has_F = "F" in param_index

            var = 0.0
            if has_sigma0:
                var += (sigma0 / sigma_E_val) ** 2 * result.get_cov("sigma0", "sigma0")
            if has_F:
                var += (0.5 * e_ref / sigma_E_val) ** 2 * result.get_cov("F", "F")
            if has_sigma0 and has_F:
                var += (
                    2
                    * (sigma0 / sigma_E_val)
                    * (0.5 * e_ref / sigma_E_val)
                    * result.get_cov("sigma0", "F")
                )

            if has_sigma0 or has_F:
                result.params["dsigma_E"] = float(np.sqrt(max(var, 0.0)))
    tol = cfg.get("spectral_fit", {}).get("spectral_peak_tolerance_mev", 0.2)
    dev = _centroid_deviation(params, known)

    for iso, dval in dev.items():
        if dval > tol:
            logging.warning(
                "%s centroid deviates by %.3f MeV from calibration",
                iso,
                dval,
            )

    if any(d > 0.5 * tol for d in dev.values()):
        new_bounds = dict(bounds or {})
        for iso, dval in dev.items():
            if dval > 0.5 * tol:
                e_known = known[iso]
                new_bounds[f"mu_{iso}"] = (e_known - 0.5 * tol, e_known + 0.5 * tol)
        fit_kwargs["bounds"] = new_bounds
        refit = fit_spectrum(**fit_kwargs)
        ref_params = refit.params if isinstance(refit, FitResult) else refit
        ref_dev = _centroid_deviation(ref_params, known)
        if max(ref_dev.values(), default=0.0) < max(dev.values(), default=0.0):
            result, dev = refit, ref_dev

    return result, dev


def window_prob(E, sigma, lo, hi):
    """Return probability that each ``E`` lies in [lo, hi].

    Elements with ``sigma == 0`` are evaluated via a simple range check instead
    of calling :func:`scipy.stats.norm.cdf` with ``scale=0``.
    Parameters may be scalar or array-like and are broadcast element-wise.
    """

    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    E, sigma = np.broadcast_arrays(E, sigma)

    if np.any(sigma < 0):
        raise ValueError("negative sigma in window_prob")
    lo_val = float(lo)
    hi_val = float(hi)

    prob = np.empty_like(E, dtype=float)
    zero_mask = sigma == 0

    if np.any(zero_mask):
        prob[zero_mask] = ((E[zero_mask] >= lo_val) & (E[zero_mask] <= hi_val)).astype(
            float
        )

    if np.any(~zero_mask):
        nz = ~zero_mask
        prob[nz] = norm.cdf(hi_val, loc=E[nz], scale=sigma[nz]) - norm.cdf(
            lo_val, loc=E[nz], scale=sigma[nz]
        )

    if prob.ndim == 0:
        return float(prob)
    return prob


def auto_expand_window(df, window, threshold, step=0.05, max_width=1.0):
    """Return events within an expanded energy window.

    The window is symmetrically expanded in ``step`` increments until the
    number of selected events meets ``threshold`` or the width reaches
    ``max_width``.
    """

    lo, hi = map(float, window)
    energies = df["energy_MeV"].values
    sigma = df["denergy_MeV"].values

    while True:
        probs = window_prob(energies, sigma, lo, hi)
        count = np.sum(probs > 0)
        if count >= threshold or (hi - lo) >= max_width:
            mask = probs > 0
            out = df[mask].copy()
            out["weight"] = probs[mask]
            return out, (lo, hi)
        lo -= float(step)
        hi += float(step)


_spike_eff_cache = {}


def get_spike_efficiency(spike_cfg):
    """Return spike efficiency using :func:`calc_spike_efficiency` with caching."""

    counts = spike_cfg.get("counts")
    activity = spike_cfg.get("activity_bq")
    live_time = spike_cfg.get("live_time_s")

    key = (counts, activity, live_time)
    if key not in _spike_eff_cache:
        from efficiency import calc_spike_efficiency

        _spike_eff_cache[key] = calc_spike_efficiency(key[0], key[1], key[2])
    return _spike_eff_cache[key]


def prepare_analysis_df(
    df: pd.DataFrame,
    spike_start: pd.Timestamp | None,
    spike_end: pd.Timestamp | None,
    spike_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    run_periods: list[tuple[pd.Timestamp, pd.Timestamp]],
    analysis_end: pd.Timestamp | int | float | None,
    *,
    t0_global: datetime,
    cfg: dict,
    args,
) -> tuple[
    pd.DataFrame,
    datetime,
    datetime,
    float,
    float,
    str | None,
    Any,
]:
    """Apply time window cuts and derive drift parameters."""

    df_analysis = df.copy()
    ts = df_analysis["timestamp"]
    if not pd.api.types.is_datetime64_any_dtype(ts):
        df_analysis["timestamp"] = ts.map(parse_timestamp)
    else:
        if ts.dt.tz is None:
            df_analysis["timestamp"] = ts.map(parse_timestamp)
        else:
            df_analysis["timestamp"] = tz_convert_utc(ts)

    if spike_start is not None and spike_end is not None:
        mask = (df_analysis["timestamp"] >= spike_start) & (
            df_analysis["timestamp"] < spike_end
        )
        if mask.any():
            df_analysis = df_analysis[~mask].reset_index(drop=True)
    elif spike_start is not None:
        df_analysis = df_analysis[df_analysis["timestamp"] <= spike_start].reset_index(
            drop=True
        )
    elif spike_end is not None:
        df_analysis = df_analysis[df_analysis["timestamp"] >= spike_end].reset_index(
            drop=True
        )

    for start_ts, end_ts in spike_periods:
        mask = (df_analysis["timestamp"] >= start_ts) & (
            df_analysis["timestamp"] < end_ts
        )
        if mask.any():
            df_analysis = df_analysis[~mask].reset_index(drop=True)

    if run_periods:
        keep_mask = np.zeros(len(df_analysis), dtype=bool)
        for start_ts, end_ts in run_periods:
            keep_mask |= (df_analysis["timestamp"] >= start_ts) & (
                df_analysis["timestamp"] < end_ts
            )
        df_analysis = df_analysis[keep_mask].reset_index(drop=True)
        if analysis_end is None and len(df_analysis) > 0:
            analysis_end = df_analysis["timestamp"].max()

    if analysis_end is not None:
        df_analysis = df_analysis[df_analysis["timestamp"] <= analysis_end].reset_index(
            drop=True
        )
    else:
        analysis_end = df_analysis["timestamp"].max()

    if not isinstance(analysis_end, (int, float)):
        t_end_global_ts = to_utc_datetime(analysis_end).timestamp()
    else:
        t_end_global_ts = float(analysis_end)
    analysis_end_dt = datetime.fromtimestamp(t_end_global_ts, tz=timezone.utc)

    _ensure_events(df_analysis, "time-window selection")

    analysis_start = to_utc_datetime(t0_global)

    drift_cfg = cfg.get("systematics", {})
    drift_rate = (
        float(args.slope)
        if args.slope is not None
        else float(drift_cfg.get("adc_drift_rate", 0.0))
    )
    drift_mode = (
        "linear"
        if args.slope is not None
        else drift_cfg.get("adc_drift_mode", "linear")
    )
    drift_params = drift_cfg.get("adc_drift_params")

    return (
        df_analysis,
        analysis_start,
        analysis_end_dt,
        t_end_global_ts,
        drift_rate,
        drift_mode,
        drift_params,
    )


def _ts_bin_centers_widths(times, cfg, t_start, t_end):
    """Return bin centers and widths matching :func:`plot_time_series`."""
    arr = np.asarray(times)
    if np.issubdtype(arr.dtype, "datetime64"):
        arr = arr.astype("int64") / 1e9
    elif np.issubdtype(arr.dtype, np.object_):
        if arr.size > 0 and isinstance(arr.flat[0], datetime):
            arr = np.array([dt.timestamp() for dt in arr], dtype=float)
        else:
            arr = arr.astype(float)
    else:
        arr = arr.astype(float)

    if isinstance(t_start, datetime):
        t_start = t_start.timestamp()
    elif isinstance(t_start, np.datetime64):
        t_start = float(t_start.astype("int64") / 1e9)
    if isinstance(t_end, datetime):
        t_end = t_end.timestamp()
    elif isinstance(t_end, np.datetime64):
        t_end = float(t_end.astype("int64") / 1e9)

    bin_mode = str(
        cfg.get("plot_time_binning_mode", cfg.get("time_bin_mode", "fixed"))
    ).lower()
    bin_width_s = float(cfg.get("plot_time_bin_width_s", cfg.get("time_bin_s", 3600.0)))
    time_bins_fallback = int(cfg.get("time_bins_fallback", 1))

    periods = _resolve_run_periods(cfg, t_start, t_end)
    segments = _build_time_segments(
        arr,
        periods=periods,
        bin_mode=bin_mode,
        bin_width_s=bin_width_s,
        time_bins_fallback=time_bins_fallback,
        t_start=t_start,
    )
    if not segments:
        segments = _build_time_segments(
            arr,
            periods=[(float(t_start), float(t_end))],
            bin_mode=bin_mode,
            bin_width_s=bin_width_s,
            time_bins_fallback=time_bins_fallback,
            t_start=t_start,
        )

    centers_lists = [
        seg["centers_rel_global"] for seg in segments if seg["centers_rel_global"].size
    ]
    width_lists = [seg["bin_widths"] for seg in segments if seg["bin_widths"].size]

    centers = (
        np.concatenate(centers_lists) if centers_lists else np.array([], dtype=float)
    )
    widths = np.concatenate(width_lists) if width_lists else np.array([], dtype=float)
    return centers, widths


def _segments_to_isotope_series(ts_metadata):
    """Convert plot_time_series metadata to per-isotope count entries."""

    if not isinstance(ts_metadata, Mapping):
        return {}

    segments = ts_metadata.get("segments") or []
    iso_map: dict[str, list[dict[str, float]]] = {}
    for seg_idx, seg in enumerate(segments):
        counts_map = seg.get("counts") or {}
        centers = np.asarray(seg.get("centers_abs", []), dtype=float)
        widths = np.asarray(seg.get("bin_widths", []), dtype=float)
        for iso, counts in counts_map.items():
            counts_arr = np.asarray(counts, dtype=float)
            n = min(counts_arr.size, centers.size, widths.size)
            if n == 0:
                continue
            entries = iso_map.setdefault(iso, [])
            for idx in range(n):
                t_val = float(centers[idx]) if np.isfinite(centers[idx]) else None
                dt_val = float(widths[idx]) if np.isfinite(widths[idx]) else None
                if t_val is None or dt_val is None or dt_val <= 0:
                    continue
                entries.append(
                    {
                        "t": t_val,
                        "counts": float(counts_arr[idx]),
                        "dt": dt_val,
                        "segment_index": seg_idx,
                        "bin_index": idx,
                    }
                )

    for entries in iso_map.values():
        entries.sort(key=lambda row: row["t"])

    return iso_map


def dedupe_isotope_series(isotope_series_data, tol_seconds=0.5):
    """
    Remove duplicate time bins from isotope series data.

    Input:
        isotope_series_data: {"Po214": [{"t": ...,"counts": ...,"dt": ...}, ...], ...}
        tol_seconds: tolerance for considering timestamps equal (default 0.5 seconds)

    Output:
        Same shape as input, but with duplicate time bins removed.
        Duplicates are defined as entries with the same isotope where:
        - |t1 - t2| < tol_seconds
        - counts are equal
        - dt are equal

    The first occurrence of each unique entry is kept.
    """
    if not isinstance(isotope_series_data, dict):
        return isotope_series_data

    deduplicated = {}

    for isotope, entries in isotope_series_data.items():
        if not entries:
            deduplicated[isotope] = []
            continue

        # Sort by timestamp to ensure stable deduplication
        sorted_entries = sorted(entries, key=lambda row: row.get("t", 0.0))

        unique_entries = []
        for entry in sorted_entries:
            t_val = entry.get("t")
            counts_val = entry.get("counts")
            dt_val = entry.get("dt")

            if t_val is None or counts_val is None or dt_val is None:
                # Keep entries with missing values
                unique_entries.append(entry)
                continue

            # Check if this entry is a duplicate of any already-added entry
            is_duplicate = False
            for existing in unique_entries:
                existing_t = existing.get("t")
                existing_counts = existing.get("counts")
                existing_dt = existing.get("dt")

                if existing_t is None or existing_counts is None or existing_dt is None:
                    continue

                # Check if timestamps are within tolerance and counts/dt match
                if (
                    abs(existing_t - t_val) < tol_seconds
                    and existing_counts == counts_val
                    and existing_dt == dt_val
                ):
                    is_duplicate = True
                    break

            if not is_duplicate:
                unique_entries.append(entry)

        deduplicated[isotope] = unique_entries

    return deduplicated


def _model_uncertainty(centers, widths, fit_obj, iso, cfg, normalise):
    """Propagate fit parameter errors to the model curve."""
    if fit_obj is None:
        return None
    params = _fit_params(fit_obj)
    hl = _hl_value(cfg, iso)
    eff_cfg = cfg.get("time_fit", {}).get(f"eff_{iso.lower()}")
    if isinstance(eff_cfg, list):
        eff = eff_cfg[0]
    else:
        eff = eff_cfg if eff_cfg is not None else 1.0
    lam = math.log(2.0) / float(hl)
    dE = params.get("dE_corrected", params.get(f"dE_{iso}", 0.0))
    dN0 = params.get(f"dN0_{iso}", 0.0)
    dB = params.get(f"dB_{iso}", params.get("dB", 0.0))
    cov = _cov_lookup(fit_obj, f"E_{iso}", f"N0_{iso}")
    t = np.asarray(centers, dtype=float)
    exp_term = np.exp(-lam * t)
    dR_dE = eff * (1.0 - exp_term)
    dR_dN0 = eff * lam * exp_term
    dR_dB = 1.0
    var = (dR_dE * dE) ** 2 + (dR_dN0 * dN0) ** 2 + (dR_dB * dB) ** 2
    if cov and np.isfinite(cov):
        var += 2.0 * dR_dE * dR_dN0 * cov
    sigma_rate = np.sqrt(var)
    return sigma_rate if normalise else sigma_rate * widths
