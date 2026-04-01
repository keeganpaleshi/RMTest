#!/usr/bin/env python3
"""
analyze.py

Full Radon Monitor Analysis Pipeline
====================================

Usage:
    python analyze.py \\
        --config   config.yaml \\
        [--input   merged_output.csv] \\
        --output-dir  results \\
        [--baseline-range ISO_START ISO_END]

Pipeline steps:

  1. Load configuration (YAML or JSON).
  2. Load merged CSV of event data (timestamps, ADC channels).
  3. Energy calibration (two-point or auto) - assigns energy_MeV to each event.
  4. (Optional) Baseline interval extraction for background estimation.
  5. Spectral fit (Po-210, Po-218, Po-214, Po-216, Po-212) via binned
     chi-squared minimization with Gaussian + EMG tail + shelf + halo
     peak shapes. Supports ADC-width binning with optional DNL correction
     (Fourier parameterized or bandpass self-estimated). The two-stage
     full-resolution pipeline estimates Fourier DNL at bin_width=1, then
     rebins before the final fit. Per-period crossvalidation selects
     which Fourier harmonics improve held-out NLL.
  6. Time-series analysis per isotope energy window - binned count rates
     with Poisson + calibration-uncertainty error bars, optional decay
     model overlays.
  7. (Optional) Systematics scan around user-specified sigma shifts.
  8. JSON summary output (calibration, spectral fit, time-series,
     pull diagnostics, split-half validation, DNL crossval results).
  9. Plots saved under output-dir/<timestamp>/.

Example:

    python analyze.py \\
       --config    config.yaml \\
       --output-dir  results
"""


import argparse
import sys
import logging
import random
import time
import warnings

logger = logging.getLogger(__name__)
from datetime import datetime, timezone, timedelta
import subprocess
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, Sequence, cast
from contextlib import contextmanager

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from dateutil.tz import UTC, gettz
import radon_activity

from radon.external_rn_loader import load_external_rn_series
from radon.radon_inference import run_radon_inference
from radon.radon_plots import (
    plot_ambient_rn_vs_time,
    plot_rn_inferred_vs_time,
    plot_volume_equiv_vs_time,
)

from hierarchical import fit_hierarchical_runs

# ‣ Import our supporting modules (all must live in the same folder).
from io_utils import (
    load_config,
    copy_config,
    load_events,
    write_summary,
    apply_burst_filter,
    Summary,
)
from utils import to_native
from calibration import (
    derive_calibration_constants,
    derive_calibration_constants_auto,
    apply_calibration,
)

from fitting import fit_spectrum, fit_time_series, FitResult, FitParams
from reporting import build_diagnostics, start_warning_capture

from constants import (
    DEFAULT_NOISE_CUTOFF,
    NEGATIVE_ACTIVITY_CLAMP_UNCERTAINTY_BQ,
    PO210,
    PO212,
    PO214,
    PO216,
    PO218,
    RN222,
    DEFAULT_ADC_CENTROIDS,
    DEFAULT_KNOWN_ENERGIES,
)

NUCLIDES = {
    "Po210": PO210,
    "Po212": PO212,
    "Po214": PO214,
    "Po216": PO216,
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
            self.logger.info("%s took %.2f s", name, duration)

    def report(self):
        if not self._sections:
            return
        total = time.perf_counter() - self._start
        lines = [f"Pipeline timing summary (total {total:.2f} s):"]
        lines.extend(f"  - {name}: {duration:.2f} s" for name, duration in self._sections)
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
        if iso in consts:
            val = consts[iso].half_life_s
        elif iso in NUCLIDES:
            val = NUCLIDES[iso].half_life_s
        else:
            raise ValueError(f"Unknown isotope '{iso}' - not in config or NUCLIDES dictionary")
    return float(val)


def _summary_radon_background_mode(
    summary: Mapping[str, Any],
    cfg: Mapping[str, Any],
    time_fit_results: Mapping[str, Any],
) -> str | None:
    """Return the best available radon background mode for plotting."""

    time_fit_summary = summary.get("time_fit") if isinstance(summary, Mapping) else None
    if isinstance(time_fit_summary, Mapping):
        for iso in ("Po214", "Po218"):
            fit_summary = time_fit_summary.get(iso)
            if not isinstance(fit_summary, Mapping):
                continue
            mode = baseline_handling.normalize_background_mode(
                fit_summary.get("background_mode")
                or fit_summary.get("background_source")
            )
            if mode is not None:
                return mode

    return baseline_handling.normalize_background_mode(
        _radon_background_mode(cfg, time_fit_results)
    )


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

    try:
        import matplotlib.pyplot as plt
    except Exception as exc:  # pragma: no cover - matplotlib import guard
        raise RuntimeError("matplotlib is required to save spectrum plots") from exc

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


from plot_utils import (
    plot_spectrum,
    plot_spectrum_dnl_corrected,
    plot_time_series,
    plot_equivalent_air,
    plot_radon_activity_full,
    plot_total_radon_full,
    plot_radon_trend_full,
    plot_spectrum_comparison,
    plot_activity_grid,
    _resolve_run_periods,
    _build_time_segments,
)

from plot_utils.radon import (
    plot_radon_activity as _plot_radon_activity,
    plot_radon_trend as _plot_radon_trend,
)


def plot_radon_activity_dict(ts_dict, outdir, maybe_outdir=None, *_, **__):
    """Compatibility wrapper for tests expecting three arguments with dict input."""
    target = maybe_outdir or outdir
    Path(target).mkdir(parents=True, exist_ok=True)
    return _plot_radon_activity(ts_dict, target)


def plot_radon_trend_dict(ts_dict, outdir, maybe_outdir=None, *_, **__):
    """Compatibility wrapper for tests expecting three arguments with dict input."""
    target = maybe_outdir or outdir
    Path(target).mkdir(parents=True, exist_ok=True)
    return _plot_radon_trend(ts_dict, target)


from systematics import scan_systematics, apply_linear_adc_shift
from visualize import cov_heatmap, efficiency_bar
from utils import (
    find_adc_bin_peaks,
    adc_hist_edges,
    rebin_histogram,
    fd_rebin_factor,
    parse_time_arg,
    to_utc_datetime,
)
from utils.time_utils import (
    parse_timestamp,
    to_epoch_seconds,
    to_datetime_utc,
    tz_convert_utc,
)
from baseline_utils import (
    subtract_baseline_counts,
    subtract_baseline_rate,
    compute_dilution_factor,
    summarize_baseline,
    BaselineError,
)
import baseline
import baseline_handling
from time_fitting import two_pass_time_fit
from config.validation import validate_baseline_window


def plot_radon_activity(
    times,
    activity,
    out_png,
    errors=None,
    *,
    config=None,
    sample_volume_l=None,
    background_mode=None,
):
    """Wrapper used by tests expecting output path as third argument."""

    return plot_radon_activity_full(
        times,
        activity,
        errors,
        out_png,
        config=config,
        sample_volume_l=sample_volume_l,
        background_mode=background_mode,
    )


def plot_total_radon(
    times,
    total_bq,
    out_png,
    errors=None,
    *,
    config=None,
    background_mode=None,
):
    """Wrapper used by tests expecting output path as third argument."""

    return plot_total_radon_full(
        times,
        total_bq,
        errors,
        out_png,
        config=config,
        background_mode=background_mode,
    )


def plot_radon_trend(
    times,
    activity,
    out_png,
    *,
    config=None,
    sample_volume_l=None,
    fit_valid=True,
):
    """Wrapper used by tests expecting output path as third argument."""
    return plot_radon_trend_full(
        times,
        activity,
        out_png,
        config=config,
        sample_volume_l=sample_volume_l,
        fit_valid=fit_valid,
    )


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

    return to_epoch_seconds(value)

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
    # Daughter-fit parameters are converted into an Rn-222 activity curve, so
    # evaluate them with the radon half-life instead of the daughter half-life.
    hl = _hl_value(cfg, "Rn222")
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


def _compute_cal_window_rel_unc(cal_result, cfg: Mapping[str, Any]) -> dict[str, float]:
    """Return relative efficiency uncertainty per isotope from calibration energy scale.

    For each isotope whose time-fit window is configured, computes the fractional
    systematic uncertainty on the count rate due to calibration energy scale uncertainty
    propagated through the Gaussian window efficiency.  When the calibration peak
    position shifts by δμ (from ``cal_result.uncertainty``), the fraction of events
    captured in the fixed window [E_lo, E_hi] changes by δf, giving a relative
    rate uncertainty δf/f.

    Returns
    -------
    dict
        Mapping of isotope name to relative uncertainty (dimensionless fraction).
    """
    try:
        from scipy.stats import norm as _norm
    except ImportError:
        return {}

    time_fit_cfg = cfg.get("time_fit", {})
    result: dict[str, float] = {}
    iso_map = {
        "Po218": "window_po218",
        "Po214": "window_po214",
        "Po210": "window_po210",
        "Po212": "window_po212",
    }
    peaks = getattr(cal_result, "peaks", None) or {}
    sigma_e_global = getattr(cal_result, "sigma_E", None)

    for iso, window_key in iso_map.items():
        window = time_fit_cfg.get(window_key)
        if window is None or len(window) != 2:
            continue
        try:
            e_lo, e_hi = float(window[0]), float(window[1])
        except (TypeError, ValueError):
            continue

        peak = peaks.get(iso, {})
        mu_mev = peak.get("centroid_mev")
        if mu_mev is None:
            mu_mev = (e_lo + e_hi) / 2.0

        # Peak width in MeV
        sigma_adc = peak.get("sigma_adc")
        try:
            a_coeff = float(cal_result.coeffs[1]) if len(cal_result.coeffs) > 1 else 1.0
        except (IndexError, TypeError, AttributeError):
            a_coeff = 1.0
        if sigma_adc is not None and a_coeff != 0:
            sigma_mev = abs(a_coeff) * float(sigma_adc)
        elif sigma_e_global is not None and float(sigma_e_global) > 0:
            sigma_mev = float(sigma_e_global)
        else:
            continue

        if sigma_mev <= 0:
            continue

        # Energy uncertainty at peak from calibration covariance
        centroid_adc = peak.get("centroid_adc")
        try:
            if centroid_adc is not None:
                delta_mu_mev = float(cal_result.uncertainty(float(centroid_adc)))
            else:
                c_coeff = float(cal_result.coeffs[0]) if cal_result.coeffs else 0.0
                adc_approx = (float(mu_mev) - c_coeff) / a_coeff if a_coeff != 0 else 1000.0
                delta_mu_mev = float(cal_result.uncertainty(adc_approx))
        except Exception:
            continue

        if not math.isfinite(delta_mu_mev) or delta_mu_mev <= 0:
            continue

        # Gaussian window efficiency: f = Phi(hi_z) - Phi(lo_z)
        lo_z = (e_lo - float(mu_mev)) / sigma_mev
        hi_z = (e_hi - float(mu_mev)) / sigma_mev
        f = _norm.cdf(hi_z) - _norm.cdf(lo_z)
        if f < 1e-10:
            continue

        # |df/dmu| = |phi(lo_z) - phi(hi_z)| / sigma_mev
        df_dmu = abs(_norm.pdf(lo_z) - _norm.pdf(hi_z)) / sigma_mev
        rel_unc = df_dmu * delta_mu_mev / f
        if math.isfinite(rel_unc) and rel_unc >= 0:
            result[iso] = float(rel_unc)

    return result


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
        sys.exit(1)


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
    cubic_coeff: float = 0.0,
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
        raise ValueError(
            "mu_bounds_units must be either 'mev' or 'adc'"
        )

    normalised: dict[str, tuple[float, float]] = {}
    for iso, bounds in bounds_cfg.items():
        if bounds is None:
            continue
        if isinstance(bounds, (str, bytes)) or not isinstance(bounds, Sequence):
            raise ValueError(
                f"mu_bounds for {iso} must be a sequence of two numbers"
            )
        if len(bounds) != 2:
            raise ValueError(
                f"mu_bounds for {iso} must contain exactly two elements"
            )
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
                cubic_coeff=cubic_coeff,
            )
            lo_val = float(np.min(energies))
            hi_val = float(np.max(energies))
        else:
            lo_val = float(lo_raw)
            hi_val = float(hi_raw)

        normalised[iso] = (lo_val, hi_val)

    return normalised


def _normalise_fit_energy_range(
    fit_range_cfg: Sequence[float] | None,
) -> tuple[float, float] | None:
    """Return a validated spectral fit window in MeV."""

    if fit_range_cfg is None:
        return None
    if isinstance(fit_range_cfg, (str, bytes)) or not isinstance(fit_range_cfg, Sequence):
        raise ValueError("fit_energy_range must be a sequence of two numbers")
    if len(fit_range_cfg) != 2:
        raise ValueError("fit_energy_range must contain exactly two elements")
    try:
        lo = float(fit_range_cfg[0])
        hi = float(fit_range_cfg[1])
    except (TypeError, ValueError) as exc:
        raise ValueError(
            f"fit_energy_range must be numeric; got {fit_range_cfg}"
        ) from exc
    if not np.isfinite(lo) or not np.isfinite(hi):
        raise ValueError("fit_energy_range values must be finite")
    if hi <= lo:
        raise ValueError("fit_energy_range requires lower < upper")
    return float(lo), float(hi)


def _select_spectral_fit_frame(
    df: pd.DataFrame,
    spectral_cfg: Mapping[str, Any],
) -> tuple[pd.DataFrame, tuple[float, float] | None]:
    """Return the event subset used for the spectral fit."""

    fit_range = _normalise_fit_energy_range(spectral_cfg.get("fit_energy_range"))
    if fit_range is None:
        return df, None

    lo, hi = fit_range
    energies = df["energy_MeV"].to_numpy(dtype=float, copy=False)
    mask = np.isfinite(energies) & (energies >= lo) & (energies <= hi)
    return df.loc[mask], fit_range


def _estimate_loglin_background_prior(
    energies: Sequence[float] | np.ndarray,
    peak_centroids: Mapping[str, float],
    *,
    peak_width: float,
    prior_hint: Sequence[float] | None = None,
) -> tuple[float, float]:
    """Return a broad total-count prior for the log-linear background."""

    energies_arr = np.asarray(energies, dtype=float)
    energies_arr = energies_arr[np.isfinite(energies_arr)]
    if energies_arr.size == 0:
        return 1.0, 1.0

    continuum_mask = np.ones(energies_arr.shape, dtype=bool)
    width = float(peak_width)
    if np.isfinite(width) and width > 0.0:
        for mu in peak_centroids.values():
            mu_val = float(mu)
            if np.isfinite(mu_val):
                continuum_mask &= np.abs(energies_arr - mu_val) > width

    continuum_counts = float(np.count_nonzero(continuum_mask))
    if continuum_counts <= 0.0:
        continuum_counts = float(energies_arr.size)

    prior_mean = float("nan")
    prior_sigma = float("nan")
    if prior_hint is not None:
        if isinstance(prior_hint, (str, bytes)) or not isinstance(prior_hint, Sequence):
            raise ValueError("S_bkg_prior must be a sequence of two numbers")
        if len(prior_hint) != 2:
            raise ValueError("S_bkg_prior must contain exactly two elements")
        try:
            prior_mean = float(prior_hint[0])
            prior_sigma = float(prior_hint[1])
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"S_bkg_prior must be numeric; got {prior_hint}"
            ) from exc

    mean = continuum_counts
    if np.isfinite(prior_mean) and prior_mean > 0.0:
        mean = max(mean, prior_mean)

    sigma_scale = abs(prior_sigma) if np.isfinite(prior_sigma) and prior_sigma > 0.0 else 1.0
    sigma = max(np.sqrt(mean), sigma_scale * mean, 1.0)
    return float(mean), float(sigma)


def _preprocess_full_resolution_dnl(
    adc_all: np.ndarray,
    energies: np.ndarray,
    cal_slope: float,
    cal_intercept: float,
    cal_a2: float,
    cal_a3: float,
    priors: Mapping[str, tuple[float, float]],
    flags: Mapping[str, object],
    cfg: Mapping[str, Any],
    *,
    bounds: Mapping[str, tuple[float | None, float | None]] | None = None,
    timestamps: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, dict, dict]:
    """Two-stage DNL: estimate Fourier DNL at bin_width=1, apply, then rebin.

    Returns
    -------
    rebinned_hist : np.ndarray
        DNL-corrected, rebinned histogram counts.
    rebinned_edges_mev : np.ndarray
        Bin edges in MeV for the rebinned histogram.
    dnl_meta : dict
        DNL metadata for the result.
    info : dict
        Additional info (rebin_factor, n_bins_full_res, etc.).
    """
    from fitting import fit_spectrum, FitResult

    dnl_cfg = cfg.get("spectral_fit", {}).get("dnl_correction", {})
    fourier_periods = dnl_cfg.get(
        "fourier_periods_codes", [4, 8, 16, 32, 64, 128, 256, 512]
    )

    # ── 1. Histogram at bin_width=1 (full ADC resolution) ──────────
    edges_adc_full = adc_hist_edges(adc_all, channel_width=1)
    hist_full, _ = np.histogram(adc_all, bins=edges_adc_full)
    edges_mev_full = apply_calibration(
        edges_adc_full, cal_slope, cal_intercept,
        quadratic_coeff=cal_a2, cubic_coeff=cal_a3,
    )
    n_full = hist_full.size
    logger.info(
        "Full-resolution DNL: %d bins at bin_width=1", n_full
    )

    # ── 2. Preliminary fit at full resolution (DNL disabled) ───────
    prelim_flags = dict(flags)
    prelim_priors = dict(priors)
    # Build a cfg copy with DNL disabled for the preliminary fit
    import copy
    prelim_cfg = copy.deepcopy(dict(cfg))
    prelim_dnl = prelim_cfg.get("spectral_fit", {}).get("dnl_correction", {})
    prelim_dnl["enabled"] = False
    prelim_dnl["full_resolution_estimate"] = False

    # If the main fit uses "none" background, the prelim fit still needs a
    # polynomial background to estimate DNL residuals properly.
    _prelim_bkg_model = str(prelim_flags.get("background_model", "")).lower()
    if _prelim_bkg_model == "none":
        prelim_flags["background_model"] = "loglin_unit"
        # Add default b0/b1 priors for the prelim fit
        if "b0" not in prelim_priors:
            prelim_priors["b0"] = (0.0, 4.0)
        if "b1" not in prelim_priors:
            prelim_priors["b1"] = (0.0, 4.0)
        logger.info(
            "Prelim fit: overriding 'none' background → loglin_unit for DNL estimation"
        )

    prelim_flags["cfg"] = prelim_cfg

    prelim_result = fit_spectrum(
        energies,
        prelim_priors,
        flags=prelim_flags,
        bin_edges=edges_mev_full,
        bounds=bounds,
        skip_minos=True,
    )
    if isinstance(prelim_result, FitResult):
        prelim_params = prelim_result.params
        logger.info("Preliminary fit returned FitResult with %d param keys", len(prelim_params))
    else:
        prelim_params = prelim_result
        logger.info("Preliminary fit returned dict with %d keys", len(prelim_params))

    # ── 3. Compute residuals and fit Fourier DNL ───────────────────
    # Reconstruct model prediction at full resolution
    model_total = prelim_params.get("_plot_model_total")
    logger.info(
        "Preliminary fit _plot_model_total: %s (type=%s)",
        "present" if model_total is not None else "MISSING",
        type(model_total).__name__ if model_total is not None else "None",
    )
    if model_total is None:
        # Log available keys for debugging
        _internal_keys = [k for k in prelim_params if k.startswith("_")]
        logger.warning(
            "Full-resolution preliminary fit did not return model_total; "
            "cannot estimate Fourier DNL. Internal keys: %s",
            _internal_keys,
        )
        # Fall back: return un-corrected histogram at requested binning
        rebin_spec = dnl_cfg.get("post_dnl_rebin", "fd")
        factor = _resolve_rebin(rebin_spec, adc_all, cal_slope, n_full)
        rebinned, rebinned_edges_adc = rebin_histogram(
            hist_full.astype(float), edges_adc_full, factor
        )
        rebinned_edges_mev = apply_calibration(
            rebinned_edges_adc, cal_slope, cal_intercept,
            quadratic_coeff=cal_a2, cubic_coeff=cal_a3,
        )
        return rebinned, rebinned_edges_mev, {}, {"rebin_factor": factor}

    model_arr = np.asarray(model_total, dtype=float)
    if model_arr.size != n_full:
        logger.warning(
            "Model size (%d) != histogram size (%d); skipping DNL",
            model_arr.size, n_full,
        )
        rebin_spec = dnl_cfg.get("post_dnl_rebin", "fd")
        factor = _resolve_rebin(rebin_spec, adc_all, cal_slope, n_full)
        rebinned, rebinned_edges_adc = rebin_histogram(
            hist_full.astype(float), edges_adc_full, factor
        )
        rebinned_edges_mev = apply_calibration(
            rebinned_edges_adc, cal_slope, cal_intercept,
            quadratic_coeff=cal_a2, cubic_coeff=cal_a3,
        )
        return rebinned, rebinned_edges_mev, {}, {"rebin_factor": factor}

    # Residuals: (data / model) - 1
    safe_model = np.where(model_arr > 0, model_arr, 1.0)
    residuals = (hist_full.astype(float) / safe_model) - 1.0
    valid_mask = model_arr > float(dnl_cfg.get("min_counts", 5))
    bin_indices = np.arange(n_full)

    # ── 3a. Per-period crossval: auto-select validated frequencies ────
    # Split events by time into two halves, histogram each, compute
    # per-period Fourier coefficients independently, keep only periods
    # where the two halves agree (real hardware DNL vs noise).
    periods_bin = [p / 1.0 for p in fourier_periods]  # bin_width=1
    resolvable = [
        (pc, pb) for pc, pb in zip(fourier_periods, periods_bin) if pb >= 2.0
    ]
    per_period_crossval = {}  # {period: {amp_A, amp_B, phase_A, phase_B, ...}}

    if resolvable and timestamps is not None and len(timestamps) >= 200:
        order = np.argsort(timestamps)
        mid = len(order) // 2
        idx_A, idx_B = order[:mid], order[mid:]
        adc_A, adc_B = adc_all[idx_A], adc_all[idx_B]

        hist_A, _ = np.histogram(adc_A, bins=edges_adc_full)
        hist_B, _ = np.histogram(adc_B, bins=edges_adc_full)

        # Model scales linearly with counts  - use half the full model
        model_half = safe_model * 0.5
        model_half_safe = np.where(model_half > 0, model_half, 1.0)
        resid_A = (hist_A.astype(float) / model_half_safe) - 1.0
        resid_B = (hist_B.astype(float) / model_half_safe) - 1.0

        # Require slightly more counts for half-data validity
        valid_half = model_half > max(float(dnl_cfg.get("min_counts", 5)) * 0.5, 2.0)
        valid_idx_half = np.flatnonzero(valid_half)

        if valid_idx_half.size > 2 * len(resolvable):
            # Fit each period individually on each half
            for pc, pb in resolvable:
                # 2-column design matrix for this single period
                phase_v = 2.0 * np.pi * bin_indices[valid_idx_half] / pb
                A_single = np.column_stack([np.cos(phase_v), np.sin(phase_v)])

                coeff_A, _, _, _ = np.linalg.lstsq(A_single, resid_A[valid_idx_half], rcond=None)
                coeff_B, _, _, _ = np.linalg.lstsq(A_single, resid_B[valid_idx_half], rcond=None)

                amp_A = np.hypot(coeff_A[0], coeff_A[1])
                amp_B = np.hypot(coeff_B[0], coeff_B[1])
                phase_A = np.arctan2(coeff_A[1], coeff_A[0])
                phase_B = np.arctan2(coeff_B[1], coeff_B[0])

                # Phase difference (wrapped to [-pi, pi])
                d_phase = (phase_A - phase_B + np.pi) % (2 * np.pi) - np.pi

                # Amplitude agreement: ratio closer to 1 = better
                amp_ratio = min(amp_A, amp_B) / max(amp_A, amp_B) if max(amp_A, amp_B) > 0 else 0.0

                # Noise floor estimate: for N valid bins and Poisson stats,
                # the expected amplitude of a noise-only Fourier component is
                # ~ 1/sqrt(N_valid) * sqrt(2/N_valid) for the residual.
                # Use the actual residual RMS as a scale.
                resid_rms_A = float(np.std(resid_A[valid_idx_half]))
                resid_rms_B = float(np.std(resid_B[valid_idx_half]))
                noise_amp = np.sqrt(2.0 / valid_idx_half.size) * 0.5 * (resid_rms_A + resid_rms_B)
                snr_A = amp_A / noise_amp if noise_amp > 0 else 0.0
                snr_B = amp_B / noise_amp if noise_amp > 0 else 0.0

                per_period_crossval[pc] = {
                    "amp_A": float(amp_A),
                    "amp_B": float(amp_B),
                    "phase_A": float(phase_A),
                    "phase_B": float(phase_B),
                    "d_phase": float(d_phase),
                    "amp_ratio": float(amp_ratio),
                    "snr_A": float(snr_A),
                    "snr_B": float(snr_B),
                    "noise_amp": float(noise_amp),
                }

            # Auto-select: keep periods where both halves have SNR > 2
            # AND amplitude ratio > 0.3 AND phase difference < 90°
            validated = []
            rejected = []
            for pc, pb in resolvable:
                m = per_period_crossval.get(pc, {})
                snr_ok = m.get("snr_A", 0) > 2.0 and m.get("snr_B", 0) > 2.0
                amp_ok = m.get("amp_ratio", 0) > 0.3
                phase_ok = abs(m.get("d_phase", np.pi)) < np.pi / 2
                if snr_ok and amp_ok and phase_ok:
                    validated.append((pc, pb))
                    logger.info(
                        "  Period %4d: PASS  amp=%.4f/%.4f  ratio=%.2f  "
                        "dφ=%+.1f°  SNR=%.1f/%.1f",
                        pc, m["amp_A"], m["amp_B"], m["amp_ratio"],
                        np.degrees(m["d_phase"]), m["snr_A"], m["snr_B"],
                    )
                else:
                    rejected.append(pc)
                    logger.info(
                        "  Period %4d: FAIL  amp=%.4f/%.4f  ratio=%.2f  "
                        "dφ=%+.1f°  SNR=%.1f/%.1f  [%s]",
                        pc, m.get("amp_A", 0), m.get("amp_B", 0),
                        m.get("amp_ratio", 0),
                        np.degrees(m.get("d_phase", 0)),
                        m.get("snr_A", 0), m.get("snr_B", 0),
                        ", ".join(
                            f for f, ok in [("snr", snr_ok), ("amp", amp_ok), ("phase", phase_ok)]
                            if not ok
                        ),
                    )

            logger.info(
                "Per-period crossval: %d/%d periods validated, %d rejected %s",
                len(validated), len(resolvable), len(rejected), rejected,
            )

            # Replace resolvable with validated periods only
            if validated:
                resolvable = validated
            else:
                logger.warning("No Fourier periods passed per-period crossval; using all")
        else:
            logger.info("Per-period crossval: too few valid half-bins (%d); skipping",
                        valid_idx_half.size)
    elif resolvable and timestamps is None:
        logger.info("Per-period crossval: no timestamps provided; skipping auto-selection")

    # ── 3b. Fit Fourier DNL using validated periods ──────────────────
    if not resolvable:
        logger.warning("No resolvable Fourier periods at bin_width=1")
        dnl_factors = np.ones(n_full, dtype=float)
        fourier_coeffs = {}
    else:
        valid_idx = np.flatnonzero(valid_mask)
        n_terms = len(resolvable)
        A = np.zeros((valid_idx.size, 2 * n_terms))
        for k, (pc, pb) in enumerate(resolvable):
            phase = 2.0 * np.pi * bin_indices[valid_idx] / pb
            A[:, 2 * k] = np.cos(phase)
            A[:, 2 * k + 1] = np.sin(phase)
        b = residuals[valid_idx]
        result_lstsq, _, _, _ = np.linalg.lstsq(A, b, rcond=None)

        dnl_model = np.zeros(n_full, dtype=float)
        fourier_coeffs = {}
        for k, (pc, pb) in enumerate(resolvable):
            a_k = float(result_lstsq[2 * k])
            b_k = float(result_lstsq[2 * k + 1])
            phase = 2.0 * np.pi * bin_indices / pb
            dnl_model += a_k * np.cos(phase) + b_k * np.sin(phase)
            fourier_coeffs[pc] = (a_k, b_k)

        dnl_factors = 1.0 + dnl_model
        logger.info(
            "Fourier DNL estimated: %d validated terms, amplitude RMS=%.4f",
            len(fourier_coeffs),
            float(np.sqrt(np.mean(dnl_model ** 2))),
        )

    # ── 4. Apply DNL correction to full-resolution counts ──────────
    # DNL factor > 1 means the channel is wider, collecting more counts.
    # Corrected counts = raw / dnl_factor (normalize to nominal width).
    corrected = hist_full.astype(float) / np.clip(dnl_factors, 0.5, 2.0)

    # ── 5. Rebin the corrected histogram ───────────────────────────
    rebin_spec = dnl_cfg.get("post_dnl_rebin", "fd")
    factor = _resolve_rebin(rebin_spec, adc_all, cal_slope, n_full)

    rebinned, rebinned_edges_adc = rebin_histogram(
        corrected, edges_adc_full, factor
    )
    rebinned_edges_mev = apply_calibration(
        rebinned_edges_adc, cal_slope, cal_intercept,
        quadratic_coeff=cal_a2, cubic_coeff=cal_a3,
    )
    logger.info(
        "Post-DNL rebin: factor=%d, %d -> %d bins",
        factor, n_full, rebinned.size,
    )

    # ── 6. Build metadata ──────────────────────────────────────────
    dnl_meta = {
        "dnl_applied": True,
        "dnl_iterations": 1,
        "operator_class": "full_resolution_fourier_rebin",
        "calibration_source": "self",
        "fourier_coefficients": {
            str(k): list(v) for k, v in fourier_coeffs.items()
        },
        "effective_dnl_params": 2 * len(fourier_coeffs),
        "full_resolution_bins": n_full,
        "rebin_factor": factor,
        "rebinned_bins": int(rebinned.size),
        "post_dnl_rebin_spec": str(rebin_spec),
        "dnl_factors_full_res": dnl_factors.tolist(),
        "dnl_amplitude_rms": float(
            np.sqrt(np.mean((dnl_factors - 1.0) ** 2))
        ),
        "statistical_model": "fourier_corrected_rebinned_poisson",
    }
    if per_period_crossval:
        dnl_meta["per_period_crossval"] = {
            str(k): v for k, v in per_period_crossval.items()
        }
        dnl_meta["validated_periods"] = [pc for pc, _ in resolvable]
        dnl_meta["auto_selected"] = True

    # Extract prelim fit background params for seeding the main fit
    _prelim_bkg_params = {}
    for _bk in ("b0", "b1", "b2", "b3", "S_bkg"):
        _bv = prelim_params.get(_bk)
        if _bv is not None and np.isfinite(float(_bv)):
            _prelim_bkg_params[_bk] = float(_bv)
    if _prelim_bkg_params:
        logger.info(
            "Prelim fit background params: %s",
            ", ".join(f"{k}={v:.4f}" for k, v in _prelim_bkg_params.items()),
        )

    # Save prelim fit plot data for per-stage diagnostic plotting
    _prelim_plot_data = None
    try:
        _prelim_plot_data = {
            "fit_vals": dict(prelim_params) if isinstance(prelim_params, dict) else dict(prelim_params.params) if hasattr(prelim_params, 'params') else {},
            "bins": n_full,
            "bin_edges": edges_mev_full,
            "flags": dict(prelim_flags),
        }
    except Exception as _pex:
        logger.debug("Could not assemble prelim plot data: %s", _pex)

    info = {
        "rebin_factor": factor,
        "n_bins_full_res": n_full,
        "fourier_coeffs": fourier_coeffs,
        "dnl_factors_full_res": dnl_factors,
        "prelim_chi2_ndf": prelim_params.get("chi2_ndf"),
        "per_period_crossval": per_period_crossval,
        # Full-res corrected histogram + ADC edges for 3-stage re-rebinning
        "corrected_full_res": corrected,
        "edges_adc_full": edges_adc_full,
        # Prelim fit background params for seeding main fit
        "prelim_bkg_params": _prelim_bkg_params,
        # Prelim fit plot data for per-stage diagnostics
        "prelim_plot_data": _prelim_plot_data,
    }

    return rebinned, rebinned_edges_mev, dnl_meta, info


def _resolve_rebin(
    rebin_spec: str | int | float,
    adc_all: np.ndarray,
    cal_slope: float,
    n_bins: int,
) -> int:
    """Resolve a rebin specification to an integer rebin factor.

    Parameters
    ----------
    rebin_spec : str or int or float
        ``"fd"`` for Freedman-Diaconis rule, an integer for a fixed rebin
        factor, or a float ending with ``"w"`` (e.g. ``"5w"``) to specify
        a target bin width in ADC channels.
    adc_all : np.ndarray
        Raw ADC values (for FD calculation).
    cal_slope : float
        Calibration slope MeV/channel.
    n_bins : int
        Number of full-resolution bins (for sanity clamp).
    """
    spec = str(rebin_spec).strip().lower()
    if spec == "fd":
        factor = fd_rebin_factor(adc_all, cal_slope, n_full_bins=n_bins)
    elif spec.endswith("w"):
        # Interpret as target bin width in ADC channels
        try:
            target_width = float(spec[:-1])
            factor = max(1, round(target_width))
        except ValueError:
            logger.warning("Invalid rebin spec '%s', using fd", rebin_spec)
            factor = fd_rebin_factor(adc_all, cal_slope, n_full_bins=n_bins)
    else:
        try:
            factor = max(1, int(float(spec)))
        except (ValueError, TypeError):
            logger.warning("Invalid rebin spec '%s', using fd", rebin_spec)
            factor = fd_rebin_factor(adc_all, cal_slope, n_full_bins=n_bins)
    # Sanity: don't rebin more than half the bins away
    factor = min(factor, max(1, n_bins // 4))
    return factor


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
    pre_binned_hist: np.ndarray | None = None,
    pre_dnl_meta: dict | None = None,
) -> tuple[FitResult | dict[str, float], dict[str, float]]:
    """Run :func:`fit_spectrum` and apply centroid consistency checks."""

    priors_mapped = dict(priors)
    if "sigma_E" in priors_mapped:
        mean, sig = priors_mapped.pop("sigma_E")
        priors_mapped.setdefault("sigma0", (mean, sig))
        priors_mapped.setdefault("F", (0.0, sig))

    fit_flags = dict(flags)
    fit_flags.setdefault("cfg", cfg)

    # If F is fixed but no explicit prior is supplied, keep it near zero to
    # avoid unphysical broadening from a large default.
    if fit_flags.get("fix_F", False) and "F" not in priors:
        priors_mapped["F"] = (0.0, 0.01)

    fit_kwargs = {
        "energies": energies,
        "priors": priors_mapped,
        "flags": fit_flags,
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
    if pre_binned_hist is not None:
        fit_kwargs["pre_binned_hist"] = pre_binned_hist
    if pre_dnl_meta is not None:
        fit_kwargs["pre_dnl_meta"] = pre_dnl_meta

    # skip_minos: use Hesse diagonal errors instead of profile likelihood
    _skip_minos = cfg.get("spectral_fit", {}).get("skip_minos", False)
    if _skip_minos:
        fit_kwargs["skip_minos"] = True

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
                f"{iso} centroid deviates by {dval:.3f} MeV from calibration"
            )

    _centroid_refit_triggered = False
    _centroid_refit_accepted = False
    _centroid_refit_isotopes = []
    if any(d > 0.5 * tol for d in dev.values()):
        _centroid_refit_isotopes = [iso for iso, d in dev.items() if d > 0.5 * tol]
        _centroid_refit_triggered = True
        logging.info(
            "Centroid refit triggered: isotopes=%s, deviations=%s",
            _centroid_refit_isotopes,
            {iso: f"{dev[iso]:.4f}" for iso in _centroid_refit_isotopes},
        )
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
            _centroid_refit_accepted = True
            logging.info("Centroid refit accepted (improved deviations)")
        else:
            logging.info("Centroid refit rejected (did not improve)")

    # Attach refit metadata to result
    _refit_meta = {
        "centroid_refit_triggered": _centroid_refit_triggered,
        "centroid_refit_accepted": _centroid_refit_accepted,
        "centroid_refit_isotopes": _centroid_refit_isotopes,
    }
    if isinstance(result, FitResult):
        result.params["_centroid_refit"] = _refit_meta
    elif isinstance(result, dict):
        result["_centroid_refit"] = _refit_meta

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
                if (abs(existing_t - t_val) < tol_seconds and
                    existing_counts == counts_val and
                    existing_dt == dt_val):
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


def _time_fit_weight_scale(
    counts: float,
    live_time_s: float,
    efficiency: float,
    target_sigma: float,
) -> float:
    """Return the uniform weight scaling needed to match ``target_sigma``."""

    try:
        counts_val = float(counts)
        live_time_val = float(live_time_s)
        eff_val = float(efficiency)
        sigma_val = float(target_sigma)
    except (TypeError, ValueError):
        return 1.0

    if (
        not math.isfinite(counts_val)
        or not math.isfinite(live_time_val)
        or not math.isfinite(eff_val)
        or not math.isfinite(sigma_val)
        or counts_val <= 0.0
        or live_time_val <= 0.0
        or eff_val <= 0.0
        or sigma_val <= 0.0
    ):
        return 1.0

    raw_sigma = math.sqrt(counts_val) / (live_time_val * eff_val)
    if not math.isfinite(raw_sigma) or raw_sigma <= 0.0:
        return 1.0

    scale = (raw_sigma / sigma_val) ** 2
    if not math.isfinite(scale) or scale <= 0.0:
        return 1.0

    return min(scale, 1.0)


def _override_help(text: str, config_path: str) -> str:
    """Return help text that explicitly names the overridden config key."""

    return f"{text} Overrides `{config_path}`."



def _underscore_flag_alias(flag: str) -> str | None:
    """Return the underscore-spelling alias for a hyphenated long flag."""

    if not flag.startswith("--") or "-" not in flag[2:]:
        return None
    return "--" + flag[2:].replace("-", "_")



def _warn_for_deprecated_cli_aliases(
    argv: Sequence[str], alias_map: Mapping[str, str]
) -> None:
    """Warn when deprecated compatibility aliases are used."""

    seen: set[str] = set()
    for token in argv:
        option = token.split("=", 1)[0]
        canonical = alias_map.get(option)
        if canonical is None or option in seen:
            continue
        warnings.warn(
            f"{option} is deprecated; use {canonical}",
            DeprecationWarning,
            stacklevel=3,
        )
        seen.add(option)



def _add_cli_argument(
    container,
    *flags: str,
    alias_map: dict[str, str],
    aliases: Sequence[str] = (),
    add_underscore_aliases: bool = True,
    **kwargs,
) -> argparse.Action:
    """Add a CLI argument with hidden compatibility aliases."""

    action = container.add_argument(*flags, **kwargs)
    canonical = next((flag for flag in flags if flag.startswith("--")), flags[0])

    hidden_aliases: list[str] = []
    for alias in aliases:
        if alias not in flags and alias not in hidden_aliases:
            hidden_aliases.append(alias)

    if add_underscore_aliases:
        for flag in (*flags, *aliases):
            alias = _underscore_flag_alias(flag)
            if alias and alias not in flags and alias not in hidden_aliases:
                hidden_aliases.append(alias)

    alias_kwargs = dict(kwargs)
    alias_kwargs["dest"] = action.dest
    alias_kwargs["help"] = argparse.SUPPRESS
    alias_kwargs["default"] = argparse.SUPPRESS
    alias_kwargs.pop("required", None)

    for alias in hidden_aliases:
        container.add_argument(alias, **alias_kwargs)
        alias_map[alias] = canonical

    return action



def parse_args(argv=None):
    """Parse command line arguments."""

    p = argparse.ArgumentParser(
        description="Run the full Radon Monitor analysis pipeline on merged event data.",
        epilog=(
            "Hyphenated long flags are canonical. Deprecated underscore aliases "
            "remain accepted for compatibility. Experimental options: "
            "--background-model loglin_unit and --likelihood extended."
        ),
    )
    alias_map: dict[str, str] = {}
    default_cfg = Path(__file__).resolve().with_name("config.yaml")
    default_input = Path.cwd() / "merged_output.csv"

    inputs_group = p.add_argument_group("Inputs and outputs")
    time_group = p.add_argument_group("Time selection and baseline")
    fit_group = p.add_argument_group("Calibration and fit controls")
    aux_group = p.add_argument_group("Efficiency and systematics inputs")
    output_group = p.add_argument_group("Plotting, diagnostics, and reproducibility")

    _add_cli_argument(
        inputs_group,
        "--config",
        "-c",
        alias_map=alias_map,
        default=str(default_cfg),
        help="YAML or JSON configuration file.",
    )
    _add_cli_argument(
        inputs_group,
        "--input",
        "-i",
        alias_map=alias_map,
        default=str(default_input),
        help=(
            "Merged event CSV. Must contain at least `timestamp` and `adc`. "
            f"Default: {default_input}."
        ),
    )
    _add_cli_argument(
        inputs_group,
        "--output-dir",
        "-o",
        alias_map=alias_map,
        default="results",
        help=(
            "Parent directory for the timestamped results folder. "
            "Use `--job-id` to choose the folder name. Default: results."
        ),
    )
    _add_cli_argument(
        inputs_group,
        "--job-id",
        alias_map=alias_map,
        help="Use this exact results folder name instead of an auto-generated timestamp.",
    )
    _add_cli_argument(
        inputs_group,
        "--overwrite",
        alias_map=alias_map,
        action="store_true",
        help="Replace an existing results folder.",
    )
    _add_cli_argument(
        inputs_group,
        "--timezone",
        alias_map=alias_map,
        default="UTC",
        help="Timezone applied to naive input timestamps. Default: UTC.",
    )

    _add_cli_argument(
        time_group,
        "--analysis-start-time",
        alias_map=alias_map,
        type=str,
        help=_override_help(
            "Reference start time for the analysis window (ISO string or epoch).",
            "analysis.analysis_start_time",
        ),
    )
    _add_cli_argument(
        time_group,
        "--analysis-end-time",
        alias_map=alias_map,
        type=str,
        help=_override_help(
            "Ignore events after this timestamp (ISO string or epoch).",
            "analysis.analysis_end_time",
        ),
    )
    _add_cli_argument(
        time_group,
        "--baseline-range",
        alias_map=alias_map,
        nargs=2,
        metavar=("TSTART", "TEND"),
        type=str,
        help=_override_help(
            "Baseline interval to extract with the same energy cuts used for the main run. "
            "Provide ISO timestamps or epoch seconds.",
            "baseline.range",
        ),
    )
    _add_cli_argument(
        time_group,
        "--baseline-mode",
        alias_map=alias_map,
        choices=["none", "electronics", "radon", "all"],
        default="all",
        help="Background removal strategy. Default: all.",
    )
    _add_cli_argument(
        time_group,
        "--allow-negative-baseline",
        alias_map=alias_map,
        action="store_true",
        help="Preserve negative baseline-corrected rates instead of clipping them to zero.",
    )
    _add_cli_argument(
        time_group,
        "--allow-negative-activity",
        alias_map=alias_map,
        action="store_true",
        help="Continue when the inferred total radon activity is negative.",
    )
    _add_cli_argument(
        time_group,
        "--check-baseline-only",
        alias_map=alias_map,
        action="store_true",
        help="Print baseline diagnostics and exit without running the full analysis.",
    )
    _add_cli_argument(
        time_group,
        "--spike-start-time",
        alias_map=alias_map,
        help=_override_help(
            "Discard events after this timestamp.",
            "analysis.spike_start_time",
        ),
    )
    _add_cli_argument(
        time_group,
        "--spike-end-time",
        alias_map=alias_map,
        help=_override_help(
            "Discard events before this timestamp.",
            "analysis.spike_end_time",
        ),
    )
    _add_cli_argument(
        time_group,
        "--spike-period",
        alias_map=alias_map,
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help=_override_help(
            "Discard events between START and END. Repeat the flag to add more windows.",
            "analysis.spike_periods",
        ),
    )
    _add_cli_argument(
        time_group,
        "--run-period",
        alias_map=alias_map,
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help=_override_help(
            "Keep events between START and END. Repeat the flag to add more windows.",
            "analysis.run_periods",
        ),
    )
    _add_cli_argument(
        time_group,
        "--radon-interval",
        alias_map=alias_map,
        nargs=2,
        metavar=("START", "END"),
        help=_override_help(
            "Time interval used for radon delta calculations.",
            "analysis.radon_interval",
        ),
    )
    _add_cli_argument(
        time_group,
        "--settle-s",
        alias_map=alias_map,
        type=float,
        help="Discard this many seconds from the start before fitting the decay curve.",
    )

    _add_cli_argument(
        fit_group,
        "--iso",
        alias_map=alias_map,
        choices=["radon", "po218", "po214"],
        help="Choose which progeny drives the final radon estimate.",
    )
    _add_cli_argument(
        fit_group,
        "--burst-mode",
        alias_map=alias_map,
        choices=["none", "micro", "rate", "both"],
        help=_override_help(
            "Burst filtering mode passed to `apply_burst_filter`.",
            "burst_filter.burst_mode",
        ),
    )
    _add_cli_argument(
        fit_group,
        "--burst-sensitivity-scan",
        alias_map=alias_map,
        action="store_true",
        help="Sweep burst parameters and plot activity versus burst window and multiplier.",
    )
    _add_cli_argument(
        fit_group,
        "--slope",
        alias_map=alias_map,
        type=float,
        help=_override_help(
            "Apply a linear ADC drift correction with this slope.",
            "systematics.adc_drift_rate",
        ),
    )
    _add_cli_argument(
        fit_group,
        "--noise-cutoff",
        alias_map=alias_map,
        type=int,
        help=_override_help(
            "ADC threshold for the noise cut.",
            "calibration.noise_cutoff",
        ),
    )
    _add_cli_argument(
        fit_group,
        "--calibration-slope",
        alias_map=alias_map,
        type=float,
        help=_override_help(
            "Fixed MeV-per-ADC conversion slope.",
            "calibration.slope_mev_per_ch",
        ),
    )
    _add_cli_argument(
        fit_group,
        "--float-slope",
        alias_map=alias_map,
        action="store_true",
        help="Treat a supplied calibration slope as an initial guess instead of fixing it.",
    )
    _add_cli_argument(
        fit_group,
        "--calibration-method",
        alias_map=alias_map,
        choices=["two-point", "auto"],
        help=_override_help(
            "Energy calibration strategy.",
            "calibration.method",
        ),
    )
    _add_cli_argument(
        fit_group,
        "--background-model",
        alias_map=alias_map,
        choices=["linear", "loglin_unit"],
        help="Experimental background model. Omitting this keeps the legacy behavior.",
    )
    _add_cli_argument(
        fit_group,
        "--likelihood",
        alias_map=alias_map,
        choices=["current", "extended"],
        help="Experimental spectral likelihood. Omitting this keeps the legacy behavior.",
    )
    _add_cli_argument(
        fit_group,
        "--hl-po214",
        alias_map=alias_map,
        type=float,
        help=_override_help(
            "Po-214 half-life in seconds.",
            "time_fit.hl_po214",
        ),
    )
    _add_cli_argument(
        fit_group,
        "--hl-po218",
        alias_map=alias_map,
        type=float,
        help=_override_help(
            "Po-218 half-life in seconds.",
            "time_fit.hl_po218",
        ),
    )
    _add_cli_argument(
        fit_group,
        "--eff-fixed",
        alias_map=alias_map,
        action="store_true",
        help="Fix all efficiencies to exactly 1.0 with no prior width.",
    )

    _add_cli_argument(
        aux_group,
        "--efficiency-json",
        alias_map=alias_map,
        help="JSON file whose `efficiency` block is merged into the config.",
    )
    _add_cli_argument(
        aux_group,
        "--systematics-json",
        alias_map=alias_map,
        help="JSON file whose `systematics` block overrides the config.",
    )
    _add_cli_argument(
        aux_group,
        "--spike-count",
        alias_map=alias_map,
        type=float,
        help="Counts observed during the spike run.",
    )
    _add_cli_argument(
        aux_group,
        "--spike-count-err",
        alias_map=alias_map,
        type=float,
        help="Uncertainty on the observed spike counts.",
    )
    _add_cli_argument(
        aux_group,
        "--spike-activity",
        alias_map=alias_map,
        type=float,
        help="Known spike activity in Bq.",
    )
    _add_cli_argument(
        aux_group,
        "--spike-duration",
        alias_map=alias_map,
        type=float,
        help="Spike-run duration in seconds.",
    )
    _add_cli_argument(
        aux_group,
        "--no-spike",
        alias_map=alias_map,
        action="store_true",
        help="Disable spike-efficiency handling entirely.",
    )

    _add_cli_argument(
        output_group,
        "--plot-time-binning-mode",
        alias_map=alias_map,
        dest="time_bin_mode_new",
        choices=["auto", "fd", "fixed"],
        help=_override_help(
            "Time-series binning mode.",
            "plotting.plot_time_binning_mode",
        ),
    )
    _add_cli_argument(
        output_group,
        "--time-bin-mode",
        alias_map=alias_map,
        dest="time_bin_mode_old",
        choices=["auto", "fd", "fixed"],
        help="Deprecated alias for `--plot-time-binning-mode`.",
    )
    _add_cli_argument(
        output_group,
        "--plot-time-bin-width",
        alias_map=alias_map,
        dest="time_bin_width",
        type=float,
        help=_override_help(
            "Fixed time-bin width in seconds.",
            "plotting.plot_time_bin_width_s",
        ),
    )
    _add_cli_argument(
        output_group,
        "--dump-ts-json",
        alias_map=alias_map,
        dest="dump_ts_json",
        aliases=("--dump-time-series-json",),
        action="store_true",
        help="Write `*_ts.json` files with binned time-series data.",
    )
    _add_cli_argument(
        output_group,
        "--ambient-file",
        alias_map=alias_map,
        help="Two-column text file containing timestamp and ambient concentration in Bq/L.",
    )
    _add_cli_argument(
        output_group,
        "--ambient-concentration",
        alias_map=alias_map,
        type=float,
        help=_override_help(
            "Constant ambient radon concentration in Bq/L for the equivalent-air plot.",
            "analysis.ambient_concentration",
        ),
    )
    _add_cli_argument(
        output_group,
        "--palette",
        alias_map=alias_map,
        help=_override_help(
            "Color palette used for plots.",
            "plotting.palette",
        ),
    )
    _add_cli_argument(
        output_group,
        "--strict-covariance",
        alias_map=alias_map,
        action="store_true",
        help="Fail instead of continuing when a fit covariance matrix is not positive definite.",
    )
    _add_cli_argument(
        output_group,
        "--debug",
        alias_map=alias_map,
        action="store_true",
        help=_override_help(
            "Enable debug logging.",
            "pipeline.log_level",
        ),
    )
    _add_cli_argument(
        output_group,
        "--seed",
        alias_map=alias_map,
        type=int,
        help=_override_help(
            "Override the random seed used by the analysis.",
            "pipeline.random_seed",
        ),
    )
    _add_cli_argument(
        output_group,
        "--hierarchical-summary",
        alias_map=alias_map,
        metavar="OUTFILE",
        help="Combine previous run summaries and write a hierarchical fit report to OUTFILE.",
    )
    _add_cli_argument(
        output_group,
        "--reproduce",
        alias_map=alias_map,
        metavar="SUMMARY",
        help="Load config and seed from a previous run's `summary.json`.",
    )

    argv_list = list(sys.argv[1:] if argv is None else argv)
    _warn_for_deprecated_cli_aliases(argv_list, alias_map)
    args = p.parse_args(argv_list)

    if args.time_bin_mode_new is not None and args.time_bin_mode_old is not None:
        if args.time_bin_mode_new != args.time_bin_mode_old:
            p.error(
                "--plot-time-binning-mode conflicts with deprecated --time-bin-mode"
            )

    args.time_bin_mode = (
        args.time_bin_mode_new
        if args.time_bin_mode_new is not None
        else args.time_bin_mode_old
    )
    del args.time_bin_mode_new
    del args.time_bin_mode_old

    return args



def main(argv=None):
    cli_args = [sys.argv[0]] + (list(argv) if argv is not None else sys.argv[1:])
    cli_sha256 = hashlib.sha256(" ".join(cli_args).encode("utf-8")).hexdigest()
    try:
        commit = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], encoding="utf-8"
        ).strip()
    except Exception:
        commit = "unknown"

    try:
        req_path = Path(__file__).resolve().parent / "requirements.txt"
        with open(req_path, "rb") as f:
            requirements_sha256 = hashlib.sha256(f.read()).hexdigest()
    except Exception:
        requirements_sha256 = "unknown"

    args = parse_args(argv)
    timer = PipelineTimer(logging.getLogger("analyze.timer"))

    if args.reproduce:
        rep_path = Path(args.reproduce)
        try:
            with open(rep_path, "r", encoding="utf-8") as f:
                rep_summary = json.load(f)
        except Exception as e:
            logger.error("Could not load summary '%s': %s", args.reproduce, e)
            sys.exit(1)
        args.config = rep_path.parent / "config_used.json"
        args.seed = rep_summary.get("random_seed")

    # Convert CLI paths to Path objects
    args.config = Path(args.config)
    args.input = Path(args.input)
    args.output_dir = Path(args.output_dir)
    if args.efficiency_json:
        args.efficiency_json = Path(args.efficiency_json)
    if args.systematics_json:
        args.systematics_json = Path(args.systematics_json)
    if args.ambient_file:
        args.ambient_file = Path(args.ambient_file)
    if args.hierarchical_summary:
        args.hierarchical_summary = Path(args.hierarchical_summary)

    pre_spec_energies = np.array([])
    post_spec_energies = np.array([])
    roi_diff = {}
    scan_results = {}
    best_params = None

    # Resolve timezone for subsequent time parsing
    tzinfo = gettz(args.timezone)
    if tzinfo is None:
        logger.error("Unknown timezone '%s'", args.timezone)
        sys.exit(1)

    if args.baseline_range:
        args.baseline_range = [
            parse_time_arg(t, tz=tzinfo) for t in args.baseline_range
        ]
    if args.analysis_end_time is not None:
        args.analysis_end_time = parse_time_arg(args.analysis_end_time, tz=tzinfo)
    if args.analysis_start_time is not None:
        args.analysis_start_time = parse_time_arg(args.analysis_start_time, tz=tzinfo)
    if args.spike_start_time is not None:
        args.spike_start_time = parse_time_arg(args.spike_start_time, tz=tzinfo)
    if args.spike_end_time is not None:
        args.spike_end_time = parse_time_arg(args.spike_end_time, tz=tzinfo)
    if args.spike_period:
        args.spike_period = [
            [parse_time_arg(s, tz=tzinfo), parse_time_arg(e, tz=tzinfo)]
            for s, e in args.spike_period
        ]
    if args.run_period:
        args.run_period = [
            [parse_time_arg(s, tz=tzinfo), parse_time_arg(e, tz=tzinfo)]
            for s, e in args.run_period
        ]
    if args.radon_interval:
        args.radon_interval = [
            parse_time_arg(args.radon_interval[0], tz=tzinfo),
            parse_time_arg(args.radon_interval[1], tz=tzinfo),
        ]

    # ────────────────────────────────────────────────────────────
    # 1. Load configuration
    # ────────────────────────────────────────────────────────────
    with timer.section("load_config"):
        try:
            cfg = load_config(args.config)
        except Exception as e:
            logger.error("Could not load config '%s': %s", args.config, e)
            sys.exit(1)

    def _log_override(section, key, new_val):
        prev = cfg.get(section, {}).get(key)
        if prev is not None and prev != new_val:
            logging.info(
                f"Overriding {section}.{key}={prev!r} with {new_val!r} from CLI"
            )

    # Apply optional overrides from command-line arguments
    if args.efficiency_json:
        try:
            with open(args.efficiency_json, "r", encoding="utf-8") as f:
                cfg["efficiency"] = json.load(f)
        except Exception as e:
            logger.error(
                "Could not load efficiency JSON '%s': %s", args.efficiency_json, e
            )
            sys.exit(1)

    if args.systematics_json:
        try:
            with open(args.systematics_json, "r", encoding="utf-8") as f:
                cfg["systematics"] = json.load(f)
        except Exception as e:
            logger.error(
                "Could not load systematics JSON '%s': %s",
                args.systematics_json,
                e,
            )
            sys.exit(1)

    if args.seed is not None:
        _log_override("pipeline", "random_seed", int(args.seed))
        cfg.setdefault("pipeline", {})["random_seed"] = int(args.seed)

    if args.ambient_concentration is not None:
        ambient_cli = _safe_float(args.ambient_concentration)
        if ambient_cli is None:
            logger.warning(
                "Ignoring ambient concentration override %r; could not convert to float",
                args.ambient_concentration,
            )
        else:
            _log_override(
                "analysis",
                "ambient_concentration",
                ambient_cli,
            )
            cfg.setdefault("analysis", {})["ambient_concentration"] = ambient_cli

    if args.analysis_end_time is not None:
        _log_override("analysis", "analysis_end_time", args.analysis_end_time)
        cfg.setdefault("analysis", {})["analysis_end_time"] = args.analysis_end_time

    if args.analysis_start_time is not None:
        _log_override("analysis", "analysis_start_time", args.analysis_start_time)
        cfg.setdefault("analysis", {})["analysis_start_time"] = args.analysis_start_time

    if args.spike_start_time is not None:
        _log_override("analysis", "spike_start_time", args.spike_start_time)
        cfg.setdefault("analysis", {})["spike_start_time"] = args.spike_start_time

    if args.spike_end_time is not None:
        _log_override("analysis", "spike_end_time", args.spike_end_time)
        cfg.setdefault("analysis", {})["spike_end_time"] = args.spike_end_time

    if args.spike_period:
        _log_override("analysis", "spike_periods", args.spike_period)
        cfg.setdefault("analysis", {})["spike_periods"] = [
            [s, e] for s, e in args.spike_period
        ]

    if args.run_period:
        _log_override("analysis", "run_periods", args.run_period)
        cfg.setdefault("analysis", {})["run_periods"] = [
            [s, e] for s, e in args.run_period
        ]

    if args.radon_interval:
        _log_override("analysis", "radon_interval", args.radon_interval)
        cfg.setdefault("analysis", {})["radon_interval"] = [
            args.radon_interval[0],
            args.radon_interval[1],
        ]

    if args.background_model is not None:
        _log_override("analysis", "background_model", args.background_model)
        cfg.setdefault("analysis", {})["background_model"] = args.background_model

    if args.likelihood is not None:
        _log_override("analysis", "likelihood", args.likelihood)
        cfg.setdefault("analysis", {})["likelihood"] = args.likelihood

    if args.settle_s is not None:
        _log_override("analysis", "settle_s", float(args.settle_s))
        cfg.setdefault("analysis", {})["settle_s"] = float(args.settle_s)

    if args.hl_po214 is not None:
        tf = cfg.setdefault("time_fit", {})
        sig = 0.0
        current = tf.get("hl_po214")
        if isinstance(current, list) and len(current) > 1:
            sig = current[1]
        _log_override("time_fit", "hl_po214", [float(args.hl_po214), sig])
        tf["hl_po214"] = [float(args.hl_po214), sig]

    if args.hl_po218 is not None:
        tf = cfg.setdefault("time_fit", {})
        sig = 0.0
        current = tf.get("hl_po218")
        if isinstance(current, list) and len(current) > 1:
            sig = current[1]
        _log_override("time_fit", "hl_po218", [float(args.hl_po218), sig])
        tf["hl_po218"] = [float(args.hl_po218), sig]

    if args.time_bin_mode:
        _log_override("plotting", "plot_time_binning_mode", args.time_bin_mode)
        cfg.setdefault("plotting", {})["plot_time_binning_mode"] = args.time_bin_mode
    if args.time_bin_width is not None:
        _log_override(
            "plotting",
            "plot_time_bin_width_s",
            float(args.time_bin_width),
        )
        cfg.setdefault("plotting", {})["plot_time_bin_width_s"] = float(
            args.time_bin_width
        )
    if args.dump_ts_json:
        cfg.setdefault("plotting", {})["dump_time_series_json"] = True

    if args.burst_mode is not None:
        _log_override("burst_filter", "burst_mode", args.burst_mode)
        cfg.setdefault("burst_filter", {})["burst_mode"] = args.burst_mode

    if (
        args.spike_count is not None
        or args.spike_count_err is not None
        or args.spike_activity is not None
        or args.spike_duration is not None
        or args.no_spike
    ):
        eff_sec = cfg.setdefault("efficiency", {}).setdefault("spike", {})
        if args.spike_count is not None:
            eff_sec["counts"] = float(args.spike_count)
        if args.spike_count_err is not None:
            eff_sec["error"] = float(args.spike_count_err)
        if args.spike_activity is not None:
            eff_sec["activity_bq"] = float(args.spike_activity)
        if args.spike_duration is not None:
            eff_sec["live_time_s"] = float(args.spike_duration)
        if args.no_spike:
            eff_sec["enabled"] = False

    if args.slope is not None:
        _log_override("systematics", "adc_drift_rate", float(args.slope))
        cfg.setdefault("systematics", {})["adc_drift_rate"] = float(args.slope)

    if args.noise_cutoff is not None:
        _log_override(
            "calibration",
            "noise_cutoff",
            int(args.noise_cutoff),
        )
        cfg.setdefault("calibration", {})["noise_cutoff"] = int(args.noise_cutoff)

    if args.calibration_slope is not None:
        _log_override(
            "calibration",
            "slope_MeV_per_ch",
            float(args.calibration_slope),
        )
        cfg.setdefault("calibration", {})["slope_MeV_per_ch"] = float(args.calibration_slope)

    if args.float_slope:
        cfg.setdefault("calibration", {})["float_slope"] = True

    if args.iso is not None:
        prev = cfg.get("analysis_isotope")
        if prev is not None and prev != args.iso:
            logging.info(
                f"Overriding analysis_isotope={prev!r} with {args.iso!r} from CLI"
            )
        cfg["analysis_isotope"] = args.iso
    assert cfg.get("analysis_isotope", "radon") in {"radon", "po218", "po214"}

    if args.calibration_method is not None:
        _log_override("calibration", "method", args.calibration_method)
        cfg.setdefault("calibration", {})["method"] = args.calibration_method

    if args.allow_negative_baseline:
        cfg["allow_negative_baseline"] = True

    if args.debug:
        cfg.setdefault("pipeline", {})["log_level"] = "DEBUG"

    if args.palette:
        cfg.setdefault("plotting", {})["palette"] = args.palette

    # Timestamp for this analysis run
    now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    base_cfg_sha = hashlib.sha256(
        json.dumps(to_native(cfg), sort_keys=True).encode("utf-8")
    ).hexdigest()

    # Configure logging as early as possible
    log_level = cfg.get("pipeline", {}).get("log_level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level, format="%(levelname)s:%(name)s:%(message)s"
    )

    start_warning_capture()

    seed = cfg.get("pipeline", {}).get("random_seed")
    seed_used = None
    if seed is not None:
        try:
            seed_int = int(seed)
            np.random.seed(seed_int)
            random.seed(seed_int)
            seed_used = seed_int
        except Exception:
            logging.warning(f"Invalid random_seed '{seed}' ignored")
    else:
        derived_seed = int(
            hashlib.sha256((base_cfg_sha + now_str).encode("utf-8")).hexdigest()[:8],
            16,
        )
        np.random.seed(derived_seed)
        random.seed(derived_seed)
        seed_used = derived_seed
        cfg.setdefault("pipeline", {})["random_seed"] = derived_seed

    cfg_sha256 = hashlib.sha256(
        json.dumps(to_native(cfg), sort_keys=True).encode("utf-8")
    ).hexdigest()

    # ────────────────────────────────────────────────────────────
    # 2. Load event data
    # ────────────────────────────────────────────────────────────
    with timer.section("load_events"):
        try:
            events_all = load_events(args.input, column_map=cfg.get("columns"))

            # Parse timestamps to UTC ``Timestamp`` objects
            events_all["timestamp"] = events_all["timestamp"].map(parse_timestamp)

        except Exception as e:
            logger.error("Could not load events from '%s': %s", args.input, e)
            sys.exit(1)

    if events_all.empty:
        logger.info("No events found in the input CSV. Exiting.")
        sys.exit(0)

    # ``load_events()`` already returns timezone-aware ``datetime64`` values.
    # Timestamps are kept in this form and converted to epoch seconds only for
    # numerical operations.

    # ───────────────────────────────────────────────
    # 2a. Pedestal / electronic-noise cut (integer ADC)
    # ───────────────────────────────────────────────
    noise_thr = cfg.get("calibration", {}).get("noise_cutoff")
    n_removed_noise = 0
    noise_thr_val = None
    events_filtered = events_all.copy()
    with timer.section("noise_cut"):
        if noise_thr is not None:
            try:
                noise_thr_val = int(noise_thr)
            except (ValueError, TypeError):
                logging.warning(
                    f"Invalid noise_cutoff '{noise_thr}' - skipping noise cut"
                )
                noise_thr_val = None
            else:
                before = len(events_filtered)
                events_filtered = events_filtered[
                    events_filtered["adc"] > noise_thr_val
                ].reset_index(drop=True)
                n_removed_noise = before - len(events_filtered)
                if before > 0:
                    frac_removed_noise = n_removed_noise / before
                    logging.info(
                        f"Noise cut removed {n_removed_noise} events ({frac_removed_noise:.1%})"
                    )
                else:
                    logging.info(f"Noise cut removed {n_removed_noise} events")

        _ensure_events(events_filtered, "noise cut")

    events_after_noise = events_filtered.copy()

    # Optional burst filter to remove high-rate clusters
    with timer.section("burst_filter"):
        total_span = (
            events_filtered["timestamp"].max() - events_filtered["timestamp"].min()
        )
        if isinstance(total_span, (np.timedelta64, pd.Timedelta)):
            total_span = total_span / np.timedelta64(1, "s")
        rate_cps = len(events_filtered) / max(float(total_span), 1e-9)
        if args.burst_mode is None:
            current_mode = cfg.get("burst_filter", {}).get("burst_mode", "rate")
            if current_mode == "rate" and rate_cps < 0.1:
                cfg.setdefault("burst_filter", {})["burst_mode"] = "none"

        burst_mode = (
            args.burst_mode
            if args.burst_mode is not None
            else cfg.get("burst_filter", {}).get("burst_mode", "rate")
        )

        n_before_burst = len(events_filtered)
        events_filtered, n_removed_burst = apply_burst_filter(
            events_filtered, cfg, mode=burst_mode
        )
        events_after_burst = events_filtered.copy()
        if n_before_burst > 0:
            frac_removed = n_removed_burst / n_before_burst
            logging.info(
                f"Burst filter removed {n_removed_burst} events ({frac_removed:.1%})"
            )
            if frac_removed > 0.5:
                logging.warning(
                    f"More than half of events vetoed by burst filter ({frac_removed:.1%})"
                )

        _ensure_events(events_filtered, "burst filtering")

    # Global t₀ reference
    t0_cfg = cfg.get("analysis", {}).get("analysis_start_time")
    if t0_cfg is not None:
        try:
            t0_global = to_utc_datetime(t0_cfg)
            t0_cfg = t0_global
            cfg.setdefault("analysis", {})["analysis_start_time"] = t0_global
        except Exception:
            logging.warning(
                f"Invalid analysis_start_time '{t0_cfg}' - using first event"
            )
            t0_global = to_utc_datetime(events_filtered["timestamp"].min())
    else:
        t0_global = to_utc_datetime(events_filtered["timestamp"].min())
        t0_cfg = t0_global

    t_end_cfg = cfg.get("analysis", {}).get("analysis_end_time")
    t_end_global = None
    t_end_global_ts = None
    if t_end_cfg is not None:
        try:
            t_end_dt = to_utc_datetime(t_end_cfg)
            t_end_global = t_end_dt
            t_end_global_ts = t_end_dt.timestamp()
            t_end_cfg = t_end_dt
            cfg.setdefault("analysis", {})["analysis_end_time"] = t_end_dt
        except Exception:
            logging.warning(
                f"Invalid analysis_end_time '{t_end_cfg}' - using last event"
            )
            t_end_global = None
            t_end_global_ts = None

    spike_start_cfg = cfg.get("analysis", {}).get("spike_start_time")
    t_spike_start = None
    if spike_start_cfg is not None:
        try:
            t_spike_start_dt = to_utc_datetime(spike_start_cfg)
            t_spike_start = t_spike_start_dt
            cfg.setdefault("analysis", {})["spike_start_time"] = t_spike_start_dt
        except Exception:
            logging.warning(f"Invalid spike_start_time '{spike_start_cfg}' - ignoring")
            t_spike_start = None

    spike_end_cfg = cfg.get("analysis", {}).get("spike_end_time")
    t_spike_end = None
    if spike_end_cfg is not None:
        try:
            t_spike_end_dt = to_utc_datetime(spike_end_cfg)
            t_spike_end = t_spike_end_dt
            cfg.setdefault("analysis", {})["spike_end_time"] = t_spike_end_dt
        except Exception:
            logging.warning(f"Invalid spike_end_time '{spike_end_cfg}' - ignoring")
            t_spike_end = None

    spike_periods_cfg = cfg.get("analysis", {}).get("spike_periods", [])
    if spike_periods_cfg is None:
        spike_periods_cfg = []
    spike_periods = []
    for period in spike_periods_cfg:
        try:
            start, end = period
            start_ts = to_utc_datetime(start)
            end_ts = to_utc_datetime(end)
            if end_ts <= start_ts:
                raise ValueError("end <= start")
            spike_periods.append([start_ts, end_ts])
        except Exception as e:
            logging.warning(f"Invalid spike_period {period} -> {e}")
    if spike_periods:
        cfg.setdefault("analysis", {})["spike_periods"] = spike_periods
        spike_periods_cfg = spike_periods

    run_periods_cfg = cfg.get("analysis", {}).get("run_periods", [])
    if run_periods_cfg is None:
        run_periods_cfg = []
    run_periods = []
    for period in run_periods_cfg:
        try:
            start, end = period
            start_ts = to_utc_datetime(start)
            end_ts = to_utc_datetime(end)
            if end_ts <= start_ts:
                raise ValueError("end <= start")
            run_periods.append([start_ts, end_ts])
        except Exception as e:
            logging.warning(f"Invalid run_period {period} -> {e}")
    if run_periods:
        cfg.setdefault("analysis", {})["run_periods"] = run_periods
        run_periods_cfg = run_periods

    radon_interval_cfg = cfg.get("analysis", {}).get("radon_interval")
    radon_interval = None
    if radon_interval_cfg:
        try:
            start_r, end_r = radon_interval_cfg
            start_r_dt = to_utc_datetime(start_r)
            end_r_dt = to_utc_datetime(end_r)
            if end_r_dt <= start_r_dt:
                raise ValueError("end <= start")
            radon_interval = [start_r_dt, end_r_dt]
            cfg.setdefault("analysis", {})["radon_interval"] = radon_interval
            radon_interval_cfg = radon_interval
        except Exception as e:
            logging.warning(f"Invalid radon_interval {radon_interval_cfg} -> {e}")
            radon_interval = None

    with timer.section("prepare_analysis_df"):
        (
            df_analysis,
            analysis_start,
            analysis_end,
            t_end_global_ts,
            drift_rate,
            drift_mode,
            drift_params,
        ) = prepare_analysis_df(
            events_filtered,
            t_spike_start,
            t_spike_end,
            spike_periods,
            run_periods,
            t_end_global,
            t0_global=t0_global,
            cfg=cfg,
            args=args,
        )
    t_end_global = analysis_end
    if t_end_cfg is None:
        t_end_cfg = t_end_global

    if drift_rate != 0.0 or drift_mode != "linear" or drift_params is not None:
        try:
            ts_seconds = df_analysis["timestamp"].map(to_epoch_seconds).to_numpy()
            df_analysis["adc"] = apply_linear_adc_shift(
                df_analysis["adc"].values,
                ts_seconds,
                float(drift_rate),
                t_ref=t0_global.timestamp(),
                mode=drift_mode,
                params=drift_params,
            )
        except Exception as e:
            if not cfg.get("allow_fallback"):
                raise
            logger.warning("Could not apply ADC drift correction -> %s", e)

    cal_window_rel_unc: dict[str, float] = {}  # populated after calibration

    # ────────────────────────────────────────────────────────────
    with timer.section("energy_calibration"):
        # 3. Energy calibration
        # ────────────────────────────────────────────────────────────
        adc_vals = df_analysis["adc"].values
        hist_bins = cfg["calibration"].get("hist_bins", 2000)
        calibration_valid = True
        try:
            if cfg.get("calibration", {}).get("method", "two-point") == "auto":
                adc_arr = df_analysis["adc"].to_numpy()
                cal_params = derive_calibration_constants_auto(
                    adc_arr,
                    noise_cutoff=cfg["calibration"].get(
                        "noise_cutoff", DEFAULT_NOISE_CUTOFF
                    ),
                    hist_bins=hist_bins,
                    peak_search_radius=cfg["calibration"].get("peak_search_radius", 200),
                    nominal_adc=cfg["calibration"].get("nominal_adc"),
                )
            else:
                # Two‐point calibration as given in config
                cal_params = derive_calibration_constants(adc_vals, config=cfg)
        except Exception:
            logging.exception("calibration failed --using defaults")
            if not cfg.get("allow_fallback"):
                raise
            calibration_valid = False
            cal_params = {"a": (0.005, 0.001), "c": (0.02, 0.005), "sigma_E": (0.3, 0.1)}
    
        def _value_sigma(val):
            if isinstance(val, (list, tuple, np.ndarray)):
                if len(val) >= 2:
                    return float(val[0]), float(val[1])
                if len(val) == 1:
                    return float(val[0]), 0.0
                return 0.0, 0.0
            return float(val), 0.0
    
        def _as_cal_result(obj):
            from calibration import CalibrationResult
    
            if isinstance(obj, CalibrationResult):
                return obj
    
            a, a_sig = _value_sigma(obj.get("a", 0.0))
            c, c_sig = _value_sigma(obj.get("c", 0.0))
            a2, a2_sig = _value_sigma(obj.get("a2", 0.0))
            a3, a3_sig = _value_sigma(obj.get("a3", 0.0))
            sigma_E, sigma_E_error = _value_sigma(obj.get("sigma_E", 0.0))

            coeffs = [c, a]
            cov = np.array([[c_sig**2, 0.0], [0.0, a_sig**2]])

            if "ac_covariance" in obj:
                cov_matrix = np.asarray(obj["ac_covariance"], dtype=float)
                if cov_matrix.shape >= (2, 2):
                    cov_ac = float(cov_matrix[0][1])
                    cov[0, 1] = cov[1, 0] = cov_ac
                else:
                    logger.warning(f"ac_covariance matrix has invalid shape {cov_matrix.shape}, expected at least (2,2)")

            if "a2" in obj:
                coeffs.append(a2)
                cov = np.pad(cov, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
                cov[2, 2] = a2_sig**2
                cov[1, 2] = cov[2, 1] = float(obj.get("cov_a_a2", 0.0))
                cov[0, 2] = cov[2, 0] = float(obj.get("cov_a2_c", 0.0))

            if "a3" in obj:
                coeffs.append(a3)
                cov = np.pad(cov, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
                cov[3, 3] = a3_sig**2
    
            return CalibrationResult(
                coeffs=coeffs,
                cov=cov,
                sigma_E=sigma_E,
                sigma_E_error=sigma_E_error,
                peaks=obj.get("peaks"),
            )
    
        cal_result = _as_cal_result(cal_params)
    
        # Save "a, c, sigma_E" so we can reconstruct energies
        if isinstance(cal_params, dict):
            a, a_sig = _value_sigma(cal_params.get("a", 0.0))
            a2, a2_sig = _value_sigma(cal_params.get("a2", 0.0))
            a3, a3_sig = _value_sigma(cal_params.get("a3", 0.0))
            c, c_sig = _value_sigma(cal_params.get("c", 0.0))
            sigE_mean, sigE_sigma = _value_sigma(cal_params.get("sigma_E", 0.0))
            cov_mat = np.asarray(
                cal_params.get("ac_covariance", [[0.0, 0.0], [0.0, 0.0]]), dtype=float
            )
            cov_ac = float(cov_mat[0, 1])
            cov_a_a2 = float(cal_params.get("cov_a_a2", 0.0))
            cov_a2_c = float(cal_params.get("cov_a2_c", 0.0))
        else:
            from calibration import CalibrationResult
    
            assert isinstance(cal_params, CalibrationResult)
            idx = {exp: i for i, exp in enumerate(cal_params._exponents)}
            c = cal_params.coeffs[idx[0]] if 0 in idx else 0.0
            a = cal_params.coeffs[idx[1]] if 1 in idx else 0.0
            a2 = cal_params.coeffs[idx[2]] if 2 in idx else 0.0
            a3 = cal_params.coeffs[idx[3]] if 3 in idx else 0.0
            sigE_mean = cal_params.sigma_E
            sigE_sigma = cal_params.sigma_E_error
            try:
                c_sig = float(np.sqrt(cal_params.get_cov("c", "c")))
            except KeyError:
                c_sig = 0.0
            try:
                a_sig = float(np.sqrt(cal_params.get_cov("a", "a")))
            except KeyError:
                a_sig = 0.0
            try:
                a2_sig = float(np.sqrt(cal_params.get_cov("a2", "a2")))
            except KeyError:
                a2_sig = 0.0
            try:
                a3_sig = float(np.sqrt(cal_params.get_cov("a3", "a3")))
            except KeyError:
                a3_sig = 0.0
            try:
                cov_ac = cal_params.get_cov("a", "c")
            except KeyError:
                cov_ac = 0.0
            try:
                cov_a_a2 = cal_params.get_cov("a", "a2")
            except KeyError:
                cov_a_a2 = 0.0
            try:
                cov_a2_c = cal_params.get_cov("a2", "c")
            except KeyError:
                cov_a2_c = 0.0
    
        # Apply calibration -> new column "energy_MeV" and its uncertainty
        energies = cal_result.predict(df_analysis["adc"])
        df_analysis["energy_MeV"] = energies
        df_analysis["denergy_MeV"] = cal_result.uncertainty(df_analysis["adc"])
    
        energies_pre_burst = cal_result.predict(events_after_noise["adc"])
        energies_post_burst = cal_result.predict(events_after_burst["adc"])
        roi_diff = _roi_diff(energies_pre_burst, energies_post_burst, cfg)
        pre_spec_energies = energies_pre_burst
        post_spec_energies = energies_post_burst
    
        # Derive default time-fit windows from calibration peaks when missing
        if getattr(cal_result, "peaks", None):
            tf_cfg = cfg.setdefault("time_fit", {})
            for iso in ("Po210", "Po218", "Po214"):
                key = f"window_{iso.lower()}"
                if tf_cfg.get(key) is None:
                    peak_E = cal_result.peaks.get(iso, {}).get("centroid_mev")
                    if peak_E is not None:
                        tf_cfg[key] = [float(peak_E - 0.08), float(peak_E + 0.08)]

        # Calibration window efficiency uncertainty (fractional systematic per isotope)
        cal_window_rel_unc: dict[str, float] = _compute_cal_window_rel_unc(cal_result, cfg)
        if cal_window_rel_unc:
            logging.info(
                "Calibration window rel. uncertainties: %s",
                {k: f"{v:.4f}" for k, v in cal_window_rel_unc.items()},
            )

        # ────────────────────────────────────────────────────────────
    with timer.section("baseline"):
        # 4. Baseline run (optional)
        # ────────────────────────────────────────────────────────────
        baseline_info = {}
        baseline_counts = {}
        baseline_record = None
        baseline_background_provenance: dict[str, dict[str, Any]] = {}
        dilution_factor = None
        baseline_cfg = cfg.get("baseline", {})
        isotopes_to_subtract = baseline_cfg.get("isotopes_to_subtract", ["Po214", "Po218"])
        baseline_range = None
        if args.baseline_range:
            _log_override("baseline", "range", args.baseline_range)
            baseline_range = (args.baseline_range[0], args.baseline_range[1])
            logging.info(
                "Baseline window: %s \u2192 %s",
                baseline_range[0].isoformat(),
                baseline_range[1].isoformat(),
            )
            cfg.setdefault("baseline", {})["range"] = [
                baseline_range[0],
                baseline_range[1],
            ]
        elif "range" in baseline_cfg:
            try:
                b0, b1 = baseline_cfg.get("range")
                start_dt = to_utc_datetime(b0)
                end_dt = to_utc_datetime(b1)
                baseline_range = (start_dt, end_dt)
                baseline_cfg["range"] = [start_dt, end_dt]
            except Exception as e:
                if not cfg.get("allow_fallback"):
                    raise
                logging.warning(
                    "Invalid baseline.range %r -> %s", baseline_cfg.get("range"), e
                )
    
        # Validate baseline window against analysis times
        try:
            validate_baseline_window(cfg)
        except ValueError as e:
            raise
    
        try:
            monitor_vol = float(baseline_cfg.get("monitor_volume_l", 605.0))
            sample_vol = float(baseline_cfg.get("sample_volume_l", 0.0))
        except (TypeError, ValueError) as exc:
            raise ValueError("baseline volumes must be numeric") from exc
        base_events = pd.DataFrame()
        baseline_live_time = 0.0
        mask_base = None
    
        if baseline_range:
            t_start_base = baseline_range[0]
            t_end_base = baseline_range[1]
            if t_end_base <= t_start_base:
                raise ValueError("baseline_range end time must be greater than start time")
            events_all_ts = to_datetime_utc(events_all["timestamp"])
            mask_base_full = (events_all_ts >= t_start_base) & (events_all_ts < t_end_base)
            mask_base = (df_analysis["timestamp"] >= t_start_base) & (
                df_analysis["timestamp"] < t_end_base
            )
            if events_all_ts.size > 0 and (
                t_end_base < events_all_ts.min() or t_start_base > events_all_ts.max()
            ):
                logging.warning(
                    "Baseline interval outside data range --taking counts anyway"
                )
            base_events = events_all[mask_base_full].copy()
            # Apply calibration to the baseline events
            if not base_events.empty:
                base_events["energy_MeV"] = cal_result.predict(base_events["adc"])
                base_events["denergy_MeV"] = cal_result.uncertainty(base_events["adc"])
            else:
                base_events["energy_MeV"] = np.array([], dtype=float)
                base_events["denergy_MeV"] = np.array([], dtype=float)
            if len(base_events) == 0:
                logging.warning("baseline_range yielded zero events")
            baseline_live_time = float((t_end_base - t_start_base).total_seconds())
            cfg.setdefault("baseline", {})["range"] = [
                t_start_base,
                t_end_base,
            ]
            baseline_info = {
                "start": t_start_base,
                "end": t_end_base,
                "n_events": len(base_events),
                "live_time": baseline_live_time,
            }
    
            try:
                dilution_factor = compute_dilution_factor(monitor_vol, sample_vol)
            except ValueError as exc:
                msg = (
                    "invalid baseline volumes: "
                    f"monitor_volume_l={monitor_vol!r}, sample_volume_l={sample_vol!r}"
                )
                if cfg.get("allow_fallback"):
                    monitor_safe = max(monitor_vol, 0.0)
                    sample_safe = max(sample_vol, 0.0)
                    total_safe = monitor_safe + sample_safe
                    if monitor_safe <= 0 or total_safe <= 0:
                        raise ValueError(msg) from exc
                    logger.warning("%s --clamping to non-negative values", msg)
                    monitor_vol = monitor_safe
                    sample_vol = sample_safe
                    dilution_factor = monitor_safe / total_safe
                    warnings_list = baseline_info.setdefault("warnings", [])
                    warnings_list.append(msg)
                    baseline_info["dilution_factor_fallback"] = True
                else:
                    raise ValueError(msg) from exc
    
            scales = {
                "Po214": dilution_factor,
                "Po218": dilution_factor,
                "Po210": 1.0,
                "noise": 1.0,
            }
            baseline_info["dilution_factor"] = dilution_factor
            baseline_info["scales"] = scales
            baseline_record = baseline_handling.initialize_baseline_record(
                baseline_info,
                calibration=cal_result,
            )
    
            # Estimate electronic noise level from ADC values below Po-210
            noise_level = None
            try:
                from baseline_noise import estimate_baseline_noise
    
                peak_adc = None
                if getattr(cal_result, "peaks", None):
                    peak_adc = cal_result.peaks.get("Po210", {}).get("centroid_adc")
                if peak_adc is not None:
                    result = estimate_baseline_noise(
                        base_events["adc"].values,
                        peak_adc=peak_adc,
                        pedestal_cut=noise_thr_val,
                        return_mask=True,
                    )
                    if isinstance(result, tuple) and len(result) == 3:
                        noise_level, _, mask_noise = result
                        baseline_info["n_noise_events"] = int(np.sum(mask_noise))
                    else:
                        noise_level, _ = result
            except Exception as e:
                if not cfg.get("allow_fallback"):
                    raise
                logger.warning("Baseline noise estimation failed -> %s", e)
    
            if noise_level is not None:
                # Store estimated noise peak amplitude in counts (not ADC units)
                baseline_info["noise_level"] = float(noise_level)
    
            # Record noise counts in ``baseline_counts``
            if "mask_noise" in locals():
                baseline_counts["noise"] = int(np.sum(mask_noise))
                if baseline_record is not None:
                    baseline_handling.update_record_with_counts(
                        baseline_record,
                        "noise",
                        baseline_counts["noise"],
                        baseline_live_time,
                        1.0,
                    )
    
        _ensure_events(df_analysis, "baseline subtraction")
    
        if args.check_baseline_only:
            try:
                summary = summarize_baseline(cfg, isotopes_to_subtract)
            except BaselineError as e:
                logger.error("BaselineError: %s", e)
                sys.exit(1)
            try:
                from tabulate import tabulate
    
                rows = [
                    (iso, f"{vals[0]:.3f}", f"{vals[1]:.3f}", f"{vals[2]:.3f}")
                    for iso, vals in summary.items()
                ]
                table = tabulate(
                    rows,
                    headers=["Isotope", "Raw rate", "Baseline rate", "Corrected"],
                    tablefmt="plain",
                )
            except Exception:
                table = "\n".join(
                    f"{iso}: raw={vals[0]:.3f} baseline={vals[1]:.3f} corrected={vals[2]:.3f}"
                    for iso, vals in summary.items()
                )
            logger.info("%s", table)
            if not args.allow_negative_baseline and any(v[2] < 0 for v in summary.values()):
                sys.exit(1)
            sys.exit(0)
    
        if args.baseline_range:
            t_base0 = args.baseline_range[0]
            t_base1 = args.baseline_range[1]
            edges = adc_hist_edges(df_analysis["adc"].values, hist_bins)
            try:
                df_analysis, _ = baseline.subtract(
                    df_analysis,
                    events_all,
                    bins=edges,
                    t_base0=t_base0,
                    t_base1=t_base1,
                    mode=args.baseline_mode,
                    live_time_analysis=(analysis_end - analysis_start).total_seconds(),
                    allow_fallback=cfg.get("allow_fallback", False),
                )
            except Exception as e:
                if not cfg.get("allow_fallback"):
                    raise
                logger.warning("Baseline subtraction failed -> %s", e)
    
        # ────────────────────────────────────────────────────────────
    with timer.section("spectral_fit"):
        # 5. Spectral fit (optional)
        # ────────────────────────────────────────────────────────────
        spectrum_results = {}
        spec_plot_data = None
        peak_deviation = {}
        split_half_result = None
        model_comparison_result = None
        _pre_info = {}
        _stage2_plot_data = None
        shelf_halo_result = None
        dnl_crossval_result = None
        _per_period_crossval_data = None
        if cfg.get("spectral_fit", {}).get("do_spectral_fit", False):
            # Decide binning: new 'binning' dict or legacy keys
            spectral_cfg = cfg["spectral_fit"]
            analysis_cfg = cfg.get("analysis", {})
            df_spectrum, fit_energy_range = _select_spectral_fit_frame(
                df_analysis,
                spectral_cfg,
            )
            E_all = df_spectrum["energy_MeV"].to_numpy(dtype=float, copy=False)
            adc_all = df_spectrum["adc"].to_numpy(dtype=float, copy=False)

            if E_all.size == 0:
                if fit_energy_range is None:
                    logger.warning("No finite energies available for spectral fit")
                else:
                    logger.warning(
                        "No events in spectral fit range %.3f-%.3f MeV; skipping spectral fit",
                        fit_energy_range[0],
                        fit_energy_range[1],
                    )
            else:
                bin_cfg = spectral_cfg.get("binning")
                if bin_cfg is not None:
                    method = bin_cfg.get("method", "adc").lower()
                    default_bins = bin_cfg.get("default_bins")
                else:
                    method = str(
                        spectral_cfg.get("spectral_binning_mode", "adc")
                    ).lower()
                    default_bins = spectral_cfg.get("fd_hist_bins")

                if method == "fd":
                    if E_all.size < 2:
                        nbins = 1
                    else:
                        q25, q75 = np.percentile(E_all, [25, 75])
                        iqr = q75 - q25
                        if iqr > 0:
                            fd_width = 2 * iqr / (E_all.size ** (1 / 3))
                            nbins = max(
                                1,
                                int(np.ceil((E_all.max() - E_all.min()) / float(fd_width))),
                            )
                        else:
                            nbins = default_bins if default_bins is not None else 100

                    if fit_energy_range is not None:
                        lo, hi = fit_energy_range
                        bin_edges = np.linspace(lo, hi, int(nbins) + 1, dtype=float)
                        bins = int(nbins)
                    else:
                        bins = int(nbins)
                        bin_edges = None
                elif method == "energy":
                    width = 0.02
                    if bin_cfg is not None:
                        width = bin_cfg.get("energy_bin_width", width)
                    else:
                        width = cfg["spectral_fit"].get("energy_bin_width", width)
                    width = float(width)
                    if width <= 0:
                        raise ValueError("energy_bin_width must be positive")

                    if fit_energy_range is not None:
                        e_min, e_max = fit_energy_range
                    else:
                        e_min = float(np.min(E_all))
                        e_max = float(np.max(E_all))

                    if np.isclose(e_min, e_max):
                        e_max = e_min + width
                    n_steps = max(1, int(np.ceil((e_max - e_min) / width)))
                    if fit_energy_range is not None:
                        bin_edges = e_min + width * np.arange(n_steps, dtype=float)
                        bin_edges = np.append(bin_edges, e_max)
                    else:
                        stop = e_min + (n_steps + 1) * width
                        bin_edges = np.arange(e_min, stop + 0.5 * width, width, dtype=float)
                    bins = bin_edges.size - 1
                else:
                    # "ADC" binning mode -> fixed width in raw channels.
                    width = 1
                    if bin_cfg is not None:
                        width = bin_cfg.get("adc_bin_width", 1)
                    else:
                        width = spectral_cfg.get("adc_bin_width", 1)
                    bin_edges_adc = adc_hist_edges(adc_all, channel_width=width)
                    bins = bin_edges_adc.size - 1
                    bin_edges = apply_calibration(
                        bin_edges_adc,
                        a,
                        c,
                        quadratic_coeff=a2,
                        cubic_coeff=a3,
                    )

                expected_peaks = spectral_cfg.get("expected_peaks")
                if expected_peaks is None:
                    expected_peaks = DEFAULT_ADC_CENTROIDS

                adc_peaks = find_adc_bin_peaks(
                    adc_all,
                    expected=expected_peaks,
                    window=spectral_cfg.get("peak_search_width_adc", 50),
                    prominence=spectral_cfg.get("peak_search_prominence", 0),
                    width=spectral_cfg.get("peak_search_width_adc", None),
                    method=spectral_cfg.get("peak_search_method", "prominence"),
                    cwt_widths=spectral_cfg.get("peak_search_cwt_widths"),
                )
                # Ensure all expected isotopes are present — inject at
                # nominal ADC if the peak finder couldn't locate them.
                _force_all_isotopes = spectral_cfg.get("force_all_isotopes", True)
                if _force_all_isotopes:
                    for _fiso, _fadc in expected_peaks.items():
                        if _fiso not in adc_peaks:
                            adc_peaks[_fiso] = float(_fadc)
                            logger.info(
                                "Injecting missing isotope %s at nominal ADC=%.0f",
                                _fiso, float(_fadc),
                            )
                # Build priors for the unbinned spectrum fit.
                priors_spec = {}
                sigma_prior_source = spectral_cfg.get("sigma_e_prior_source", spectral_cfg.get("sigma_E_prior_source"))
                sigma_prior_sigma = spectral_cfg.get("sigma_e_prior_sigma", spectral_cfg.get("sigma_E_prior_sigma", sigE_sigma))

                def _coerce_sigma(val):
                    try:
                        return float(val)
                    except (TypeError, ValueError):
                        return float("nan")

                if sigma_prior_source in (None, "calibration"):
                    sigma_E_prior = float(sigE_sigma)
                elif sigma_prior_source == "config":
                    sigma_E_prior = _coerce_sigma(sigma_prior_sigma)
                else:
                    sigma_E_prior = _coerce_sigma(sigma_prior_source)

                if not np.isfinite(sigma_E_prior) or sigma_E_prior <= 0.0:
                    sigma_E_prior = _coerce_sigma(sigma_prior_sigma)
                if not np.isfinite(sigma_E_prior) or sigma_E_prior <= 0.0:
                    sigma_E_prior = max(float(sigE_sigma), 1e-6)

                float_sigma_E = bool(spectral_cfg.get("float_sigma_e", spectral_cfg.get("float_sigma_E", True)))

                priors_spec["sigma_E"] = (sigE_mean, sigma_E_prior)
                # Fit_spectrum expects separate ``sigma0`` and ``F`` resolution terms.
                # Initialise sigma0 from the calibration-derived resolution. Allow it
                # to float within the calibration uncertainty when requested while
                # keeping the Fano term fixed by default.
                if float_sigma_E:
                    priors_spec["sigma0"] = (sigE_mean, sigma_E_prior)
                    priors_spec["F"] = (0.0, float(spectral_cfg.get("F_prior_sigma", 0.01)))
                else:
                    priors_spec["sigma0"] = (sigE_mean, 0.0)
                    priors_spec["F"] = (0.0, 0.0)

                mu_bounds_units = spectral_cfg.get("mu_bounds_units", "mev")
                mu_bounds_fit = _normalise_mu_bounds(
                    spectral_cfg.get("mu_bounds"),
                    units=mu_bounds_units,
                    slope=a,
                    intercept=c,
                    quadratic_coeff=a2,
                    cubic_coeff=a3,
                )

                peak_tol = spectral_cfg.get("spectral_peak_tolerance_mev", 0.3)
                calibration_peak_mev = {}
                if getattr(cal_result, "peaks", None):
                    for iso in adc_peaks.keys():
                        peak_info = cal_result.peaks.get(iso, {})
                        peak_E = peak_info.get("centroid_mev")
                        if peak_E is not None:
                            calibration_peak_mev[iso] = float(peak_E)
                for peak, centroid_adc in adc_peaks.items():
                    mu = calibration_peak_mev.get(
                        peak,
                        apply_calibration(centroid_adc, a, c, quadratic_coeff=a2, cubic_coeff=a3),
                    )
                    bounds = mu_bounds_fit.get(peak)
                    if bounds is not None:
                        lo, hi = bounds
                        if not (lo <= mu <= hi):
                            unclipped_mu = float(mu)
                            mu = float(np.clip(mu, lo, hi))
                            logging.warning(
                                "Initial spectral mu for %s (%.4f MeV) lies outside configured bounds [%s, %s]; clipping to %.4f MeV",
                                peak,
                                unclipped_mu,
                                lo,
                                hi,
                                mu,
                            )
                    priors_spec[f"mu_{peak}"] = (mu, spectral_cfg.get("mu_sigma"))
                    raw_count = float(
                        (
                            (E_all >= mu - peak_tol)
                            & (E_all <= mu + peak_tol)
                        ).sum()
                    )
                    mu_amp = max(raw_count, 1.0)
                    sigma_amp = max(
                        np.sqrt(mu_amp), spectral_cfg.get("amp_prior_scale") * mu_amp
                    )
                    priors_spec[f"S_{peak}"] = (mu_amp, sigma_amp)

                    _use_emg_dict = spectral_cfg.get("use_emg", {})
                    _use_emg_for_peak = _use_emg_dict.get(peak, False) if isinstance(_use_emg_dict, dict) else bool(_use_emg_dict)
                    if _use_emg_for_peak:
                        # Config keys use lowercase isotope names (tau_po210_prior_mean)
                        # while peak names are capitalized (Po210). Try both.
                        _tau_mean = spectral_cfg.get(f"tau_{peak}_prior_mean")
                        if _tau_mean is None:
                            _tau_mean = spectral_cfg.get(f"tau_{peak.lower()}_prior_mean")
                        _tau_sigma = spectral_cfg.get(f"tau_{peak}_prior_sigma")
                        if _tau_sigma is None:
                            _tau_sigma = spectral_cfg.get(f"tau_{peak.lower()}_prior_sigma")
                        if _tau_mean is not None and _tau_sigma is not None:
                            priors_spec[f"tau_{peak}"] = (
                                float(_tau_mean),
                                float(_tau_sigma),
                            )

                # Per-isotope energy resolution priors
                per_iso_sigma = spectral_cfg.get("per_isotope_sigma", {})
                for peak in adc_peaks.keys():
                    if peak in per_iso_sigma:
                        sig_prior = per_iso_sigma[peak]
                        if isinstance(sig_prior, (list, tuple)) and len(sig_prior) == 2:
                            priors_spec[f"sigma_{peak}"] = (float(sig_prior[0]), float(sig_prior[1]))

                # Low-energy shelf fraction priors
                use_shelf_cfg = spectral_cfg.get("use_shelf", {})
                for peak in adc_peaks.keys():
                    if isinstance(use_shelf_cfg, dict) and use_shelf_cfg.get(peak, False):
                        shelf_key = f"f_shelf_{peak}_prior"
                        shelf_prior = spectral_cfg.get(shelf_key)
                        if shelf_prior is not None and isinstance(shelf_prior, (list, tuple)):
                            priors_spec[f"f_shelf_{peak}"] = (float(shelf_prior[0]), float(shelf_prior[1]))
                        else:
                            # Default shelf prior: small fraction
                            priors_spec[f"f_shelf_{peak}"] = (0.02, 0.05)
                        # Separate shelf width prior (optional)
                        shelf_sigma_key = f"sigma_shelf_{peak}_prior"
                        shelf_sigma_prior = spectral_cfg.get(shelf_sigma_key)
                        if shelf_sigma_prior is not None and isinstance(shelf_sigma_prior, (list, tuple)):
                            priors_spec[f"sigma_shelf_{peak}"] = (float(shelf_sigma_prior[0]), float(shelf_sigma_prior[1]))

                # Halo (broad component) fraction and width priors
                use_halo_cfg = spectral_cfg.get("use_halo", {})
                for peak in adc_peaks.keys():
                    if isinstance(use_halo_cfg, dict) and use_halo_cfg.get(peak, False):
                        halo_key = f"f_halo_{peak}_prior"
                        halo_prior = spectral_cfg.get(halo_key)
                        if halo_prior is not None and isinstance(halo_prior, (list, tuple)):
                            priors_spec[f"f_halo_{peak}"] = (float(halo_prior[0]), float(halo_prior[1]))
                        else:
                            # Default halo prior: moderate fraction
                            priors_spec[f"f_halo_{peak}"] = (0.10, 0.10)
                        # Halo width prior
                        halo_sigma_key = f"sigma_halo_{peak}_prior"
                        halo_sigma_prior = spectral_cfg.get(halo_sigma_key)
                        if halo_sigma_prior is not None and isinstance(halo_sigma_prior, (list, tuple)):
                            priors_spec[f"sigma_halo_{peak}"] = (float(halo_sigma_prior[0]), float(halo_sigma_prior[1]))
                        else:
                            # Default: 2x the peak sigma
                            peak_sig = per_iso_sigma.get(peak)
                            if peak_sig is not None and isinstance(peak_sig, (list, tuple)):
                                priors_spec[f"sigma_halo_{peak}"] = (float(peak_sig[0]) * 2.0, float(peak_sig[1]) * 2.0)
                            else:
                                priors_spec[f"sigma_halo_{peak}"] = (0.25, 0.10)
                        # Halo EMG tail prior (decoupled from core tau)
                        halo_tau_key = f"tau_halo_{peak}_prior"
                        halo_tau_prior = spectral_cfg.get(halo_tau_key)
                        if halo_tau_prior is not None and isinstance(halo_tau_prior, (list, tuple)):
                            priors_spec[f"tau_halo_{peak}"] = (float(halo_tau_prior[0]), float(halo_tau_prior[1]))

                # Continuum priors (skip entirely for "none" background model)
                _early_bkg_model = str(analysis_cfg.get("background_model", "")).lower()
                bkg_mode = str(spectral_cfg.get("bkg_mode", "manual")).lower()
                if _early_bkg_model == "none":
                    logger.info("No-background model: skipping continuum (b0/b1) priors")
                elif bkg_mode == "auto":
                    from background import estimate_linear_background

                    mu_map = {k: priors_spec[f"mu_{k}"][0] for k in adc_peaks.keys()}
                    b0_est, b1_est = estimate_linear_background(
                        E_all,
                        mu_map,
                        peak_width=peak_tol,
                    )
                    priors_spec["b0"] = (b0_est, abs(b0_est) * 0.1 + 1e-3)
                    priors_spec["b1"] = (b1_est, abs(b1_est) * 0.1 + 1e-3)
                elif bkg_mode.startswith("auto_poly"):
                    from background import estimate_polynomial_background_auto

                    mu_map = {k: priors_spec[f"mu_{k}"][0] for k in adc_peaks.keys()}
                    try:
                        max_n = int(bkg_mode.split("auto_poly")[-1])
                    except ValueError:
                        max_n = 2
                    coeffs, order = estimate_polynomial_background_auto(
                        E_all,
                        mu_map,
                        max_order=max_n,
                        peak_width=peak_tol,
                    )
                    for i, c in enumerate(coeffs):
                        priors_spec[f"b{i}"] = (float(c), abs(float(c)) * 0.1 + 1e-3)
                    priors_spec["poly_order"] = order
                else:
                    priors_spec["b0"] = tuple(spectral_cfg.get("b0_prior"))
                    priors_spec["b1"] = tuple(spectral_cfg.get("b1_prior"))

                # Log-quadratic background curvature term (optional)
                b2_prior = spectral_cfg.get("b2_prior")
                if b2_prior is not None and isinstance(b2_prior, (list, tuple)):
                    priors_spec["b2"] = (float(b2_prior[0]), float(b2_prior[1]))

                # Log-cubic background term (optional)
                b3_prior = spectral_cfg.get("b3_prior")
                if b3_prior is not None and isinstance(b3_prior, (list, tuple)):
                    priors_spec["b3"] = (float(b3_prior[0]), float(b3_prior[1]))

                # Flags controlling the spectral fit
                spec_flags = spectral_cfg.get("flags", {}).copy()
                spec_flags.setdefault("cfg", cfg)
                bkg_model = analysis_cfg.get("background_model")
                if bkg_model is not None:
                    spec_flags["background_model"] = bkg_model
                like_model = analysis_cfg.get("likelihood")
                if like_model is not None:
                    spec_flags["likelihood"] = like_model
                if float_sigma_E and spec_flags.get("fix_sigma0"):
                    raise ValueError(
                        "Configuration error: cannot float energy resolution while fixing sigma0"
                    )
                if not float_sigma_E:
                    spec_flags["fix_sigma0"] = True
                    spec_flags.setdefault("fix_F", True)

                if "fix_sigma_E" in spec_flags:
                    if spec_flags.pop("fix_sigma_E"):
                        spec_flags.setdefault("fix_sigma0", True)
                        spec_flags.setdefault("fix_F", True)

                if spec_flags.get("fix_sigma0") and not spec_flags.get("fix_F", True):
                    raise ValueError(
                        "Configuration error: fix_sigma0 requires fix_F when energy resolution is fixed"
                    )

                use_emg_cfg = spectral_cfg.get("use_emg")
                if use_emg_cfg is not None:
                    spec_flags["use_emg"] = dict(use_emg_cfg)

                # Pass configurable halo/shelf bound parameters to the fitter
                for _bound_key in ("max_f_halo", "max_f_shelf", "sigma_halo_min_mult", "sigma_halo_max_mult", "tau_halo_max_mult", "sigma_shelf_min"):
                    _bound_val = spectral_cfg.get(_bound_key)
                    if _bound_val is not None:
                        spec_flags[_bound_key] = float(_bound_val)

                _bkg_model_name = spec_flags.get("background_model", "")
                if _bkg_model_name in ("loglin_unit", "sigmoid_unit", "exp_unit", "double_logit_unit"):
                    priors_spec["S_bkg"] = _estimate_loglin_background_prior(
                        E_all,
                        {k: priors_spec[f"mu_{k}"][0] for k in adc_peaks.keys()},
                        peak_width=peak_tol,
                        prior_hint=spectral_cfg.get("S_bkg_prior", spectral_cfg.get("s_bkg_prior")),
                    )
                elif _bkg_model_name == "none":
                    # No polynomial background — halos + shelves do all the work.
                    # Optional S_bkg for flat microphonics / residual noise floor.
                    _sbkg_hint = spectral_cfg.get("S_bkg_prior", spectral_cfg.get("s_bkg_prior"))
                    if _sbkg_hint is not None:
                        priors_spec["S_bkg"] = tuple(_sbkg_hint)
                    else:
                        # Auto-estimate: whatever isn't under peaks might be microphonics
                        _total_signal = sum(
                            priors_spec.get(f"S_{k}", (0,))[0] for k in adc_peaks.keys()
                        )
                        _remaining = max(float(len(E_all)) - _total_signal, 0.0)
                        priors_spec["S_bkg"] = (max(_remaining, 1.0), max(_remaining * 2.0, 100.0))
                    # Remove b0/b1/b2/b3 — not used by "none" model
                    priors_spec.pop("b0", None)
                    priors_spec.pop("b1", None)
                    priors_spec.pop("b2", None)
                    priors_spec.pop("b3", None)
                    logger.info(
                        "No-background model: S_bkg (microphonics) prior=%.1f±%.1f",
                        priors_spec["S_bkg"][0], priors_spec["S_bkg"][1],
                    )

                # For sigmoid/exp background, override b0/b1 priors with
                # physically meaningful values and remove b2/b3 (unused).
                if _bkg_model_name == "sigmoid_unit":
                    # b0 → E_half: sigmoid midpoint (center of energy range)
                    _E_mid = 0.5 * (E_all.min() + E_all.max())
                    _E_span = E_all.max() - E_all.min()
                    priors_spec["b0"] = (float(_E_mid), float(_E_span * 0.3))
                    # b1 → slope: positive = high on left, low on right
                    priors_spec["b1"] = (1.0, 2.0)
                    # Remove b2/b3 — not used by sigmoid
                    priors_spec.pop("b2", None)
                    priors_spec.pop("b3", None)
                    logger.info(
                        "Sigmoid background: E_half prior=%.2f±%.2f, slope prior=1.0±2.0",
                        float(_E_mid), float(_E_span * 0.3),
                    )
                elif _bkg_model_name == "exp_unit":
                    # b0 → unused (set to 0, fixed)
                    priors_spec["b0"] = (0.0, 0.01)
                    # b1 → alpha: decay rate (MeV^-1)
                    priors_spec["b1"] = (0.5, 1.0)
                    priors_spec.pop("b2", None)
                    priors_spec.pop("b3", None)
                    spec_flags["fix_b0"] = True
                    logger.info("Exponential background: alpha prior=0.5±1.0")
                elif _bkg_model_name == "double_logit_unit":
                    # Double-logit: two sigmoid transitions
                    # b0 → E_half1 (low-energy sigmoid midpoint)
                    # b1 → slope1 (low-energy sigmoid steepness)
                    # b2 → E_half2 (high-energy sigmoid midpoint)
                    # b3 → slope2 (high-energy sigmoid steepness)
                    _E_lo_fit = float(spectral_cfg.get("fit_energy_range", [1.3, 17.4])[0])
                    _E_hi_fit = float(spectral_cfg.get("fit_energy_range", [1.3, 17.4])[1])
                    # Low-energy transition: around 3-5 MeV (microphonics → alpha region)
                    _dl_b0 = spectral_cfg.get("b0_prior", [3.5, 2.0])
                    priors_spec["b0"] = (float(_dl_b0[0]), float(_dl_b0[1]))
                    # Low-energy slope: positive = high on left
                    _dl_b1 = spectral_cfg.get("b1_prior", [2.0, 3.0])
                    priors_spec["b1"] = (float(_dl_b1[0]), float(_dl_b1[1]))
                    # High-energy transition: around 9-12 MeV (above last alpha peak)
                    _dl_b2 = spectral_cfg.get("b2_prior", [10.0, 3.0])
                    priors_spec["b2"] = (float(_dl_b2[0]), float(_dl_b2[1]))
                    # High-energy slope: positive = drops off at high energy
                    _dl_b3 = spectral_cfg.get("b3_prior", [1.0, 3.0])
                    priors_spec["b3"] = (float(_dl_b3[0]), float(_dl_b3[1]))
                    logger.info(
                        "Double-logit background: E_half1=%.1f±%.1f, slope1=%.1f±%.1f, "
                        "E_half2=%.1f±%.1f, slope2=%.1f±%.1f",
                        priors_spec["b0"][0], priors_spec["b0"][1],
                        priors_spec["b1"][0], priors_spec["b1"][1],
                        priors_spec["b2"][0], priors_spec["b2"][1],
                        priors_spec["b3"][0], priors_spec["b3"][1],
                    )

                # ── Extra peaks (unknown signals, pile-up, etc.) ─────────
                _extra_peaks = spectral_cfg.get("extra_peaks", {})
                if _extra_peaks and isinstance(_extra_peaks, dict):
                    for _ep_name, _ep_cfg in _extra_peaks.items():
                        _ep_energy = float(_ep_cfg.get("energy", 0.0))
                        if _ep_energy <= 0.0:
                            continue
                        _ep_sigma = _ep_cfg.get("sigma", [0.5, 0.3])
                        if isinstance(_ep_sigma, (list, tuple)):
                            _ep_sig_mu, _ep_sig_sig = float(_ep_sigma[0]), float(_ep_sigma[1])
                        else:
                            _ep_sig_mu, _ep_sig_sig = float(_ep_sigma), 0.3
                        _ep_amp = _ep_cfg.get("amplitude", [500, 1000])
                        if isinstance(_ep_amp, (list, tuple)):
                            _ep_amp_mu, _ep_amp_sig = float(_ep_amp[0]), float(_ep_amp[1])
                        else:
                            _ep_amp_mu, _ep_amp_sig = float(_ep_amp), float(_ep_amp) * 2.0
                        _ep_mu_sigma = float(_ep_cfg.get("mu_sigma", 0.3))

                        priors_spec[f"mu_{_ep_name}"] = (_ep_energy, _ep_mu_sigma)
                        priors_spec[f"S_{_ep_name}"] = (_ep_amp_mu, _ep_amp_sig)
                        priors_spec[f"sigma_{_ep_name}"] = (_ep_sig_mu, _ep_sig_sig)

                        # Optional mu bounds
                        _ep_mu_bounds = _ep_cfg.get("mu_bounds")
                        if _ep_mu_bounds is not None and isinstance(_ep_mu_bounds, (list, tuple)):
                            mu_bounds_fit[_ep_name] = (float(_ep_mu_bounds[0]), float(_ep_mu_bounds[1]))

                        # ── EMG tail for extra peak ──────────────────────
                        _ep_use_emg = _ep_cfg.get("use_emg", False)
                        if _ep_use_emg:
                            # Inject into use_emg config dict so fitting.py sees it
                            if "use_emg" not in spec_flags:
                                spec_flags["use_emg"] = {}
                            spec_flags["use_emg"][_ep_name] = True
                            # tau prior for the extra peak
                            _ep_tau = _ep_cfg.get("tau", [0.05, 0.05])
                            if isinstance(_ep_tau, (list, tuple)):
                                priors_spec[f"tau_{_ep_name}"] = (float(_ep_tau[0]), float(_ep_tau[1]))
                            else:
                                priors_spec[f"tau_{_ep_name}"] = (float(_ep_tau), float(_ep_tau) * 0.5)

                        # ── Shelf for extra peak ─────────────────────────
                        _ep_use_shelf = _ep_cfg.get("use_shelf", False)
                        if _ep_use_shelf:
                            # Inject into use_shelf config dict
                            if isinstance(use_shelf_cfg, dict):
                                use_shelf_cfg[_ep_name] = True
                            _ep_f_shelf = _ep_cfg.get("f_shelf", [0.05, 0.10])
                            if isinstance(_ep_f_shelf, (list, tuple)):
                                priors_spec[f"f_shelf_{_ep_name}"] = (float(_ep_f_shelf[0]), float(_ep_f_shelf[1]))
                            else:
                                priors_spec[f"f_shelf_{_ep_name}"] = (float(_ep_f_shelf), 0.10)
                            _ep_sigma_shelf = _ep_cfg.get("sigma_shelf", [0.3, 0.15])
                            if isinstance(_ep_sigma_shelf, (list, tuple)):
                                priors_spec[f"sigma_shelf_{_ep_name}"] = (float(_ep_sigma_shelf[0]), float(_ep_sigma_shelf[1]))
                            else:
                                priors_spec[f"sigma_shelf_{_ep_name}"] = (float(_ep_sigma_shelf), 0.15)

                        # ── Halo for extra peak ──────────────────────────
                        _ep_use_halo = _ep_cfg.get("use_halo", False)
                        if _ep_use_halo:
                            # Inject into use_halo config dict
                            if isinstance(use_halo_cfg, dict):
                                use_halo_cfg[_ep_name] = True
                            _ep_f_halo = _ep_cfg.get("f_halo", [0.08, 0.10])
                            if isinstance(_ep_f_halo, (list, tuple)):
                                priors_spec[f"f_halo_{_ep_name}"] = (float(_ep_f_halo[0]), float(_ep_f_halo[1]))
                            else:
                                priors_spec[f"f_halo_{_ep_name}"] = (float(_ep_f_halo), 0.10)
                            _ep_sigma_halo = _ep_cfg.get("sigma_halo", [0.5, 0.25])
                            if isinstance(_ep_sigma_halo, (list, tuple)):
                                priors_spec[f"sigma_halo_{_ep_name}"] = (float(_ep_sigma_halo[0]), float(_ep_sigma_halo[1]))
                            else:
                                priors_spec[f"sigma_halo_{_ep_name}"] = (float(_ep_sigma_halo), 0.25)
                            _ep_tau_halo = _ep_cfg.get("tau_halo", [0.03, 0.05])
                            if isinstance(_ep_tau_halo, (list, tuple)):
                                priors_spec[f"tau_halo_{_ep_name}"] = (float(_ep_tau_halo[0]), float(_ep_tau_halo[1]))
                            else:
                                priors_spec[f"tau_halo_{_ep_name}"] = (float(_ep_tau_halo), 0.05)

                        # Fix flags from config
                        _ep_fix = _ep_cfg.get("fix", {})
                        if isinstance(_ep_fix, dict):
                            for _fk, _fv in _ep_fix.items():
                                spec_flags[f"fix_{_fk}_{_ep_name}"] = bool(_fv)

                        logger.info(
                            "Extra peak '%s': E=%.2f±%.2f MeV, sigma=%.2f±%.2f, amp=%.0f±%.0f, "
                            "use_emg=%s, use_shelf=%s, use_halo=%s",
                            _ep_name, _ep_energy, _ep_mu_sigma,
                            _ep_sig_mu, _ep_sig_sig, _ep_amp_mu, _ep_amp_sig,
                            _ep_use_emg, _ep_use_shelf, _ep_use_halo,
                        )

                # Launch the spectral fit
                spec_fit_out = None
                peak_deviation = {}
                try:
                    fit_kwargs = {
                        "energies": E_all,
                        "priors": priors_spec,
                        "flags": spec_flags,
                    }
                    if spectral_cfg.get("use_plot_bins_for_fit", False):
                        fit_kwargs.update({"bins": bins, "bin_edges": bin_edges})
                    if spectral_cfg.get("unbinned_likelihood", False):
                        fit_kwargs["unbinned"] = True
                    if args.strict_covariance:
                        fit_kwargs["strict"] = True
                    if mu_bounds_fit:
                        bounds_map = {
                            f"mu_{iso}": tuple(bounds)
                            for iso, bounds in mu_bounds_fit.items()
                        }
                        if bounds_map:
                            fit_kwargs["bounds"] = bounds_map

                    # ── Two-stage DNL: Fourier at full res, then rebin ──
                    _dnl_cfg_main = spectral_cfg.get("dnl_correction", {})
                    _full_res_dnl = (
                        _dnl_cfg_main.get("full_resolution_estimate", False)
                        and _dnl_cfg_main.get("enabled", False)
                    )
                    _pre_hist = None
                    _pre_dnl_meta = None
                    if _full_res_dnl and method == "adc":
                        logger.info("Two-stage DNL pipeline: full-res Fourier + rebin")
                        # Extract timestamps for per-period crossval
                        _frd_timestamps = None
                        if "timestamp" in df_spectrum.columns:
                            try:
                                _frd_timestamps = (
                                    pd.to_datetime(
                                        df_spectrum["timestamp"], utc=True, errors="coerce"
                                    ).astype(np.int64).to_numpy(dtype=float)
                                )
                            except Exception:
                                _frd_timestamps = np.arange(len(df_spectrum), dtype=float)
                        else:
                            _frd_timestamps = np.arange(len(df_spectrum), dtype=float)
                        try:
                            _pre_hist, _pre_edges, _pre_dnl_meta, _pre_info = (
                                _preprocess_full_resolution_dnl(
                                    adc_all, E_all,
                                    cal_slope=a,
                                    cal_intercept=c,
                                    cal_a2=a2,
                                    cal_a3=a3,
                                    priors=priors_spec,
                                    flags=spec_flags,
                                    cfg=cfg,
                                    bounds=fit_kwargs.get("bounds"),
                                    timestamps=_frd_timestamps,
                                )
                            )
                            # Override bin_edges with the rebinned edges
                            bin_edges = _pre_edges
                            bins = bin_edges.size - 1
                            fit_kwargs["bin_edges"] = bin_edges
                            fit_kwargs["bins"] = bins
                            logger.info(
                                "Two-stage DNL: rebin_factor=%d, %d bins for fit",
                                _pre_info.get("rebin_factor", -1), bins,
                            )
                        except Exception as _frd_exc:
                            import traceback
                            logger.error(
                                "Two-stage DNL preprocessing failed: %s\n%s",
                                _frd_exc, traceback.format_exc(),
                            )
                            _pre_hist = None
                            _pre_dnl_meta = None
                            _pre_info = {}

                    # Store per-period crossval for summary output
                    _per_period_crossval_data = (
                        _pre_info.get("per_period_crossval") if _pre_dnl_meta else None
                    )

                    # ── Seed priors from prelim fit ────────────────────────
                    _use_prelim_bkg = spectral_cfg.get(
                        "seed_bkg_from_prelim", False
                    )
                    if _use_prelim_bkg and _pre_info:
                        _pbp = _pre_info.get("prelim_bkg_params", {})
                        if _pbp:
                            for _bk in ("b1", "b2", "b3"):
                                if _bk in _pbp and f"{_bk}" in priors_spec:
                                    _old = priors_spec[_bk]
                                    _new_mu = _pbp[_bk]
                                    # Use prelim value as mean, keep sigma
                                    # from config (allows some flexibility)
                                    priors_spec[_bk] = (_new_mu, _old[1])
                            if "S_bkg" in _pbp and "S_bkg" in priors_spec:
                                _old_s = priors_spec["S_bkg"]
                                _prelim_sbkg = _pbp["S_bkg"]
                                # Guard: if prelim S_bkg collapsed to ~0, keep
                                # the original prior (don't seed a bad value).
                                if _prelim_sbkg > 1.0:
                                    priors_spec["S_bkg"] = (
                                        _prelim_sbkg,
                                        max(_old_s[1], abs(_prelim_sbkg) * 0.5),
                                    )
                                else:
                                    logger.warning(
                                        "Prelim S_bkg=%.2f collapsed to ~0; "
                                        "keeping original prior for stage 2",
                                        _prelim_sbkg,
                                    )
                            logger.info(
                                "Seeded background priors from prelim fit: %s",
                                {k: f"{v:.4f}" for k, v in _pbp.items()},
                            )

                        # Also seed shape params (shelf, tau, sigma, etc.)
                        # from the prelim fit.  The full-resolution prelim fit
                        # has much better shape information; fix these at the
                        # prelim values so the rebinned fit doesn't diverge.
                        _prelim_fv = _pre_info.get("prelim_plot_data", {}).get("fit_vals", {})
                        _seed_shape_mode = str(spectral_cfg.get(
                            "seed_shape_from_prelim", "fix"
                        )).lower()  # "fix", "seed", or "off"
                        if _prelim_fv and _seed_shape_mode != "off":
                            _shape_prefixes = (
                                "tau_", "f_shelf_", "sigma_shelf_",
                                "f_halo_", "sigma_halo_", "tau_halo_",
                            )
                            _seeded_shape = {}
                            for _sk, _sv in _prelim_fv.items():
                                if any(_sk.startswith(pfx) for pfx in _shape_prefixes):
                                    if _sk in priors_spec:
                                        try:
                                            _sv_f = float(_sv)
                                            if np.isfinite(_sv_f):
                                                _old_mu, _old_sig = priors_spec[_sk]
                                                if _seed_shape_mode == "fix":
                                                    # Fix at prelim value with very tight prior
                                                    priors_spec[_sk] = (_sv_f, _old_sig * 0.1)
                                                    spec_flags[f"fix_{_sk}"] = True
                                                else:
                                                    # Seed: use prelim value as mean
                                                    priors_spec[_sk] = (_sv_f, _old_sig)
                                                _seeded_shape[_sk] = _sv_f
                                        except (TypeError, ValueError):
                                            pass
                            if _seeded_shape:
                                logger.info(
                                    "Seeded %d shape priors from prelim (%s mode): %s",
                                    len(_seeded_shape), _seed_shape_mode,
                                    ", ".join(f"{k}={v:.4f}" for k, v in sorted(_seeded_shape.items())),
                                )

                    spec_fit_out, peak_deviation = _spectral_fit_with_check(
                        E_all,
                        priors_spec,
                        spec_flags,
                        cfg,
                        bins=fit_kwargs.get("bins"),
                        bin_edges=fit_kwargs.get("bin_edges"),
                        bounds=fit_kwargs.get("bounds"),
                        unbinned=fit_kwargs.get("unbinned", False),
                        strict=fit_kwargs.get("strict", False),
                        pre_binned_hist=_pre_hist,
                        pre_dnl_meta=_pre_dnl_meta,
                    )
                    if isinstance(spec_fit_out, FitResult) and not spec_fit_out.params.get(
                        "fit_valid", True
                    ):
                        tau_keys = [k for k in priors_spec if k.startswith("tau_")]
                        if tau_keys:
                            priors_shrunk = priors_spec.copy()
                            for t in tau_keys:
                                mu, sig = priors_shrunk[t]
                                priors_shrunk[t] = (mu, sig * 0.5)
                            flags_fix = spec_flags.copy()
                            for t in tau_keys:
                                flags_fix[f"fix_{t}"] = True
                            refit = fit_spectrum(
                                E_all,
                                priors_shrunk,
                                flags=flags_fix,
                                bins=fit_kwargs.get("bins"),
                                bin_edges=fit_kwargs.get("bin_edges"),
                                bounds=fit_kwargs.get("bounds"),
                                unbinned=fit_kwargs.get("unbinned", False),
                                strict=fit_kwargs.get("strict", False),
                                skip_minos=cfg.get("spectral_fit", {}).get("skip_minos", False),
                            )
                            if isinstance(refit, FitResult) and refit.params.get(
                                "fit_valid", False
                            ):
                                thresh = spectral_cfg.get("refit_aic_threshold", 2.0)
                                if (
                                    refit.params.get("aic", float("inf"))
                                    > spec_fit_out.params.get("aic", float("inf")) - thresh
                                ):
                                    spec_fit_out = refit
                                else:
                                    free_fit = fit_spectrum(
                                        E_all,
                                        priors_shrunk,
                                        flags=spec_flags,
                                        bins=fit_kwargs.get("bins"),
                                        bin_edges=fit_kwargs.get("bin_edges"),
                                        bounds=fit_kwargs.get("bounds"),
                                        unbinned=fit_kwargs.get("unbinned", False),
                                        strict=fit_kwargs.get("strict", False),
                                        skip_minos=cfg.get("spectral_fit", {}).get("skip_minos", False),
                                    )
                                    if (
                                        isinstance(free_fit, FitResult)
                                        and free_fit.params.get("fit_valid", False)
                                        and free_fit.params.get("aic", float("inf"))
                                        < refit.params.get("aic", float("inf")) - thresh
                                    ):
                                        spec_fit_out = free_fit
                                    else:
                                        spec_fit_out = refit
                    spectrum_results = spec_fit_out

                    # Save stage 2 (main fit) plot data for per-stage diagnostics
                    _stage2_plot_data = None
                    if isinstance(spec_fit_out, FitResult):
                        try:
                            _s2_fv = dict(spec_fit_out.params)
                            _stage2_plot_data = {
                                "fit_vals": _s2_fv,
                                "bins": fit_kwargs.get("bins"),
                                "bin_edges": fit_kwargs.get("bin_edges"),
                                "flags": dict(spec_flags),
                            }
                        except Exception as _s2ex:
                            logger.debug("Could not assemble stage 2 plot data: %s", _s2ex)

                    # ── Three-stage binning: stage 3 refit at coarser bins ──
                    _three_stage_cfg = spectral_cfg.get("three_stage_binning", {})
                    _three_stage_enabled = (
                        _three_stage_cfg.get("enabled", False)
                        and _full_res_dnl
                        and _pre_info is not None
                        and isinstance(spectrum_results, FitResult)
                    )
                    if _three_stage_enabled:
                        try:
                            _s2_params = spectrum_results.params
                            _s3_rebin = int(_three_stage_cfg.get("stage3_rebin", 15))
                            _corrected_full = _pre_info.get("corrected_full_res")
                            _edges_adc_full = _pre_info.get("edges_adc_full")

                            if _corrected_full is None or _edges_adc_full is None:
                                logger.warning(
                                    "Three-stage: full-res corrected data not available, skipping stage 3"
                                )
                            else:
                                logger.info(
                                    "Three-stage binning: stage 2 chi2/ndf=%.3f at %d bins, "
                                    "now rebinning to factor=%d for stage 3",
                                    _s2_params.get("chi2_ndf", float("nan")),
                                    _s2_params.get("n_bins", -1),
                                    _s3_rebin,
                                )

                                # Rebin DNL-corrected full-res histogram to stage 3
                                _s3_hist, _s3_edges_adc = rebin_histogram(
                                    _corrected_full, _edges_adc_full, _s3_rebin
                                )
                                _s3_edges_mev = apply_calibration(
                                    _s3_edges_adc, a, c,
                                    quadratic_coeff=a2, cubic_coeff=a3,
                                )
                                logger.info(
                                    "Stage 3: %d bins (rebin factor %d from %d full-res)",
                                    _s3_hist.size, _s3_rebin, len(_corrected_full),
                                )

                                # Extract stage 2 shape param values and fix them
                                _s3_priors = dict(priors_spec)
                                _s3_flags = dict(spec_flags)
                                _shape_prefixes = [
                                    "tau_", "f_shelf_", "sigma_shelf_",
                                    "f_halo_", "sigma_halo_", "tau_halo_",
                                    "sigma_",
                                ]
                                _isotopes = ["Po210", "Po218", "Po214", "Po216", "Po212"]
                                # Include extra peaks in the seed list
                                _extra_peaks_cfg = spectral_cfg.get("extra_peaks", {})
                                if _extra_peaks_cfg and isinstance(_extra_peaks_cfg, dict):
                                    _isotopes = _isotopes + list(_extra_peaks_cfg.keys())
                                _s3_shape_seeds = {}
                                for _iso in _isotopes:
                                    for _pfx in _shape_prefixes:
                                        _pkey = f"{_pfx}{_iso}"
                                        _s2_val = _s2_params.get(_pkey)
                                        if _s2_val is not None and np.isfinite(float(_s2_val)):
                                            # Update prior to stage 2 best-fit value with tight sigma
                                            _s2_val_f = float(_s2_val)
                                            if _pkey in _s3_priors:
                                                _old_mu, _old_sig = _s3_priors[_pkey]
                                                _s3_priors[_pkey] = (_s2_val_f, _old_sig * 0.1)
                                            # Fix the shape parameter
                                            _fix_key = f"fix_{_pkey}"
                                            _s3_flags[_fix_key] = True
                                            _s3_shape_seeds[_pkey] = _s2_val_f

                                logger.info(
                                    "Stage 3: fixed %d shape params from stage 2: %s",
                                    len(_s3_shape_seeds),
                                    ", ".join(f"{k}={v:.4f}" for k, v in sorted(_s3_shape_seeds.items())),
                                )

                                # Build stage 3 DNL metadata (same DNL, different binning)
                                _s3_dnl_meta = dict(_pre_dnl_meta) if _pre_dnl_meta else {}
                                _s3_dnl_meta["stage3_rebin_factor"] = _s3_rebin
                                _s3_dnl_meta["rebinned_bins"] = int(_s3_hist.size)
                                _s3_dnl_meta["statistical_model"] = "fourier_corrected_rebinned_poisson"

                                # Run stage 3 fit
                                _s3_result, _s3_peak_dev = _spectral_fit_with_check(
                                    E_all,
                                    _s3_priors,
                                    _s3_flags,
                                    cfg,
                                    bins=_s3_hist.size,
                                    bin_edges=_s3_edges_mev,
                                    bounds=fit_kwargs.get("bounds"),
                                    unbinned=False,
                                    strict=False,
                                    pre_binned_hist=_s3_hist,
                                    pre_dnl_meta=_s3_dnl_meta,
                                )

                                if isinstance(_s3_result, FitResult):
                                    _s3_p = _s3_result.params
                                    logger.info(
                                        "Three-stage complete: stage2 chi2/ndf=%.3f (%d bins), "
                                        "stage3 chi2/ndf=%.3f (%d bins)",
                                        _s2_params.get("chi2_ndf", float("nan")),
                                        _s2_params.get("n_bins", -1),
                                        _s3_p.get("chi2_ndf", float("nan")),
                                        _s3_p.get("n_bins", -1),
                                    )
                                    # Store stage 2 metadata inside stage 3 result
                                    _s3_p["_three_stage"] = {
                                        "stage2_chi2_ndf": _s2_params.get("chi2_ndf"),
                                        "stage2_n_bins": _s2_params.get("n_bins"),
                                        "stage2_rebin": _pre_info.get("rebin_factor"),
                                        "stage3_rebin": _s3_rebin,
                                        "stage2_shape_seeds": _s3_shape_seeds,
                                    }
                                    # Stage 3 becomes the primary result
                                    spectrum_results = _s3_result
                                else:
                                    logger.warning(
                                        "Stage 3 fit did not return FitResult; "
                                        "keeping stage 2 result as primary"
                                    )
                        except Exception as _s3_exc:
                            import traceback
                            logger.error(
                                "Three-stage binning stage 3 failed: %s\n%s",
                                _s3_exc, traceback.format_exc(),
                            )

                except Exception as e:
                    logger.warning("Spectral fit failed -> %s", e)
                    spectrum_results = {}

                # ── Split-half overfitting validation ──────────────────
                split_half_result = None
                if (
                    spectral_cfg.get("split_half_validation", False)
                    and isinstance(spectrum_results, FitResult)
                ):
                    try:
                        # Odd/even event split: statistically independent samples
                        # from the same underlying spectrum, free of temporal drift.
                        _idx = np.arange(len(df_spectrum))
                        _mask_a = (_idx % 2 == 0)
                        _mask_b = ~_mask_a
                        _E_a = df_spectrum.iloc[_mask_a]["energy_MeV"].to_numpy(dtype=float)
                        _E_b = df_spectrum.iloc[_mask_b]["energy_MeV"].to_numpy(dtype=float)
                        logger.info(
                            "Split-half validation: %d + %d events (odd/even split)",
                            _E_a.size, _E_b.size,
                        )
                        _sh_kwargs = {
                            "priors": priors_spec,
                            "flags": spec_flags,
                            "bins": fit_kwargs.get("bins"),
                            "bin_edges": fit_kwargs.get("bin_edges"),
                            "bounds": fit_kwargs.get("bounds"),
                            "unbinned": fit_kwargs.get("unbinned", False),
                            "strict": False,
                        }
                        _fit_a = fit_spectrum(_E_a, **_sh_kwargs)
                        _fit_b = fit_spectrum(_E_b, **_sh_kwargs)
                        _pa = _fit_a.params if isinstance(_fit_a, FitResult) else {}
                        _pb = _fit_b.params if isinstance(_fit_b, FitResult) else {}
                        # Compute z-scores for free parameters.
                        # Categorise: amplitude (S_*) and position (mu_*) params
                        # are expected to differ between time halves (activity and
                        # calibration drift).  Shape params (sigma, tau, f_*, b*)
                        # should be stable  - these are the overfitting indicators.
                        _sh_params = []
                        _skip = {
                            "_plot_", "_dnl", "fit_valid", "likelihood_path",
                            "aic", "nll", "chi2", "ndf", "n_free", "cov_",
                        }
                        for _k in sorted(_pa.keys()):
                            if _k.startswith("d") or _k.startswith("_") or _k.startswith("F"):
                                continue
                            if any(_k.startswith(s) for s in _skip):
                                continue
                            _va = _pa.get(_k)
                            _vb = _pb.get(_k)
                            _ea = _pa.get("d" + _k, 0.0)
                            _eb = _pb.get("d" + _k, 0.0)
                            if not all(isinstance(x, (int, float)) for x in [_va, _vb, _ea, _eb]):
                                continue
                            if _ea <= 0 and _eb <= 0:
                                continue  # both fixed
                            _denom = np.sqrt(_ea**2 + _eb**2) if (_ea**2 + _eb**2) > 0 else 1.0
                            _z = (_va - _vb) / _denom
                            if _k.startswith("S_") or _k.startswith("s_"):
                                _cat = "amplitude"
                            elif _k.startswith("mu_"):
                                _cat = "position"
                            else:
                                _cat = "shape"
                            _sh_params.append({
                                "name": _k,
                                "category": _cat,
                                "value_A": round(float(_va), 8),
                                "value_B": round(float(_vb), 8),
                                "error_A": round(float(_ea), 8),
                                "error_B": round(float(_eb), 8),
                                "z_score": round(float(_z), 4),
                            })
                        _z_all = [abs(p["z_score"]) for p in _sh_params]
                        _z_shape = [
                            abs(p["z_score"]) for p in _sh_params
                            if p["category"] == "shape"
                        ]
                        split_half_result = {
                            "n_events_A": int(_E_a.size),
                            "n_events_B": int(_E_b.size),
                            "chi2_ndf_A": round(float(_pa.get("chi2_ndf", 0)), 4),
                            "chi2_ndf_B": round(float(_pb.get("chi2_ndf", 0)), 4),
                            "parameters": _sh_params,
                            "max_z_all": round(max(_z_all) if _z_all else 0.0, 4),
                            "max_z_shape": round(
                                max(_z_shape) if _z_shape else 0.0, 4
                            ),
                            "mean_abs_z_shape": round(
                                float(np.mean(_z_shape)) if _z_shape else 0.0, 4
                            ),
                            "n_shape_z_gt_2": sum(1 for z in _z_shape if z > 2.0),
                            "n_shape_z_gt_3": sum(1 for z in _z_shape if z > 3.0),
                            "pass": all(z < 3.0 for z in _z_shape),
                        }
                        logger.info(
                            "Split-half validation (shape params): "
                            "max|z|=%.2f, mean|z|=%.2f, |z|>2: %d, |z|>3: %d -> %s",
                            split_half_result["max_z_shape"],
                            split_half_result["mean_abs_z_shape"],
                            split_half_result["n_shape_z_gt_2"],
                            split_half_result["n_shape_z_gt_3"],
                            "PASS" if split_half_result["pass"] else "WARN",
                        )
                    except Exception as e:
                        logger.warning("Split-half validation failed: %s", e)

                # ── Model complexity comparison (AIC/BIC scan) ─────────
                model_comparison_result = None
                if (
                    spectral_cfg.get("split_half_validation", False)
                    and isinstance(spectrum_results, FitResult)
                ):
                    try:
                        _base_params = spectrum_results.params
                        _base_nll = float(_base_params.get("nll", 0.0))
                        _base_aic = float(_base_params.get("aic", 0.0))
                        _base_bic = float(_base_params.get("bic", 0.0))
                        _base_aicc = float(_base_params.get("aicc", 0.0))
                        _base_chi2 = float(_base_params.get("chi2", 0.0))
                        _base_ndf = int(_base_params.get("ndf", 0))
                        _base_nfree = int(_base_params.get("n_free_params", 0))

                        # Identify which shape parameters are currently FREE
                        # (i.e., not fixed by a fix_* flag).
                        _candidate_fixes = [
                            ("tau_Po210", "fix_tau_Po210",
                             "Fix EMG tail (Po210)"),
                            ("f_halo_Po210", "fix_f_halo_Po210",
                             "Fix halo fraction (Po210)"),
                            ("f_shelf_Po214", "fix_f_shelf_Po214",
                             "Fix shelf fraction (Po214)"),
                            ("f_halo_Po214", "fix_f_halo_Po214",
                             "Fix halo fraction (Po214)"),
                        ]
                        _reduced_models = []
                        for _param, _flag, _desc in _candidate_fixes:
                            if spec_flags.get(_flag, False):
                                continue  # already fixed
                            _flags_red = spec_flags.copy()
                            _flags_red[_flag] = True
                            try:
                                _red_result = fit_spectrum(
                                    E_all,
                                    priors_spec,
                                    flags=_flags_red,
                                    bins=fit_kwargs.get("bins"),
                                    bin_edges=fit_kwargs.get("bin_edges"),
                                    bounds=fit_kwargs.get("bounds"),
                                    unbinned=fit_kwargs.get("unbinned", False),
                                    strict=False,
                                )
                                _rp = (
                                    _red_result.params
                                    if isinstance(_red_result, FitResult)
                                    else _red_result
                                )
                                _reduced_models.append({
                                    "description": _desc,
                                    "fixed_param": _param,
                                    "n_free": int(_rp.get("n_free_params", 0)),
                                    "nll": float(_rp.get("nll", 0.0)),
                                    "aic": float(_rp.get("aic", 0.0)),
                                    "bic": float(_rp.get("bic", 0.0)),
                                    "aicc": float(_rp.get("aicc", 0.0)),
                                    "chi2": float(_rp.get("chi2", 0.0)),
                                    "chi2_ndf": float(_rp.get("chi2_ndf", 0.0)),
                                    "delta_aic": float(_rp.get("aic", 0.0)) - _base_aic,
                                    "delta_bic": float(_rp.get("bic", 0.0)) - _base_bic,
                                    "delta_chi2": (
                                        float(_rp.get("chi2", 0.0)) - _base_chi2
                                    ),
                                })
                            except Exception as _re:
                                logger.warning(
                                    "Model comparison: %s failed: %s",
                                    _desc, _re,
                                )

                        # Also test minimal model: fix ALL shape params
                        # except sigma (mu + sigma + S per isotope + b1 + S_bkg)
                        _flags_min = spec_flags.copy()
                        for _p, _f, _ in _candidate_fixes:
                            _flags_min[_f] = True
                        try:
                            _min_result = fit_spectrum(
                                E_all,
                                priors_spec,
                                flags=_flags_min,
                                bins=fit_kwargs.get("bins"),
                                bin_edges=fit_kwargs.get("bin_edges"),
                                bounds=fit_kwargs.get("bounds"),
                                unbinned=fit_kwargs.get("unbinned", False),
                                strict=False,
                            )
                            _mp = (
                                _min_result.params
                                if isinstance(_min_result, FitResult)
                                else _min_result
                            )
                            _reduced_models.append({
                                "description": "Minimal (fix all shape params)",
                                "fixed_param": "all_shape",
                                "n_free": int(_mp.get("n_free_params", 0)),
                                "nll": float(_mp.get("nll", 0.0)),
                                "aic": float(_mp.get("aic", 0.0)),
                                "bic": float(_mp.get("bic", 0.0)),
                                "aicc": float(_mp.get("aicc", 0.0)),
                                "chi2": float(_mp.get("chi2", 0.0)),
                                "chi2_ndf": float(_mp.get("chi2_ndf", 0.0)),
                                "delta_aic": float(_mp.get("aic", 0.0)) - _base_aic,
                                "delta_bic": float(_mp.get("bic", 0.0)) - _base_bic,
                                "delta_chi2": (
                                    float(_mp.get("chi2", 0.0)) - _base_chi2
                                ),
                            })
                        except Exception as _me:
                            logger.warning(
                                "Model comparison: minimal failed: %s", _me,
                            )

                        model_comparison_result = {
                            "base_model": {
                                "description": "Full model (current)",
                                "n_free": _base_nfree,
                                "nll": _base_nll,
                                "aic": _base_aic,
                                "bic": _base_bic,
                                "aicc": _base_aicc,
                                "chi2": _base_chi2,
                                "chi2_ndf": round(
                                    _base_chi2 / _base_ndf if _base_ndf else 0, 4
                                ),
                            },
                            "reduced_models": _reduced_models,
                            "conclusion": "full_model_preferred",
                        }
                        # If ANY reduced model has lower BIC, the full
                        # model may be over-parameterised.
                        for _rm in _reduced_models:
                            if _rm["delta_bic"] < -2.0:
                                model_comparison_result["conclusion"] = (
                                    "simpler_model_preferred"
                                )
                                break

                        logger.info(
                            "Model complexity comparison: %d reduced models, "
                            "conclusion=%s",
                            len(_reduced_models),
                            model_comparison_result["conclusion"],
                        )
                        for _rm in _reduced_models:
                            logger.info(
                                "  %-40s k=%d  ΔAIC=%+.1f  ΔBIC=%+.1f  "
                                "Δχ²=%+.1f  χ²/NDF=%.3f",
                                _rm["description"], _rm["n_free"],
                                _rm["delta_aic"], _rm["delta_bic"],
                                _rm["delta_chi2"], _rm["chi2_ndf"],
                            )
                    except Exception as e:
                        logger.warning("Model complexity comparison failed: %s", e)

                # ── D2: Shelf/halo held-out cross-validation ──────────
                shelf_halo_result = None
                if (
                    spectral_cfg.get("shelf_halo_test", False)
                    and isinstance(spectrum_results, FitResult)
                ):
                    try:
                        logger.info("Running shelf/halo held-out test...")
                        _sh_idx = np.arange(len(df_spectrum))
                        _sh_even = (_sh_idx % 2 == 0)
                        _sh_odd = ~_sh_even
                        E_even = E_all[_sh_even]
                        E_odd = E_all[_sh_odd]

                        # Full model on even events
                        _sh_full_even = fit_spectrum(
                            E_even, priors_spec, flags=spec_flags,
                            bins=fit_kwargs.get("bins"),
                            bin_edges=fit_kwargs.get("bin_edges"),
                            bounds=fit_kwargs.get("bounds"),
                            unbinned=fit_kwargs.get("unbinned", False),
                            strict=False,
                        )

                        # Reduced model: fix shelf and halo fractions to 0
                        _sh_flags_reduced = dict(spec_flags)
                        for _iso in ["Po210", "Po218", "Po214", "Po212"]:
                            _sh_flags_reduced[f"fix_f_shelf_{_iso}"] = True
                            _sh_flags_reduced[f"fix_f_halo_{_iso}"] = True
                        _sh_priors_reduced = dict(priors_spec)
                        for _iso in ["Po210", "Po218", "Po214", "Po212"]:
                            _sh_priors_reduced[f"f_shelf_{_iso}"] = (0.0, 0.001)
                            _sh_priors_reduced[f"f_halo_{_iso}"] = (0.0, 0.001)

                        _sh_red_even = fit_spectrum(
                            E_even, _sh_priors_reduced,
                            flags=_sh_flags_reduced,
                            bins=fit_kwargs.get("bins"),
                            bin_edges=fit_kwargs.get("bin_edges"),
                            bounds=fit_kwargs.get("bounds"),
                            unbinned=fit_kwargs.get("unbinned", False),
                            strict=False,
                        )

                        # Evaluate on held-out odd events
                        _sh_full_odd = fit_spectrum(
                            E_odd, priors_spec, flags=spec_flags,
                            bins=fit_kwargs.get("bins"),
                            bin_edges=fit_kwargs.get("bin_edges"),
                            bounds=fit_kwargs.get("bounds"),
                            unbinned=fit_kwargs.get("unbinned", False),
                            strict=False,
                        )
                        _sh_red_odd = fit_spectrum(
                            E_odd, _sh_priors_reduced,
                            flags=_sh_flags_reduced,
                            bins=fit_kwargs.get("bins"),
                            bin_edges=fit_kwargs.get("bin_edges"),
                            bounds=fit_kwargs.get("bounds"),
                            unbinned=fit_kwargs.get("unbinned", False),
                            strict=False,
                        )

                        def _get_nll(r):
                            p = r.params if isinstance(r, FitResult) else r
                            return float(p.get("nll", float("inf")))

                        _nll_full_in = _get_nll(_sh_full_even)
                        _nll_red_in = _get_nll(_sh_red_even)
                        _nll_full_out = _get_nll(_sh_full_odd)
                        _nll_red_out = _get_nll(_sh_red_odd)

                        _in_sample_impr = _nll_red_in - _nll_full_in
                        _held_out_impr = _nll_red_out - _nll_full_out

                        shelf_halo_result = {
                            "in_sample_improvement": round(float(_in_sample_impr), 2),
                            "held_out_improvement": round(float(_held_out_impr), 2),
                            "n_events_train": int(_sh_even.sum()),
                            "n_events_test": int(_sh_odd.sum()),
                            "verdict": (
                                "justified"
                                if _held_out_impr > 1.0
                                else "marginal"
                                if _held_out_impr > 0.0
                                else "overfitting"
                            ),
                        }
                        logger.info(
                            "Shelf/halo test: in-sample ΔNLL=%.1f, "
                            "held-out ΔNLL=%.1f -> %s",
                            _in_sample_impr, _held_out_impr,
                            shelf_halo_result["verdict"],
                        )
                    except Exception as e:
                        logger.warning("Shelf/halo test failed: %s", e)

                # ── DNL cross-validation ────────────────────────────────
                dnl_crossval_result = None
                _dnl_cv_cfg = spectral_cfg.get("dnl_correction", {})
                if (
                    _dnl_cv_cfg.get("crossval", False)
                    and _dnl_cv_cfg.get("enabled", False)
                    and isinstance(spectrum_results, FitResult)
                ):
                    try:
                        from src.rmtest.spectral.dnl_crossval import run_dnl_crossval

                        _cv_timestamps = None
                        if "timestamp" in df_spectrum.columns:
                            _ts_raw = df_spectrum["timestamp"]
                            try:
                                _cv_timestamps = (
                                    pd.to_datetime(_ts_raw, utc=True, errors="coerce")
                                    .astype(np.int64).to_numpy(dtype=float)
                                )
                            except Exception:
                                _cv_timestamps = np.arange(len(df_spectrum), dtype=float)
                        else:
                            _cv_timestamps = np.arange(len(df_spectrum), dtype=float)

                        # Ensure bin_edges are always passed so both halves
                        # use identical binning (required for cross-applying
                        # DNL factors).  Fall back to bin_edges from the
                        # spectral fit result if not in fit_kwargs.
                        _cv_bin_edges = fit_kwargs.get("bin_edges")
                        if _cv_bin_edges is None and isinstance(spectrum_results, FitResult):
                            _plot_edges = spectrum_results.params.get("_plot_edges")
                            if _plot_edges is not None:
                                _cv_bin_edges = np.asarray(_plot_edges, dtype=float)
                                logger.info(
                                    "DNL crossval: recovered bin_edges from fit "
                                    "result (%d edges)", len(_cv_bin_edges),
                                )
                        _cv_fit_kwargs = {
                            "priors": priors_spec,
                            "flags": spec_flags,
                            "bins": fit_kwargs.get("bins"),
                            "bin_edges": _cv_bin_edges,
                            "bounds": fit_kwargs.get("bounds"),
                            "unbinned": fit_kwargs.get("unbinned", False),
                            "strict": False,
                        }
                        dnl_crossval_result = run_dnl_crossval(
                            energies=E_all,
                            timestamps=_cv_timestamps,
                            fit_kwargs=_cv_fit_kwargs,
                            cfg=cfg,
                        )
                        logger.info(
                            "DNL cross-validation: verdict=%s, corr=%.3f",
                            dnl_crossval_result.verdict,
                            dnl_crossval_result.dnl_correlation,
                        )

                        # ── B1: Auto-disable DNL on overfitting ───────────
                        # If the self-estimated DNL correction is not
                        # reproducible across time halves, disable it and
                        # refit with raw bin widths.  The model without DNL
                        # is statistically more honest: bins are independent
                        # Poisson with no self-reference covariance.
                        if dnl_crossval_result.verdict == "overfitting":
                            logger.warning(
                                "DNL cross-validation indicates overfitting "
                                "(r=%.3f). Disabling self-estimated DNL "
                                "and refitting with raw bin widths.",
                                dnl_crossval_result.dnl_correlation,
                            )
                            _no_dnl_flags = dict(spec_flags)
                            _no_dnl_flags["cfg"] = dict(cfg)
                            _no_dnl_sc = dict(
                                _no_dnl_flags["cfg"].get("spectral_fit", {})
                            )
                            _no_dnl_dc = dict(
                                _no_dnl_sc.get("dnl_correction", {})
                            )
                            _no_dnl_dc["enabled"] = False
                            _no_dnl_sc["dnl_correction"] = _no_dnl_dc
                            _no_dnl_flags["cfg"]["spectral_fit"] = _no_dnl_sc

                            _no_dnl_skip = cfg.get("spectral_fit", {}).get(
                                "skip_minos", False
                            )
                            _no_dnl_result = fit_spectrum(
                                E_all,
                                priors_spec,
                                flags=_no_dnl_flags,
                                bins=fit_kwargs.get("bins"),
                                bin_edges=fit_kwargs.get("bin_edges"),
                                bounds=fit_kwargs.get("bounds"),
                                unbinned=fit_kwargs.get("unbinned", False),
                                strict=fit_kwargs.get("strict", False),
                                skip_minos=_no_dnl_skip,
                            )
                            if isinstance(_no_dnl_result, FitResult):
                                _no_dnl_p = _no_dnl_result.params
                                _no_dnl_p["dnl_status"] = "disabled_by_crossval"
                                _no_dnl_p.setdefault("_dnl", {})
                                _no_dnl_p["_dnl"]["operator_class"] = (
                                    "disabled_by_crossval"
                                )
                                _no_dnl_p["_dnl"]["statistical_model"] = (
                                    "independent_poisson"
                                )
                                _no_dnl_p["_dnl"]["covariance_note"] = (
                                    "Bins are independent Poisson; no "
                                    "self-estimated DNL correction applied."
                                )
                                # Keep the old fit for comparison
                                _old_nll = float(
                                    spec_fit_out.params.get("nll", float("inf"))
                                    if isinstance(spec_fit_out, FitResult)
                                    else spec_fit_out.get("nll", float("inf"))
                                    if isinstance(spec_fit_out, dict)
                                    else float("inf")
                                )
                                _new_nll = float(
                                    _no_dnl_p.get("nll", float("inf"))
                                )
                                logger.info(
                                    "Refit without DNL: NLL %.1f -> %.1f "
                                    "(delta=%.1f)",
                                    _old_nll, _new_nll,
                                    _new_nll - _old_nll,
                                )
                                spec_fit_out = _no_dnl_result
                                spectrum_results = spec_fit_out

                    except Exception as e:
                        logger.warning("DNL cross-validation failed: %s", e)

                fit_vals = None
                if isinstance(spec_fit_out, FitResult):
                    fit_vals = spec_fit_out
                elif isinstance(spec_fit_out, dict):
                    fit_vals = spec_fit_out

                plot_energies = df_analysis["energy_MeV"].to_numpy(dtype=float, copy=False)
                plot_bins = bins
                plot_bin_edges = bin_edges
                plot_flags = dict(spec_flags)
                configured_fit_range = _normalise_fit_energy_range(
                    spectral_cfg.get("fit_energy_range")
                )
                if configured_fit_range is not None:
                    plot_flags["fit_energy_range"] = configured_fit_range

                if method == "energy" and bin_edges is not None:
                    finite_plot = plot_energies[np.isfinite(plot_energies)]
                    if finite_plot.size:
                        plot_min = float(np.min(finite_plot))
                        plot_max = float(np.max(finite_plot))
                        start_steps = int(np.floor((plot_min - float(bin_edges[0])) / width))
                        stop_steps = int(np.ceil((plot_max - float(bin_edges[0])) / width))
                        plot_bin_edges = float(bin_edges[0]) + width * np.arange(
                            start_steps,
                            stop_steps + 1,
                            dtype=float,
                        )
                        if plot_bin_edges.size < 2:
                            plot_bin_edges = np.array(
                                [plot_min, plot_min + width],
                                dtype=float,
                            )
                        elif plot_bin_edges[-1] < plot_max:
                            plot_bin_edges = np.append(
                                plot_bin_edges,
                                plot_bin_edges[-1] + width,
                            )
                        plot_bins = plot_bin_edges.size - 1
                elif method == "adc" and bin_edges is not None:
                    adc_plot = df_analysis["adc"].to_numpy(dtype=float, copy=False)
                    adc_plot = adc_plot[np.isfinite(adc_plot)]
                    if adc_plot.size:
                        plot_edges_adc = adc_hist_edges(adc_plot, channel_width=width)
                        plot_bin_edges = apply_calibration(
                            plot_edges_adc,
                            a,
                            c,
                            quadratic_coeff=a2,
                            cubic_coeff=a3,
                        )
                        plot_bins = plot_bin_edges.size - 1

                spec_plot_data = {
                    "energies": plot_energies,
                    "fit_vals": fit_vals,
                    "bins": plot_bins,
                    "bin_edges": plot_bin_edges,
                    "flags": plot_flags,
                }

        # ────────────────────────────────────────────────────────────
    with timer.section("time_series"):
        # 6. Time‐series decay fits for Po‐218 and Po‐214
        # ────────────────────────────────────────────────────────────
        time_fit_results = {}
        time_fit_background_meta: dict[str, dict[str, Any]] = {}
        priors_time_all = {}
        time_plot_data = {}
        iso_live_time = {}
        t_start_map = {}
        iso_counts = {}
        iso_counts_raw = {}
        radon_estimate_info = None
        po214_estimate_info = None
        po218_estimate_info = None
        allow_negative_baseline = bool(cfg.get("allow_negative_baseline"))
        if cfg.get("time_fit", {}).get("do_time_fit", False):
            time_fit_section = cfg.get("time_fit") or {}
            if isinstance(time_fit_section, Mapping):
                time_cfg: dict[str, Any] = dict(time_fit_section)
            else:
                logger.debug(
                    "time_fit config is not a mapping (%r); using defaults instead",
                    type(time_fit_section),
                )
                time_cfg = {}
    
            model = str(time_cfg.get("model", "single_exp"))
    
            t0_raw = time_cfg.get("t0")
            t0: float | None
            if t0_raw is None:
                t0 = None
            else:
                try:
                    t0 = float(t0_raw)
                except (TypeError, ValueError):
                    logger.debug("Invalid time_fit.t0=%r; ignoring", t0_raw)
                    t0 = None
    
            fix_lambda = bool(time_cfg.get("fix_lambda", False))
    
            lambda_raw = time_cfg.get("lambda")
            lambda_val: float | None = None
            if lambda_raw is not None:
                try:
                    lambda_val = float(lambda_raw)
                except (TypeError, ValueError):
                    logger.debug("Invalid time_fit.lambda=%r; ignoring", lambda_raw)
    
            units = str(time_cfg.get("activity_units", "Bq"))
    
            base_fit_kwargs: dict[str, Any] = {"model": model, "units": units}
            if t0 is not None:
                base_fit_kwargs["t0"] = t0
            if fix_lambda and lambda_val is not None:
                base_fit_kwargs["lambda_fixed"] = lambda_val
    
            for iso in ("Po218", "Po214"):
                win_key = f"window_{iso.lower()}"
    
                # Missing energy window for this isotope -> skip gracefully
                win_range = cfg.get("time_fit", {}).get(win_key)
                if win_range is None:
                    logger.info(
                        "Config key '%s' not found. Skipping time fit for %s.",
                        win_key,
                        iso,
                    )
                    continue
    
                lo, hi = win_range
                probs = window_prob(
                    df_analysis["energy_MeV"].values,
                    df_analysis["denergy_MeV"].values,
                    lo,
                    hi,
                )
                iso_mask = probs > 0
                iso_events = df_analysis[iso_mask].copy()
                iso_events["weight"] = probs[iso_mask]
    
                # Derive minimum counts automatically if not provided
                thr_cfg = cfg.get("time_fit", {}).get("min_counts")
                if thr_cfg is not None:
                    thr = int(thr_cfg)
                    if len(iso_events) < thr:
                        iso_events, (lo, hi) = auto_expand_window(
                            df_analysis, (lo, hi), thr
                        )
                        if len(iso_events) >= thr:
                            logger.info(
                                "expanded %s window to [%.2f, %.2f] MeV", iso, lo, hi
                            )
                else:
                    thr = len(iso_events)
    
                if iso_events.empty:
                    logger.warning(
                        "No events found for %s in [%.3f, %.3f] MeV.", iso, lo, hi
                    )
                    continue
    
                first_ts = to_datetime_utc(iso_events["timestamp"].iloc[0])
                t0_dt = to_utc_datetime(t0_global)
                settle = timedelta(seconds=float(args.settle_s or 0))
                # Start the fit just before the first kept event so fixed-background /
                # fixed-N0 models do not evaluate exactly at r(t=0)=0.
                t_start_fit_dt = max(first_ts, t0_dt + settle) - timedelta(microseconds=1)
                t_start_map[iso] = t_start_fit_dt
                iso_live_time[iso] = (t_end_global - t_start_fit_dt).total_seconds()
    
            # Build priors for time fit
            priors_time = {}
    
            # Efficiency prior per isotope
    
            eff_cfg_val = cfg["time_fit"].get(f"eff_{iso.lower()}")
    
            eff_nom = (
                eff_cfg_val[0] if isinstance(eff_cfg_val, (list, tuple)) else eff_cfg_val
            )
    
            if args.eff_fixed:
                priors_time["eff"] = (1.0, np.inf)
                eff_val = 1.0
            else:
                eff_val, sigma = _eff_prior(eff_cfg_val)
                priors_time["eff"] = (eff_val, sigma)
    
            # Half-life prior (user must supply [T1/2, sigma(T1/2)] in seconds)
            hl_key = f"hl_{iso.lower()}"
            hl_val = cfg["time_fit"].get(hl_key)
            if hl_val is not None:
                if isinstance(hl_val, list):
                    T12 = hl_val[0] if hl_val else None
                    T12sig = hl_val[1] if len(hl_val) > 1 else 0.0
                else:
                    T12 = hl_val
                    T12sig = 0.0
                if T12 is not None:
                    priors_time["tau"] = (T12 / np.log(2), T12sig / np.log(2))
    
            # Background‐rate prior
            if f"bkg_{iso.lower()}" in cfg["time_fit"]:
                priors_time["B0"] = tuple(cfg["time_fit"][f"bkg_{iso.lower()}"])
    
            # Initial N₀ from baseline (if provided)
            if baseline_range:
                # Count baseline events in this energy window
                probs_base = window_prob(
                    base_events["energy_MeV"].values,
                    base_events["denergy_MeV"].values,
                    lo,
                    hi,
                )
                n0_count = float(np.sum(probs_base))
                if iso in isotopes_to_subtract:
                    baseline_counts[iso] = n0_count
    
                eff_cfg = cfg["time_fit"].get(f"eff_{iso.lower()}")
                if isinstance(eff_cfg, list):
                    eff = eff_cfg[0]
                else:
                    eff = eff_cfg if eff_cfg is not None else 1.0
    
                if baseline_record is not None:
                    baseline_handling.update_record_with_counts(
                        baseline_record,
                        iso,
                        n0_count,
                        baseline_live_time,
                        eff,
                    )
                if baseline_live_time > 0 and eff > 0:
                    n0_activity = n0_count / (baseline_live_time * eff)
                    n0_sigma = np.sqrt(n0_count) / (baseline_live_time * eff)
                else:
                    n0_activity = 0.0
                    n0_sigma = 1.0
    
                priors_time["N0"] = (
                    n0_activity,
                    cfg["time_fit"].get(
                        f"sig_n0_{iso.lower()}",
                        cfg["time_fit"].get(f"sig_N0_{iso}", n0_sigma),
                    ),
                )
    
                analysis_counts = float(np.sum(iso_events["weight"]))
                iso_counts_raw[iso] = analysis_counts
                live_time_iso = iso_live_time.get(iso, 0.0)
                if (
                    iso in isotopes_to_subtract
                    and live_time_iso > 0
                    and baseline_live_time > 0
                    and eff > 0
                ):
                    c_rate, c_sigma = subtract_baseline_counts(
                        analysis_counts,
                        eff,
                        live_time_iso,
                        baseline_counts.get(iso, 0.0),
                        baseline_live_time,
                    )
                else:
                    if eff > 0 and live_time_iso > 0:
                        c_rate = analysis_counts / (live_time_iso * eff)
                        c_sigma = math.sqrt(analysis_counts) / (live_time_iso * eff)
                    else:
                        c_rate = 0.0
                        c_sigma = 0.0
                if not allow_negative_baseline and c_rate < 0.0:
                    c_rate = 0.0
                baseline_info.setdefault("corrected_activity", {})[iso] = {
                    "value": c_rate,
                    "uncertainty": c_sigma,
                }
                weight_factor = _time_fit_weight_scale(
                    analysis_counts,
                    live_time_iso,
                    eff,
                    c_sigma,
                )
                iso_events["weight"] *= weight_factor
            else:
                priors_time["N0"] = (
                    0.0,
                    cfg["time_fit"].get(
                        f"sig_n0_{iso.lower()}",
                        cfg["time_fit"].get(f"sig_N0_{iso}", 1.0),
                    ),
                )
    
                analysis_counts = float(np.sum(iso_events["weight"]))
                iso_counts_raw[iso] = analysis_counts
                eff_cfg = cfg["time_fit"].get(f"eff_{iso.lower()}")
                if isinstance(eff_cfg, list):
                    eff = eff_cfg[0]
                else:
                    eff = eff_cfg if eff_cfg is not None else 1.0
                live_time_iso = iso_live_time.get(iso, 0.0)
                if eff > 0 and live_time_iso > 0:
                    c_rate = analysis_counts / (live_time_iso * eff)
                    c_sigma = math.sqrt(analysis_counts) / (live_time_iso * eff)
                else:
                    c_rate = 0.0
                    c_sigma = 0.0
                if not allow_negative_baseline and c_rate < 0.0:
                    c_rate = 0.0
                baseline_info.setdefault("corrected_activity", {})[iso] = {
                    "value": c_rate,
                    "uncertainty": c_sigma,
                }
                weight_factor = _time_fit_weight_scale(
                    analysis_counts,
                    live_time_iso,
                    eff,
                    c_sigma,
                )
                iso_events["weight"] *= weight_factor
    
            # Store priors for use in systematics scanning
            priors_time_all[iso] = priors_time
    
            # Build configuration for fit_time_series
            if args.settle_s is not None:
                t0_dt = to_utc_datetime(t0_global)
                cut = t0_dt + timedelta(seconds=float(args.settle_s))
                iso_events = iso_events[iso_events["timestamp"] >= cut]
            ts_vals = iso_events["timestamp"].map(to_epoch_seconds).to_numpy()
            times_dict = {iso: ts_vals}
            weights_map = {iso: iso_events["weight"].values}
            eff_key = f"eff_{iso.lower()}"
            eff_cfg_val = cfg["time_fit"].get(eff_key)
            eff_value: float | None
            if args.eff_fixed:
                eff_value = None
            else:
                explicit_null = False
                if eff_key in cfg["time_fit"]:
                    if eff_cfg_val in (None, "null"):
                        explicit_null = True
                    elif isinstance(eff_cfg_val, (list, tuple)):
                        explicit_null = bool(eff_cfg_val) and eff_cfg_val[0] in (
                            None,
                            "null",
                        )
                if explicit_null:
                    eff_value = None
                else:
                    eff_value = _config_efficiency(cfg, iso)
            fit_cfg = {
                "isotopes": {
                    iso: {
                        "half_life_s": _hl_value(cfg, iso),
                        "efficiency": eff_value,
                    }
                },
                "fit_background": not cfg["time_fit"]["flags"].get(
                    "fix_background_b", False
                ),
                "fit_initial": not (
                    cfg["time_fit"]["flags"].get(f"fix_n0_{iso.lower()}", False)
                    or cfg["time_fit"]["flags"].get(f"fix_N0_{iso.lower()}", False)
                ),
                "background_guess": cfg["time_fit"].get("background_guess", 0.0),
                "n0_guess_fraction": cfg["time_fit"].get("n0_guess_fraction", 0.1),
                "min_counts": thr,
                "fix_background_b_first_pass": cfg["time_fit"].get(
                    "fix_background_b_first_pass", True
                ),
                "background_b_fixed_value": cfg["time_fit"].get(
                    "background_b_fixed_value"
                ),
            }
    
            # Determine baseline rate for fixed-background first pass
            baseline_rate_iso = None
            fixed_from_baseline_info = None
            if baseline_record is not None:
                fixed_from_baseline_info = baseline_handling.get_fixed_background_for_time_fit(
                    baseline_record,
                    iso,
                    cfg.get("baseline", {}),
                )
                if fixed_from_baseline_info:
                    baseline_rate_iso = fixed_from_baseline_info.get("background_rate_Bq")
    
            if baseline_rate_iso is None and baseline_live_time > 0:
                eff_cfg = cfg["time_fit"].get(f"eff_{iso.lower()}")
                if isinstance(eff_cfg, list):
                    eff_rate = eff_cfg[0]
                else:
                    eff_rate = eff_cfg if eff_cfg is not None else 1.0
                if eff_rate > 0:
                    baseline_rate_iso = baseline_counts.get(iso, 0.0) / (
                        baseline_live_time * eff_rate
                    )
    
            # Run time-series fit (two-pass)
            decay_out = None  # fresh variable each iteration
            try:
                t_start_val = t_start_map.get(iso)
                if isinstance(t_start_val, datetime):
                    t_start_fit = t_start_val.timestamp()
                else:
                    t_start_fit = to_utc_datetime(
                        t_start_val if t_start_val is not None else t0_global
                    ).timestamp()
                decay_out = two_pass_time_fit(
                    times_dict,
                    t_start_fit,
                    t_end_global_ts,
                    fit_cfg,
                    baseline_rate=baseline_rate_iso,
                    weights=weights_map,
                    strict=args.strict_covariance,
                    fit_func=fit_time_series,
                    fit_kwargs=base_fit_kwargs,
                )
                time_fit_results[iso] = decay_out
            except Exception as e:
                logging.warning("Decay-curve fit for %s failed -> %s", iso, e)
                time_fit_results[iso] = {}
    
            # Record how the background parameter was treated
            background_mode = "floated"
            baseline_rate_meta: float | None = None
            if isinstance(decay_out, FitResult):
                param_index = decay_out.param_index or {}
                has_background_param = f"B_{iso}" in param_index
                background_mode = "floated" if has_background_param else "fixed"
            elif isinstance(decay_out, Mapping):
                has_background_param = f"B_{iso}" in decay_out
                background_mode = "floated" if has_background_param else "fixed"
            else:
                background_mode = "floated" if fit_cfg.get("fit_background") else "fixed"
    
            if background_mode == "fixed":
                if fixed_from_baseline_info:
                    baseline_rate_meta = float(
                        fixed_from_baseline_info.get(
                            "background_rate_Bq", baseline_rate_iso or 0.0
                        )
                    )
                    norm_mode = baseline_handling.normalize_background_mode(
                        fixed_from_baseline_info.get("mode")
                    ) or "fixed_from_baseline"
                    fixed_from_baseline_info = dict(fixed_from_baseline_info)
                    fixed_from_baseline_info["mode"] = norm_mode
                    background_mode = norm_mode
                    baseline_background_provenance[iso] = dict(fixed_from_baseline_info)
                elif baseline_rate_iso is not None:
                    baseline_rate_meta = float(baseline_rate_iso)
                    background_mode = baseline_handling.normalize_background_mode(
                        "fixed_from_baseline"
                    ) or "fixed_from_baseline"
    
            background_mode = baseline_handling.normalize_background_mode(background_mode)
    
            meta_entry: dict[str, Any] = {"mode": background_mode}
            if baseline_rate_meta is not None:
                meta_entry["baseline_rate_Bq"] = baseline_rate_meta
            if (
                fixed_from_baseline_info
                and fixed_from_baseline_info.get("background_unc_Bq") is not None
            ):
                meta_entry["baseline_unc_Bq"] = float(
                    fixed_from_baseline_info["background_unc_Bq"]
                )
            time_fit_background_meta[iso] = meta_entry
    
            # Store inputs for plotting later
            time_plot_data[iso] = {
                "events_times": iso_events["timestamp"].values,
                "events_energy": iso_events["energy_MeV"].values,
            }
    
        def _counts_corrected_rate(
            iso: str, params: Mapping[str, Any]
        ) -> tuple[float, float] | None:
            """Return a baseline-corrected rate from raw counts when fits fail."""
    
            counts_val = iso_counts_raw.get(iso)
            if counts_val is None:
                return None
    
            live_time_iso = iso_live_time.get(iso)
            if live_time_iso is None or live_time_iso <= 0:
                return None
    
            eff_val = _resolved_efficiency(cfg, iso, params)
            if eff_val is None or eff_val <= 0:
                eff_val = _config_efficiency(cfg, iso)
            if eff_val is None or eff_val <= 0:
                return None
    
            if (
                args.baseline_mode != "none"
                and iso in isotopes_to_subtract
                and baseline_live_time > 0
            ):
                base_counts = baseline_counts.get(iso, 0.0)
                try:
                    rate, sigma = subtract_baseline_counts(
                        float(counts_val),
                        float(eff_val),
                        float(live_time_iso),
                        float(base_counts),
                        float(baseline_live_time),
                    )
                except ValueError:
                    return None
                rate = float(rate)
                if not allow_negative_baseline and rate < 0.0:
                    rate = 0.0
                return rate, float(sigma)
    
            rate = float(counts_val) / (float(live_time_iso) * float(eff_val))
            sigma = math.sqrt(abs(float(counts_val))) / (
                float(live_time_iso) * float(eff_val)
            )
            if not allow_negative_baseline and rate < 0.0:
                rate = 0.0
            return rate, sigma
    
        # --- Radon combination ---
        from radon_joint_estimator import estimate_radon_activity
        from types import SimpleNamespace
    
        fit214_obj = time_fit_results.get("Po214")
        fit218_obj = time_fit_results.get("Po218")
        fit214 = fit218 = None
    
        def _coerce_float(value: Any) -> float | None:
            try:
                return float(value)
            except (TypeError, ValueError):
                return None
    
        if fit214_obj:
            p = _fit_params(fit214_obj)
            rate_val = _coerce_float(p.get("E_corrected", p.get("E_Po214")))
            err_val = _coerce_float(p.get("dE_corrected", p.get("dE_Po214")))
            fallback_needed = False
            if rate_val is None or not math.isfinite(rate_val):
                fallback_needed = True
            if not bool(p.get("fit_valid", True)):
                fallback_needed = True
            if fallback_needed:
                fallback_res = _counts_corrected_rate("Po214", p)
                if fallback_res is not None:
                    rate_val, err_val = fallback_res
                    p["E_corrected"] = rate_val
                    p["dE_corrected"] = err_val
                    p["counts_fallback"] = True
            if err_val is None or not math.isfinite(err_val) or err_val < 0:
                err_val = _fallback_uncertainty(rate_val, fit214_obj, "E_Po214")
            fit214 = SimpleNamespace(
                rate=rate_val,
                err=err_val,
                counts=getattr(fit214_obj, "counts", None),
                params=p,
            )
        if fit218_obj:
            p = _fit_params(fit218_obj)
            rate_val = _coerce_float(p.get("E_corrected", p.get("E_Po218")))
            err_val = _coerce_float(p.get("dE_corrected", p.get("dE_Po218")))
            fallback_needed = False
            if rate_val is None or not math.isfinite(rate_val):
                fallback_needed = True
            if not bool(p.get("fit_valid", True)):
                fallback_needed = True
            if fallback_needed:
                fallback_res = _counts_corrected_rate("Po218", p)
                if fallback_res is not None:
                    rate_val, err_val = fallback_res
                    p["E_corrected"] = rate_val
                    p["dE_corrected"] = err_val
                    p["counts_fallback"] = True
            if err_val is None or not math.isfinite(err_val) or err_val < 0:
                err_val = _fallback_uncertainty(rate_val, fit218_obj, "E_Po218")
            fit218 = SimpleNamespace(
                rate=rate_val,
                err=err_val,
                counts=getattr(fit218_obj, "counts", None),
                params=p,
            )
    
        iso_mode = cfg.get("analysis_isotope", "radon").lower()
    
        if iso_mode == "radon":
            have_218 = (
                fit218
                and fit218.counts is not None
                and _fit_efficiency(fit218.params, "Po218") is not None
            )
            have_214 = (
                fit214
                and fit214.counts is not None
                and _fit_efficiency(fit214.params, "Po214") is not None
            )
            if have_218 or have_214:
                N218 = fit218.counts if have_218 else None
                N214 = fit214.counts if have_214 else None
                eps218 = (
                    _resolved_efficiency(cfg, "Po218", fit218.params)
                    if fit218
                    else 1.0
                )
                eps214 = (
                    _resolved_efficiency(cfg, "Po214", fit214.params)
                    if fit214
                    else 1.0
                )
                lt218 = iso_live_time.get("Po218") if have_218 else None
                lt214 = iso_live_time.get("Po214") if have_214 else None
                radon_estimate_info = estimate_radon_activity(
                    N218=N218,
                    epsilon218=eps218,
                    f218=1.0,
                    N214=N214,
                    epsilon214=eps214,
                    f214=1.0,
                    live_time218_s=lt218,
                    live_time214_s=lt214,
                )
            elif (fit214 and fit214.rate is not None) or (
                fit218 and fit218.rate is not None
            ):
                radon_estimate_info = estimate_radon_activity(
                    rate214=fit214.rate if fit214 else None,
                    err214=fit214.err if fit214 else None,
                    rate218=fit218.rate if fit218 else None,
                    err218=fit218.err if fit218 else None,
                )
        elif iso_mode == "po218":
            if fit218:
                po218_estimate_info = {
                    "activity_Bq": fit218.rate,
                    "stat_unc_Bq": fit218.err,
                }
        elif iso_mode == "po214":
            if fit214:
                po214_estimate_info = {
                    "activity_Bq": fit214.rate,
                    "stat_unc_Bq": fit214.err,
                }
        else:
            raise ValueError(f"Unknown analysis isotope {iso_mode}")
    
        # Also extract Po-210 events for plotting if a window is provided
        win_p210 = cfg.get("time_fit", {}).get("window_po210")
        if win_p210 is not None:
            lo, hi = win_p210
            mask210 = (
                (df_analysis["energy_MeV"] >= lo)
                & (df_analysis["energy_MeV"] <= hi)
                & (df_analysis["timestamp"] >= to_datetime_utc(t0_global))
                & (df_analysis["timestamp"] <= t_end_global)
            )
            events_p210 = df_analysis[mask210]
            time_plot_data["Po210"] = {
                "events_times": events_p210["timestamp"].values,
                "events_energy": events_p210["energy_MeV"].values,
            }

        # Also extract Po-212 events for plotting if a window is provided
        win_p212 = cfg.get("time_fit", {}).get("window_po212")
        if win_p212 is not None:
            lo, hi = win_p212
            mask212 = (
                (df_analysis["energy_MeV"] >= lo)
                & (df_analysis["energy_MeV"] <= hi)
                & (df_analysis["timestamp"] >= to_datetime_utc(t0_global))
                & (df_analysis["timestamp"] <= t_end_global)
            )
            events_p212 = df_analysis[mask212]
            time_plot_data["Po212"] = {
                "events_times": events_p212["timestamp"].values,
                "events_energy": events_p212["energy_MeV"].values,
            }

        # ────────────────────────────────────────────────────────────
    with timer.section("systematics"):
        # 7. Systematics scan (optional)
        # ────────────────────────────────────────────────────────────
        systematics_results = {}
        if cfg.get("systematics", {}).get("enable", False):
            sys_cfg = cfg.get("systematics", {})
    
            for iso, fit_out in time_fit_results.items():
                if not fit_out:
                    continue
    
                sigma_dict = {}
                for name, base in (
                    ("sigma_e_frac", "sigma_E"),
                    ("sigma_E_frac", "sigma_E"),
                    ("tail_fraction", "tail"),
                    ("energy_shift_kev", "energy_shift"),
                    ("energy_shift_keV", "energy_shift"),
                ):
                    if name in sys_cfg and base in priors_time_all.get(iso, {}):
                        sigma_dict[name] = sys_cfg[name]
    
                # Build a wrapper to re‐run fit_time_series with modified priors
                def fit_wrapper(priors_mod):
                    win_range = cfg.get("time_fit", {}).get(f"window_{iso.lower()}")
                    if win_range is None:
                        raise ValueError(
                            f"Missing window for {iso} during systematics scan"
                        )
                    probs = window_prob(
                        df_analysis["energy_MeV"].values,
                        df_analysis["denergy_MeV"].values,
                        win_range[0],
                        win_range[1],
                    )
                    mask = probs > 0
                    filtered_df = df_analysis[mask]
                    ts_vals = filtered_df["timestamp"].map(to_epoch_seconds).to_numpy()
                    times_dict = {iso: ts_vals}
                    weights_local = {iso: probs[mask]}
                    cfg_fit = {
                        "isotopes": {
                            iso: {
                                "half_life_s": _hl_value(cfg, iso),
                                "efficiency": priors_mod["eff"][0],
                            }
                        },
                        "fit_background": not cfg["time_fit"]["flags"].get(
                            "fix_background_b", False
                        ),
                        "fit_initial": not (
                            cfg["time_fit"]["flags"].get(f"fix_n0_{iso.lower()}", False)
                            or cfg["time_fit"]["flags"].get(f"fix_N0_{iso.lower()}", False)
                        ),
                        "background_guess": cfg["time_fit"].get("background_guess", 0.0),
                        "n0_guess_fraction": cfg["time_fit"].get("n0_guess_fraction", 0.1),
                    }
                    try:
                        out = fit_time_series(
                            times_dict,
                            t0_global.timestamp(),
                            t_end_global_ts,
                            cfg_fit,
                            weights=weights_local,
                            strict=args.strict_covariance,
                        )
                    except TypeError:
                        out = fit_time_series(
                            times_dict,
                            t0_global.timestamp(),
                            t_end_global_ts,
                            cfg_fit,
                            strict=args.strict_covariance,
                        )
                    # Return only the parameter dictionary so scan_systematics
                    # works with a simple mapping.
                    return out.params
    
                try:
                    deltas, total_unc = scan_systematics(
                        fit_wrapper, priors_time_all.get(iso, {}), sigma_dict
                    )
                    systematics_results[iso] = {"deltas": deltas, "total_unc": total_unc}
                except Exception as e:
                    logger.warning("Systematics scan for %s -> %s", iso, e)
    
        # ────────────────────────────────────────────────────────────
        # 7b. Optional efficiency calculations
        # ────────────────────────────────────────────────────────────
        efficiency_results = {}
        weights = None
        eff_cfg = cfg.get("efficiency", {})
        if eff_cfg:
            from efficiency import (
                calc_assay_efficiency,
                calc_decay_efficiency,
                blue_combine,
            )
    
            sources = {}
            vals, errs = [], []
    
            if "spike" in eff_cfg:
                scfg_raw = eff_cfg["spike"]
                scfg_list = [scfg_raw] if isinstance(scfg_raw, dict) else list(scfg_raw)
                for idx, scfg in enumerate(scfg_list, start=1):
                    key = "spike" if isinstance(scfg_raw, dict) else f"spike_{idx}"
                    if not scfg.get("enabled", True):
                        logger.info("Spike efficiency '%s' disabled", key)
                        continue
                    try:
                        val = get_spike_efficiency(scfg)
                        err = float(scfg.get("error", 0.0))
                        sources[key] = {"value": val, "error": err}
                        vals.append(val)
                        errs.append(err)
                    except Exception as e:
                        logger.warning("Spike efficiency -> %s", e)
    
            if "assay" in eff_cfg:
                acfg = eff_cfg["assay"]
                if isinstance(acfg, dict):
                    acfg_list = [acfg]
                else:
                    acfg_list = list(acfg)
                for idx, cfg_item in enumerate(acfg_list, start=1):
                    try:
                        val = calc_assay_efficiency(
                            cfg_item["rate_cps"], cfg_item["reference_bq"]
                        )
                        err = float(cfg_item.get("error", 0.0))
                        key = "assay" if isinstance(acfg, dict) else f"assay_{idx}"
                        sources[key] = {"value": val, "error": err}
                        vals.append(val)
                        errs.append(err)
                    except Exception as e:
                        logger.warning("Assay efficiency -> %s", e)
    
            if "decay" in eff_cfg:
                dcfg = eff_cfg["decay"]
                try:
                    val = calc_decay_efficiency(
                        dcfg["observed_rate"], dcfg["expected_rate"]
                    )
                    err = float(dcfg.get("error", 0.0))
                    sources["decay"] = {"value": val, "error": err}
                    vals.append(val)
                    errs.append(err)
                except Exception as e:
                    logger.warning("Decay efficiency -> %s", e)
    
            efficiency_results["sources"] = sources
            if vals:
                try:
                    comb_val, comb_err, weights = blue_combine(vals, errs)
                    efficiency_results["combined"] = {
                        "value": float(comb_val),
                        "error": float(comb_err),
                        "weights": weights.tolist(),
                    }
                except Exception as e:
                    logger.warning("BLUE combination failed -> %s", e)
    
        # ────────────────────────────────────────────────────────────
        # Baseline subtraction
        # ────────────────────────────────────────────────────────────
        """Apply baseline correction and compute associated uncertainties.
    
        The counts from the baseline interval are converted to rates and
        subtracted from the fitted activities.  The error term ``sigma_rate``
        used for this correction is derived from the **unweighted** analysis
        counts so that the statistical uncertainty reflects the raw event
        totals prior to any BLUE weighting.
        """
        baseline_rates = {}
        baseline_unc = {}
        scales = baseline_info.get("scales", {})
    
        if baseline_record is not None:
            rates_map = baseline_record.get("rates_Bq")
            if isinstance(rates_map, Mapping):
                baseline_rates = {str(k): float(v) for k, v in rates_map.items()}
            sig_map = baseline_record.get("rate_unc_Bq")
            if isinstance(sig_map, Mapping):
                baseline_unc = {str(k): float(v) for k, v in sig_map.items()}
            if not scales:
                scale_map = baseline_record.get("scale_factors")
                if isinstance(scale_map, Mapping):
                    scales = {str(k): float(v) for k, v in scale_map.items()}
                    baseline_info["scales"] = scales
        else:
            if baseline_live_time > 0:
                for iso, n in baseline_counts.items():
                    params = _fit_params(time_fit_results.get(iso))
                    eff = _resolved_efficiency(cfg, iso, params)
                    if eff > 0:
                        baseline_rates[iso] = n / (baseline_live_time * eff)
                        baseline_unc[iso] = np.sqrt(n) / (baseline_live_time * eff)
                    else:
                        baseline_rates[iso] = 0.0
                        baseline_unc[iso] = 0.0
    
            if dilution_factor is None:
                try:
                    dilution_factor = compute_dilution_factor(monitor_vol, sample_vol)
                except ValueError as exc:
                    msg = (
                        "invalid baseline volumes: "
                        f"monitor_volume_l={monitor_vol!r}, sample_volume_l={sample_vol!r}"
                    )
                    if cfg.get("allow_fallback"):
                        monitor_safe = max(monitor_vol, 0.0)
                        sample_safe = max(sample_vol, 0.0)
                        total_safe = monitor_safe + sample_safe
                        if monitor_safe <= 0 or total_safe <= 0:
                            raise ValueError(msg) from exc
                        logger.warning("%s --clamping to non-negative values", msg)
                        monitor_vol = monitor_safe
                        sample_vol = sample_safe
                        dilution_factor = monitor_safe / total_safe
                        warnings_list = baseline_info.setdefault("warnings", [])
                        warnings_list.append(msg)
                        baseline_info["dilution_factor_fallback"] = True
                    else:
                        raise ValueError(msg) from exc
            if not scales:
                if dilution_factor is not None:
                    scales = {
                        "Po214": dilution_factor,
                        "Po218": dilution_factor,
                        "Po210": 1.0,
                        "noise": 1.0,
                    }
                else:
                    scales = {
                        "Po214": 1.0,
                        "Po218": 1.0,
                        "Po210": 1.0,
                        "noise": 1.0,
                    }
                baseline_info["scales"] = scales
        if baseline_record is not None:
            baseline_handling.finalize_baseline_record(baseline_record, baseline_info)
        else:
            if baseline_rates:
                baseline_info["rate_Bq"] = baseline_rates
            if baseline_unc:
                baseline_info["rate_unc_Bq"] = baseline_unc
            if dilution_factor is not None:
                baseline_info.setdefault("dilution_factor", dilution_factor)
    
        baseline_info["analysis_counts"] = iso_counts_raw
    
        corrected_rates = {}
        corrected_unc = {}
        activity_rows = []
    
        for iso, fit in time_fit_results.items():
            params = _fit_params(fit)
            if not params or f"E_{iso}" not in params:
                continue
    
            fallback_used = False
            if not bool(params.get("fit_valid", True)):
                fallback_res = _counts_corrected_rate(iso, params)
                if fallback_res is not None:
                    corr_rate, corr_sigma = fallback_res
                    params["E_corrected"] = corr_rate
                    params["dE_corrected"] = corr_sigma
                    params["counts_fallback"] = True
                    corrected_rates[iso] = corr_rate
                    corrected_unc[iso] = corr_sigma
                    fallback_used = True
                else:
                    params["counts_fallback"] = True
    
            if fallback_used or iso not in isotopes_to_subtract or baseline_live_time <= 0:
                continue
    
            err_fit = params.get(f"dE_{iso}", 0.0)
            live_time_iso = iso_live_time.get(iso, 0.0)
            count = iso_counts_raw.get(iso, baseline_counts.get(iso, 0.0))
            eff = _resolved_efficiency(cfg, iso, params)
            base_cnt = baseline_counts.get(iso, 0.0)
            s = scales.get(iso, 1.0)
    
            if args.baseline_mode == "none":
                base_rate = baseline_rates.get(iso, 0.0)
                base_sigma = baseline_unc.get(iso, 0.0)
                corr_rate = params[f"E_{iso}"]
                corr_sigma = err_fit
            elif live_time_iso > 0 and eff > 0:
                corr_rate, corr_sigma, base_rate, base_sigma = subtract_baseline_rate(
                    params[f"E_{iso}"],
                    err_fit,
                    count,
                    eff,
                    live_time_iso,
                    base_cnt,
                    baseline_live_time,
                    scale=s,
                )
            else:
                corr_rate = params[f"E_{iso}"]
                corr_sigma = err_fit
                base_rate = 0.0
                base_sigma = 0.0
    
            if not allow_negative_baseline and corr_rate < 0.0:
                corr_rate = 0.0
    
            params["E_corrected"] = corr_rate
            params["dE_corrected"] = corr_sigma
            if allow_negative_baseline:
                corrected_map = baseline_info.setdefault("corrected_activity", {})
                entry = corrected_map.get(iso, {})
                if not isinstance(entry, dict):
                    entry = {}
                entry["value"] = corr_rate
                entry["uncertainty"] = corr_sigma
                corrected_map[iso] = entry
            baseline_rates[iso] = base_rate
            baseline_unc[iso] = base_sigma
            corrected_rates[iso] = corr_rate
            corrected_unc[iso] = corr_sigma
            activity_rows.append(
                {
                    "iso": iso,
                    "raw_rate": params[f"E_{iso}"],
                    "baseline_rate": base_rate,
                    "corrected": corr_rate,
                    "err_raw": err_fit,
                    "err_corrected": corr_sigma,
                }
            )
    
        if baseline_rates:
            baseline_info["rate_Bq"] = baseline_rates
            baseline_info["rate_unc_Bq"] = baseline_unc
            baseline_info["dilution_factor"] = dilution_factor
        if baseline_info.get("corrected_activity"):
            baseline_info["corrected_rate_Bq"] = {
                iso: vals["value"]
                for iso, vals in baseline_info["corrected_activity"].items()
            }
            baseline_info["corrected_sigma_Bq"] = {
                iso: vals["uncertainty"]
                for iso, vals in baseline_info["corrected_activity"].items()
            }
    
        try:
            _ = summarize_baseline(
                {
                    "baseline": baseline_info,
                    "time_fit": {
                        iso: _fit_params(time_fit_results.get(iso))
                        for iso in isotopes_to_subtract
                    },
                    "allow_negative_baseline": cfg.get("allow_negative_baseline"),
                },
                isotopes_to_subtract,
            )
        except BaselineError as e:
            logger.error("%s", e)
            sys.exit(1)
    
        # ────────────────────────────────────────────────────────────
        # Radon activity extrapolation
        # ────────────────────────────────────────────────────────────
        from radon_activity import compute_radon_activity
    
        radon_results = {}
        radon_combined_info = None
        iso_mode = cfg.get("analysis_isotope", "radon").lower()
    
        def _eff_value_local(key):
            val = cfg.get("time_fit", {}).get(key)
            if isinstance(val, list):
                return val[0]
            return val if val is not None else 1.0
    
        eff_po214 = _eff_value_local("eff_po214")
        eff_po218 = _eff_value_local("eff_po218")
    
        rate214 = None
        err214 = None
        if "Po214" in time_fit_results:
            fit_dict = _fit_params(time_fit_results["Po214"])
            rate_raw = fit_dict.get("E_corrected", fit_dict.get("E_Po214"))
            try:
                rate214 = float(rate_raw) if rate_raw is not None else None
            except (TypeError, ValueError):
                rate214 = None
            err_raw = fit_dict.get("dE_corrected", fit_dict.get("dE_Po214"))
            try:
                err_val = float(err_raw) if err_raw is not None else None
            except (TypeError, ValueError):
                err_val = None
            if err_val is not None and (not math.isfinite(err_val) or err_val < 0):
                err_val = None
            if err_val is None:
                err214 = _fallback_uncertainty(
                    rate214,
                    time_fit_results.get("Po214"),
                    "E_Po214",
                )
            else:
                err214 = err_val
    
        rate218 = None
        err218 = None
        if "Po218" in time_fit_results:
            fit_dict = _fit_params(time_fit_results["Po218"])
            rate_raw = fit_dict.get("E_corrected", fit_dict.get("E_Po218"))
            try:
                rate218 = float(rate_raw) if rate_raw is not None else None
            except (TypeError, ValueError):
                rate218 = None
            err_raw = fit_dict.get("dE_corrected", fit_dict.get("dE_Po218"))
            try:
                err_val = float(err_raw) if err_raw is not None else None
            except (TypeError, ValueError):
                err_val = None
            if err_val is not None and (not math.isfinite(err_val) or err_val < 0):
                err_val = None
            if err_val is None:
                err218 = _fallback_uncertainty(
                    rate218,
                    time_fit_results.get("Po218"),
                    "E_Po218",
                )
            else:
                err218 = err_val
    
        if iso_mode == "radon":
            r218 = rate218
            e218 = err218
            r214 = rate214
            e214 = err214
        elif iso_mode == "po218":
            r218 = rate218
            e218 = err218
            r214 = None
            e214 = None
        elif iso_mode == "po214":
            r218 = None
            e218 = None
            r214 = rate214
            e214 = err214
        else:
            raise ValueError(f"Unknown analysis_isotope {iso_mode!r}")
    
        A_radon, dA_radon = compute_radon_activity(
            r218,
            e218,
            eff_po218,
            r214,
            e214,
            eff_po214,
        )
    
        if iso_mode == "radon":
            radon_combined_info = {
                "activity_Bq": A_radon,
                "unc_Bq": dA_radon,
            }
    
        # Convert activity to a concentration per liter of the sampled air (when present)
        # and retain the total amount of radon inferred from the sample without
        # diluting by the chamber volume.
        try:
            conc, dconc, total_bq, dtotal_bq = radon_activity.compute_total_radon(
                A_radon,
                dA_radon,
                monitor_vol,
                sample_vol,
                allow_negative_activity=args.allow_negative_activity,
            )
        except RuntimeError as e:
            logger.error("%s", e)
            sys.exit(1)
    
        radon_results["radon_activity_Bq"] = {"value": A_radon, "uncertainty": dA_radon}
        radon_results["radon_concentration_Bq_per_L"] = {
            "value": conc,
            "uncertainty": dconc,
        }
        total_bq_display = total_bq
        if total_bq_display < 0:
            if args.allow_negative_activity:
                logger.warning(
                    "Negative total radon in sample reported (%.3f Bq) because --allow-negative-activity was requested",
                    total_bq_display,
                )
            else:
                logger.warning(
                    "Negative total radon in sample (%.3f Bq) clamped to 0.0 Bq. Re-run with --allow-negative-activity to allow negatives",
                    total_bq_display,
                )
                total_bq_display = 0.0
                dtotal_bq = max(dtotal_bq, NEGATIVE_ACTIVITY_CLAMP_UNCERTAINTY_BQ)
    
        radon_results["total_radon_in_sample_Bq"] = {
            "value": total_bq_display,
            "uncertainty": dtotal_bq,
        }
    
        scan_results = {}
        best_params = None
        if args.burst_sensitivity_scan:
            scan_results, best_params = _burst_sensitivity_scan(
                events_after_noise, cfg, cal_result
            )
    
        if args.debug:
            from radon_activity import print_activity_breakdown
    
            print_activity_breakdown(activity_rows)
    
        if radon_interval is not None:
            from radon_activity import radon_delta
    
            t_start_rel = (radon_interval[0] - analysis_start).total_seconds()
            t_end_rel = (radon_interval[1] - analysis_start).total_seconds()
    
            delta214 = err_delta214 = None
            if "Po214" in time_fit_results:
                fit_result = time_fit_results["Po214"]
                fit = _fit_params(fit_result)
                E = _safe_float(fit.get("E_corrected", fit.get("E_Po214")))
                if E is None:
                    logger.warning(
                        "Skipping radon delta calculation for Po214 because the fitted E parameter is missing or non-finite."
                    )
                else:
                    dE = _float_with_default(
                        fit.get("dE_corrected", fit.get("dE_Po214", 0.0)), 0.0
                    )
                    N0 = _float_with_default(fit.get("N0_Po214", 0.0), 0.0)
                    dN0 = _float_with_default(fit.get("dN0_Po214", 0.0), 0.0)
                    hl = _hl_value(cfg, "Po214")
                    cov = _safe_float(_cov_lookup(fit_result, "E_Po214", "N0_Po214"))
                    delta214, err_delta214 = radon_delta(
                        t_start_rel,
                        t_end_rel,
                        E,
                        dE,
                        N0,
                        dN0,
                        hl,
                        0.0 if cov is None else cov,
                    )
    
            delta218 = err_delta218 = None
            if "Po218" in time_fit_results:
                fit_result = time_fit_results["Po218"]
                fit = _fit_params(fit_result)
                E = _safe_float(fit.get("E_corrected", fit.get("E_Po218")))
                if E is None:
                    logger.warning(
                        "Skipping radon delta calculation for Po218 because the fitted E parameter is missing or non-finite."
                    )
                else:
                    dE = _float_with_default(
                        fit.get("dE_corrected", fit.get("dE_Po218", 0.0)), 0.0
                    )
                    N0 = _float_with_default(fit.get("N0_Po218", 0.0), 0.0)
                    dN0 = _float_with_default(fit.get("dN0_Po218", 0.0), 0.0)
                    hl = _hl_value(cfg, "Po218")
                    cov = _safe_float(_cov_lookup(fit_result, "E_Po218", "N0_Po218"))
                    delta218, err_delta218 = radon_delta(
                        t_start_rel,
                        t_end_rel,
                        E,
                        dE,
                        N0,
                        dN0,
                        hl,
                        0.0 if cov is None else cov,
                    )
    
            d_radon, d_err = compute_radon_activity(
                delta218,
                err_delta218,
                eff_po218,
                delta214,
                err_delta214,
                eff_po214,
            )
            radon_results["radon_delta_Bq"] = {"value": d_radon, "uncertainty": d_err}
    
        # ────────────────────────────────────────────────────────────
    with timer.section("summary_json"):
        # 8. Assemble and write out the summary JSON
        # ────────────────────────────────────────────────────────────
        spec_dict = {}
        if isinstance(spectrum_results, FitResult):
            spec_dict = dict(spectrum_results.params)
            spec_dict["cov"] = spectrum_results.cov.tolist()
            spec_dict["ndf"] = spectrum_results.ndf
            spec_dict["likelihood_path"] = spectrum_results.params.get("likelihood_path")
            # Add named correlation matrix for diagnostics
            if spectrum_results.cov is not None and spectrum_results.param_index:
                _pi = spectrum_results.param_index
                _names = sorted(_pi, key=lambda k: _pi[k])
                _cov = np.asarray(spectrum_results.cov, dtype=float)
                _n = len(_names)
                if _cov.shape == (_n, _n):
                    _diag = np.sqrt(np.clip(np.diag(_cov), 1e-30, None))
                    _corr = _cov / np.outer(_diag, _diag)
                    np.clip(_corr, -1.0, 1.0, out=_corr)
                    spec_dict["param_names"] = _names
                    spec_dict["correlation_matrix"] = _corr.tolist()
                    # Top correlations (|r| > 0.3) for quick inspection
                    _top = []
                    for _ci in range(_n):
                        for _cj in range(_ci + 1, _n):
                            if abs(_corr[_ci, _cj]) > 0.3:
                                _top.append({
                                    "p1": _names[_ci],
                                    "p2": _names[_cj],
                                    "r": round(float(_corr[_ci, _cj]), 4),
                                })
                    _top.sort(key=lambda x: -abs(x["r"]))
                    spec_dict["strong_correlations"] = _top
        elif isinstance(spectrum_results, dict):
            spec_dict = spectrum_results
            spec_dict["likelihood_path"] = spectrum_results.get("likelihood_path")
        if peak_deviation:
            spec_dict["peak_deviation"] = peak_deviation
        # Spectral fit timing breakdown
        _timers = spec_dict.pop("_fit_timers", None)
        if _timers:
            spec_dict["fit_timers"] = {
                k: round(v, 2) if isinstance(v, float) else v
                for k, v in _timers.items()
            }
        # Centroid refit tracking
        _cr_meta = spec_dict.pop("_centroid_refit", None)
        if _cr_meta:
            spec_dict["centroid_refit"] = _cr_meta
        # Three-stage binning metadata
        _ts_meta = spec_dict.pop("_three_stage", None)
        if _ts_meta:
            spec_dict["three_stage_binning"] = {
                k: (round(v, 4) if isinstance(v, float) else v)
                for k, v in _ts_meta.items()
            }
        if split_half_result is not None:
            spec_dict["split_half_validation"] = split_half_result
        if model_comparison_result is not None:
            spec_dict["model_comparison"] = model_comparison_result
        if dnl_crossval_result is not None:
            spec_dict["dnl_crossval"] = dnl_crossval_result.to_dict()
        if _per_period_crossval_data:
            spec_dict["per_period_crossval"] = _per_period_crossval_data
        if shelf_halo_result is not None:
            spec_dict["shelf_halo_crossval"] = shelf_halo_result

        # ── Adequacy verdict ────────────────────────────────────────
        _adequacy = {"level": "uncalibrated", "reasons": []}
        if dnl_crossval_result is not None:
            _cv = dnl_crossval_result
            _reasons = []
            if _cv.verdict == "hardware_signal":
                _reasons.append(
                    "DNL cross-validation confirms hardware signal"
                )
                # Check rebin10 pull sigma if available
                _r10_sigma = None
                if isinstance(spectrum_results, FitResult):
                    _dnl_meta_check = spectrum_results.params.get("_dnl", {})
                _pull_diag_check = spec_dict.get("pull_diagnostics", {})
                _r10_sigma = _pull_diag_check.get("rebin10_pull_sigma")
                if _r10_sigma is not None and _r10_sigma < 2.0:
                    _adequacy["level"] = "inference_ready"
                    _reasons.append(
                        f"rebin10 pull sigma = {_r10_sigma:.2f} < 2.0"
                    )
                else:
                    _adequacy["level"] = "descriptive_only"
                    if _r10_sigma is not None:
                        _reasons.append(
                            f"rebin10 pull sigma = {_r10_sigma:.2f} >= 2.0"
                        )
                    else:
                        _reasons.append(
                            "rebin10 pull sigma not available"
                        )
            elif _cv.verdict == "overfitting":
                # DNL was overfitting, but B1 auto-disabled it and refitted.
                # The refit has independent Poisson bins, so evaluate the
                # refit's diagnostics to decide between descriptive_only and
                # inference_ready.
                _reasons.append(
                    "DNL cross-validation detected overfitting; "
                    "self-estimated DNL auto-disabled and spectrum refitted"
                )
                _dnl_st = (
                    spectrum_results.params.get("dnl_status", "")
                    if isinstance(spectrum_results, FitResult)
                    else ""
                )
                if _dnl_st == "disabled_by_crossval":
                    _reasons.append(
                        "Refit without DNL: bins are independent Poisson"
                    )
                    # Evaluate refit quality via rebin10 pull sigma
                    _pull_diag_refit = spec_dict.get(
                        "pull_diagnostics", {}
                    )
                    _r10_refit = _pull_diag_refit.get(
                        "rebin10_pull_sigma"
                    )
                    # Also check split-half shape stability
                    _sh = spec_dict.get("split_half_validation", {})
                    _sh_pass = _sh.get("pass", True)
                    if (
                        _r10_refit is not None
                        and _r10_refit < 2.0
                        and _sh_pass
                    ):
                        _adequacy["level"] = "inference_ready"
                        _reasons.append(
                            f"rebin10 pull sigma = {_r10_refit:.2f} < 2.0; "
                            "split-half shape stable"
                        )
                    else:
                        _adequacy["level"] = "descriptive_only"
                        if _r10_refit is not None:
                            _reasons.append(
                                f"rebin10 pull sigma = {_r10_refit:.2f}"
                            )
                        if not _sh_pass:
                            _reasons.append(
                                "split-half shape stability failed"
                            )
                else:
                    _adequacy["level"] = "descriptive_only"
                    _reasons.append(
                        "DNL overfitting detected but auto-disable did "
                        "not produce a valid refit"
                    )
            elif _cv.verdict == "mixed":
                _adequacy["level"] = "descriptive_only"
                _reasons.append(
                    "DNL cross-validation is inconclusive (mixed signal)"
                )
            else:
                _reasons.append(f"DNL crossval verdict: {_cv.verdict}")
            _adequacy["reasons"] = _reasons

            # D1: Add split-half interpretation (applies to all DNL
            # crossval paths).  The split-half shape-params-only result
            # should be reported accurately: if max |z| < 2 for shape
            # parameters, the core shape family is stable, not
            # overfitting.
            _sh_d1 = spec_dict.get("split_half_validation", {})
            _sh_max_z = _sh_d1.get("max_z_shape")
            _sh_pass_d1 = _sh_d1.get("pass", True)
            if _sh_max_z is not None:
                if _sh_pass_d1:
                    _adequacy["reasons"].append(
                        f"Core shape family passes split-half stability "
                        f"(max |z| = {_sh_max_z:.2f})"
                    )
                else:
                    _adequacy["reasons"].append(
                        f"Split-half shape instability detected "
                        f"(max |z| = {_sh_max_z:.2f})"
                    )
                    # Downgrade if we were inference_ready
                    if _adequacy["level"] == "inference_ready":
                        _adequacy["level"] = "descriptive_only"

        elif (
            isinstance(spectrum_results, FitResult)
            and spectrum_results.params.get("_dnl", {}).get("dnl_applied")
        ):
            _adequacy["reasons"].append(
                "DNL applied but cross-validation not run"
            )
        else:
            _adequacy["reasons"].append("No DNL correction applied")
            _adequacy["level"] = "inference_ready"

        # Add a recommendation field
        if _adequacy["level"] == "inference_ready":
            _adequacy["recommendation"] = (
                "Peak areas, centroids, and derived activity are "
                "suitable for quantitative inference."
            )
        elif _adequacy["level"] == "descriptive_only":
            _adequacy["recommendation"] = (
                "Model provides a good descriptive fit but has not "
                "been fully calibrated through the DNL/residual "
                "pipeline. Treat quantitative uncertainties as "
                "approximate."
            )
        else:
            _adequacy["recommendation"] = (
                "Statistical claims require DNL-aware residual "
                "calibration (e.g. pseudoexperiments through the "
                "full pipeline)."
            )

        spec_dict["adequacy_verdict"] = _adequacy
        logger.info(
            "Adequacy verdict: %s  - %s",
            _adequacy["level"],
            "; ".join(_adequacy["reasons"]),
        )

        # ── Fit diagnostics (overfitting + validation) ─────────────────
        _pull_diag_result = {}
        _fit_val_result = {}
        _code_diag = {}
        _bias_metrics = {}
        try:
            from plot_utils.diagnostics import (
                compute_pull_diagnostics,
                compute_fit_validation_diagnostics,
                compute_code_domain_diagnostics,
                compute_signed_bias_metrics,
                compute_whitened_residuals,
            )

            _diag_src = (
                spectrum_results.params
                if isinstance(spectrum_results, FitResult)
                else spec_dict
            )

            # ── Overfitting diagnostics (pull structure) ──────────────
            _pull_diag_result = compute_pull_diagnostics(_diag_src)
            if _pull_diag_result:
                spec_dict["pull_diagnostics"] = _pull_diag_result

            # ── Fit-validation diagnostics (model adequacy) ───────────
            _fit_val_result = compute_fit_validation_diagnostics(_diag_src)
            if _fit_val_result:
                spec_dict["fit_validation"] = _fit_val_result

            # ── Code-domain diagnostics (DNL-specific) ────────────────
            _code_diag = compute_code_domain_diagnostics(_diag_src)
            if _code_diag:
                _code_serial = {
                    k: v for k, v in _code_diag.items()
                    if k != "_arrays"
                }
                spec_dict["code_domain_diagnostics"] = _code_serial

            # ── Signed-bias metrics per peak window ───────────────────
            _bias_metrics = compute_signed_bias_metrics(_diag_src)
            if _bias_metrics:
                spec_dict["signed_bias_metrics"] = _bias_metrics

            # ── Covariance-aware whitened residuals ────────────────────
            # Only computed when self-estimated bandpass DNL is active
            _whitened = compute_whitened_residuals(_diag_src)
            if _whitened:
                spec_dict["whitened_residuals"] = _whitened
        except Exception as _e:
            logger.warning("Pull diagnostics failed: %s", _e)

        # ── C4: Pseudoexperiment threshold calibration ────────────────
        _n_pseudo = int(
            cfg.get("spectral_fit", {}).get("pseudoexperiment_trials", 0)
        )
        if _n_pseudo > 0 and isinstance(spectrum_results, FitResult):
            try:
                from src.rmtest.spectral.pseudoexperiments import (
                    run_pseudoexperiment_calibration,
                )
                _pseudo_result = run_pseudoexperiment_calibration(
                    fit_params=spectrum_results.params,
                    fit_kwargs=fit_kwargs,
                    cfg=cfg,
                    n_trials=_n_pseudo,
                )
                if _pseudo_result:
                    spec_dict["pseudoexperiment_calibration"] = _pseudo_result
            except Exception as _pe:
                logger.warning("Pseudoexperiment calibration failed: %s", _pe)

        time_fit_serializable = {}
        for iso, fit in time_fit_results.items():
            if isinstance(fit, FitResult):
                d = dict(fit.params)
                d["cov"] = fit.cov.tolist() if fit.cov is not None else None
                d["ndf"] = fit.ndf
            elif isinstance(fit, dict):
                d = fit
            else:
                d = {}
            meta = time_fit_background_meta.get(iso)
            if meta:
                d = dict(d)
                d["background_mode"] = meta.get("mode")
                if meta.get("baseline_rate_Bq") is not None:
                    d["baseline_rate_Bq"] = meta["baseline_rate_Bq"]
                if meta.get("baseline_unc_Bq") is not None:
                    d["baseline_unc_Bq"] = meta["baseline_unc_Bq"]
            time_fit_serializable[iso] = d
    
        baseline_handling.apply_time_fit_provenance(
            time_fit_serializable,
            baseline_background_provenance,
            baseline_record,
        )
    
        if isinstance(cal_params, dict):
            cal_summary = cal_params
        else:
            cal_summary = {
                "coeffs": list(cal_params.coeffs),
                "covariance": np.asarray(cal_params.cov).tolist(),
                "sigma_E": cal_params.sigma_E,
                "sigma_E_error": cal_params.sigma_E_error,
                "peaks": cal_params.peaks,
            }
    
        summary = Summary(
            timestamp=now_str,
            config_used=args.config.name,
            config_sha256=cfg_sha256,
            calibration=cal_summary,
            calibration_valid=calibration_valid,
            spectral_fit=spec_dict,
            time_fit=time_fit_serializable,
            systematics=systematics_results,
            baseline=baseline_info,
            radon_results=radon_results,
            noise_cut={"removed_events": int(n_removed_noise)},
            burst_filter={
                "removed_events": int(n_removed_burst),
                "burst_mode": burst_mode,
                "roi_diff": roi_diff,
                "sensitivity_scan": {
                    "grid": {f"{m}_{w}": v for (m, w), v in scan_results.items()},
                    "best": {
                        "burst_multiplier": best_params[0] if best_params else None,
                        "burst_window_size_s": best_params[1] if best_params else None,
                    },
                },
            },
            adc_drift_rate=drift_rate,
            adc_drift_mode=drift_mode,
            adc_drift_params=drift_params,
            efficiency=efficiency_results,
            random_seed=seed_used,
            git_commit=commit,
            requirements_sha256=requirements_sha256,
            cli_sha256=cli_sha256,
            cli_args=cli_args,
            analysis={
                "analysis_start_time": t0_cfg,
                "analysis_end_time": t_end_cfg,
                "spike_start_time": spike_start_cfg,
                "spike_end_time": spike_end_cfg,
                "spike_periods": spike_periods_cfg,
                "run_periods": run_periods_cfg,
                "radon_interval": radon_interval_cfg,
                "background_model": cfg.get("analysis", {}).get("background_model"),
                "likelihood": cfg.get("analysis", {}).get("likelihood"),
                "ambient_concentration": cfg.get("analysis", {}).get(
                    "ambient_concentration"
                ),
                "settle_s": cfg.get("analysis", {}).get("settle_s"),
            },
        )
    
        if radon_combined_info is not None:
            summary.radon_combined = radon_combined_info
    
        from radon_joint_estimator import estimate_radon_activity
    
        rate214_final = corrected_rates.get("Po214") if corrected_rates else None
        err214_final = corrected_unc.get("Po214") if corrected_unc else None
        rate218_final = corrected_rates.get("Po218") if corrected_rates else None
        err218_final = corrected_unc.get("Po218") if corrected_unc else None
    
        if rate214_final is None and fit214:
            rate214_final = fit214.rate
            err214_final = fit214.err
        if rate218_final is None and fit218:
            rate218_final = fit218.rate
            err218_final = fit218.err
    
        radon = estimate_radon_activity(
            N218=fit218.counts if fit218 else 0,
            epsilon218=(
                _resolved_efficiency(cfg, "Po218", fit218.params) if fit218 else 1.0
            ),
            N214=fit214.counts if fit214 else 0,
            epsilon214=(
                _resolved_efficiency(cfg, "Po214", fit214.params) if fit214 else 1.0
            ),
            f218=1.0,
            f214=1.0,
            live_time218_s=iso_live_time.get("Po218") if fit218 else None,
            live_time214_s=iso_live_time.get("Po214") if fit214 else None,
            rate214=rate214_final,
            err214=err214_final,
            rate218=rate218_final,
            err218=err218_final,
            analysis_isotope=iso_mode,
        )
    
        # ── Construct a minimal time-series aligned with the measurement window ──
        ts_start, ts_end = _radon_time_window(t0_cfg, t_end_cfg, radon_interval_cfg)
        ts_points: list[float]
        if not math.isfinite(ts_start):
            ts_start = to_utc_datetime(t0_cfg).timestamp()
        if math.isfinite(ts_end) and ts_end > ts_start:
            ts_points = [float(ts_start), float(ts_end)]
        else:
            ts_points = [float(ts_start)]
    
        gaussian_ok = bool(radon.get("gaussian_uncertainty_valid", True))
        stat_unc = radon.get("stat_unc_Bq") if gaussian_ok else float("nan")
        try:
            stat_unc_val = float(stat_unc) if stat_unc is not None else float("nan")
        except (TypeError, ValueError):
            stat_unc_val = float("nan")
        if not math.isfinite(stat_unc_val) or stat_unc_val < 0:
            stat_unc_val = float("nan")
        errors_ts = [stat_unc_val] * len(ts_points)
    
        radon["time_series"] = {
            "time": ts_points,
            "activity": [float(radon["Rn_activity_Bq"])] * len(ts_points),
        }
        radon["time_series"]["error"] = errors_ts
    
        total_vals, total_errs = _total_radon_series(
            radon["time_series"]["activity"],
            radon["time_series"].get("error"),
            monitor_vol,
            sample_vol,
        )
        total_ts = {
            "time": list(radon["time_series"]["time"]),
            "activity": total_vals.tolist(),
        }
        if total_errs is not None:
            total_ts["error"] = total_errs.tolist()
        radon["total_time_series"] = total_ts
    
        summary["radon"] = radon
    
        if weights is not None:
            summary.efficiency = summary.efficiency or {}
            summary.efficiency["blue_weights"] = list(weights)
    
        summary.diagnostics = build_diagnostics(
            summary, spectrum_results, time_fit_results, df_analysis, cfg
        )
    
        drift_flag, drift_msg = baseline_handling.assess_baseline_drift(
            baseline_record,
            cal_result,
            cfg.get("baseline", {}),
        )
        if drift_flag:
            summary.diagnostics = summary.diagnostics or {}
            summary.diagnostics["baseline_compat_warning"] = True
            if drift_msg:
                warn_list = summary.diagnostics.setdefault("warnings", [])
                if drift_msg not in warn_list:
                    warn_list.append(drift_msg)
    
        results_dir = Path(args.output_dir) / (args.job_id or now_str)
        if results_dir.exists():
            if args.overwrite:
                shutil.rmtree(results_dir)
            else:
                raise FileExistsError(f"Results folder already exists: {results_dir}")
    
        copy_config(results_dir, cfg, exist_ok=args.overwrite)
        out_dir = Path(write_summary(results_dir, summary))
        out_dir.mkdir(parents=True, exist_ok=True)

        radon_background_mode = locals().get("background_mode")
        if (
            iso_mode == "radon"
            and radon_background_mode is None
            and hasattr(summary, "get")
        ):
            summary_radon = summary.get("radon")
            if isinstance(summary_radon, Mapping):
                plot_payload = summary_radon.get("plot_series")
                if isinstance(plot_payload, Mapping):
                    radon_background_mode = plot_payload.get("background_mode")
    
        if iso_mode == "radon" and "radon" in summary:
            rad_ts = summary["radon"]["time_series"]
    
            plot_radon_activity(
                rad_ts["time"],
                rad_ts["activity"],
                Path(out_dir) / "radon_activity.png",
                rad_ts.get("error"),
                config=cfg.get("plotting", {}),
                sample_volume_l=sample_vol,
                background_mode=radon_background_mode,
            )
            total_vals, total_errs = _total_radon_series(
                rad_ts["activity"],
                rad_ts.get("error"),
                monitor_vol,
                sample_vol,
            )
            plot_total_radon(
                rad_ts["time"],
                total_vals,
                Path(out_dir) / "total_radon.png",
                total_errs,
                config=cfg.get("plotting", {}),
                background_mode=radon_background_mode,
            )
            plot_radon_trend(
                rad_ts["time"],
                rad_ts["activity"],
                Path(out_dir) / "radon_trend.png",
                config=cfg.get("plotting", {}),
                sample_volume_l=sample_vol,
            )
    
        # Generate plots now that the output directory exists
        spectrum_png = Path(out_dir) / "spectrum.png"
        spectrum_components_png = Path(out_dir) / "spectrum_components.png"
        if spec_plot_data:
            try:
                _ = plot_spectrum(
                    energies=spec_plot_data["energies"],
                    fit_vals=spec_plot_data["fit_vals"],
                    out_png=spectrum_png,
                    bins=spec_plot_data["bins"],
                    bin_edges=spec_plot_data["bin_edges"],
                    config=cfg.get("plotting", {}),
                    fit_flags=spec_plot_data.get("flags"),
                )
            except Exception as e:
                logger.warning("Could not create spectrum plot: %s", e)
    
            try:
                _ = plot_spectrum(
                    energies=spec_plot_data["energies"],
                    fit_vals=spec_plot_data["fit_vals"],
                    out_png=spectrum_components_png,
                    bins=spec_plot_data["bins"],
                    bin_edges=spec_plot_data["bin_edges"],
                    config=cfg.get("plotting", {}),
                    fit_flags=spec_plot_data.get("flags"),
                    show_total_model=False,
                )
            except Exception as e:
                logger.warning("Could not create component spectrum plot: %s", e)

            # --- Per-stage spectrum plots (prelim full-res + stage 2) ---
            # Stage 0: prelim fit at bin_width=1 (full resolution)
            if _pre_info and _pre_info.get("prelim_plot_data"):
                try:
                    _s0pd = _pre_info["prelim_plot_data"]
                    _s0_png = Path(out_dir) / "spectrum_stage0_fullres.png"
                    _ = plot_spectrum(
                        energies=spec_plot_data["energies"],
                        fit_vals=_s0pd["fit_vals"],
                        out_png=_s0_png,
                        bins=_s0pd["bins"],
                        bin_edges=_s0pd["bin_edges"],
                        config=cfg.get("plotting", {}),
                        fit_flags=_s0pd.get("flags"),
                    )
                    logger.info("Saved stage 0 (full-res) spectrum plot to %s", _s0_png)
                except Exception as e:
                    logger.warning("Could not create stage 0 spectrum plot: %s", e)

            # Stage 2: main fit at moderate rebin (before 3-stage override)
            if _stage2_plot_data is not None:
                try:
                    _s2_png = Path(out_dir) / "spectrum_stage2_rebin.png"
                    _ = plot_spectrum(
                        energies=spec_plot_data["energies"],
                        fit_vals=_stage2_plot_data["fit_vals"],
                        out_png=_s2_png,
                        bins=_stage2_plot_data["bins"],
                        bin_edges=_stage2_plot_data["bin_edges"],
                        config=cfg.get("plotting", {}),
                        fit_flags=_stage2_plot_data.get("flags"),
                    )
                    logger.info("Saved stage 2 (moderate rebin) spectrum plot to %s", _s2_png)
                except Exception as e:
                    logger.warning("Could not create stage 2 spectrum plot: %s", e)

            # --- DNL-corrected spectrum plot ---
            try:
                _dnl_corr_png = out_dir / "spectrum_dnl_corrected.png"
                plot_spectrum_dnl_corrected(
                    fit_vals=spec_plot_data["fit_vals"],
                    out_png=str(_dnl_corr_png),
                    config=cfg.get("plotting", {}),
                    fit_flags=spec_plot_data.get("flags"),
                )
            except Exception as e:
                logger.warning("Could not create DNL-corrected spectrum plot: %s", e)

            # --- Fit diagnostic plots ---
            try:
                from plot_utils.diagnostics import (
                    plot_correlation_matrix,
                    plot_pull_histogram,
                    plot_parameter_summary,
                    plot_split_half_comparison,
                    plot_overfitting_diagnostics,
                    plot_model_comparison,
                )
                _diag_fit_result = spec_plot_data["fit_vals"]
                _diag_params = (
                    _diag_fit_result.params
                    if hasattr(_diag_fit_result, "params")
                    else dict(_diag_fit_result)
                )
                plot_correlation_matrix(_diag_fit_result, out_dir)
                plot_pull_histogram(_diag_params, out_dir)
                _diag_minos = getattr(_diag_fit_result, "minos_errors", None)
                plot_parameter_summary(_diag_params, out_dir, minos_errors=_diag_minos)
                if split_half_result is not None:
                    plot_split_half_comparison(split_half_result, out_dir)
                if _pull_diag_result:
                    plot_overfitting_diagnostics(
                        _diag_params, _pull_diag_result, out_dir
                    )
                if model_comparison_result is not None:
                    plot_model_comparison(model_comparison_result, out_dir)
                if dnl_crossval_result is not None:
                    from plot_utils.diagnostics import plot_dnl_crossval
                    plot_dnl_crossval(dnl_crossval_result, out_dir)
                # C1: Code-domain diagnostics plot
                if _code_diag:
                    from plot_utils.diagnostics import (
                        plot_code_domain_diagnostics,
                    )
                    plot_code_domain_diagnostics(_code_diag, out_dir)
            except Exception as e:
                logger.warning("Could not create fit diagnostic plots: %s", e)

        if not spectrum_png.exists():
            try:
                stub_bins = spec_plot_data["bins"] if spec_plot_data else None
                stub_edges = spec_plot_data.get("bin_edges") if spec_plot_data else None
            except Exception:
                stub_bins = None
                stub_edges = None
            try:
                energies_stub = (
                    spec_plot_data.get("energies")
                    if spec_plot_data and isinstance(spec_plot_data, dict)
                    else df_analysis.get("energy_MeV", pd.Series(dtype=float)).to_numpy()
                )
            except Exception:
                energies_stub = (
                    df_analysis.get("energy_MeV", pd.Series(dtype=float)).to_numpy()
                    if isinstance(df_analysis, pd.DataFrame)
                    else np.asarray([], dtype=float)
                )
            try:
                _save_stub_spectrum_plot(
                    energies_stub,
                    spectrum_png,
                    bins=stub_bins,
                    bin_edges=stub_edges,
                    config=cfg.get("plotting", {}),
                )
                logger.info(
                    "Saved fallback spectrum plot after unavailable fit to %s", spectrum_png
                )
            except Exception as e:
                logger.warning("Could not create fallback spectrum plot: %s", e)
    
            try:
                _save_stub_spectrum_plot(
                    energies_stub,
                    spectrum_components_png,
                    bins=stub_bins,
                    bin_edges=stub_edges,
                    config=cfg.get("plotting", {}),
                )
                logger.info(
                    "Saved fallback component spectrum plot after unavailable fit to %s",
                    spectrum_components_png,
                )
            except Exception as e:
                logger.warning(
                    "Could not create fallback component spectrum plot: %s", e
                )
    
        try:
            _ = plot_spectrum_comparison(
                pre_spec_energies,
                post_spec_energies,
                bins=spec_plot_data.get("bins", 400) if spec_plot_data else 400,
                bin_edges=spec_plot_data.get("bin_edges") if spec_plot_data else None,
                out_png=Path(out_dir) / "spectrum_pre_post.png",
                config=cfg.get("time_fit", {}),
            )
        except Exception as e:
            logger.warning("Could not create pre/post spectrum plot -> %s", e)

        if args.burst_sensitivity_scan and scan_results:
            try:
                plot_activity_grid(
                    scan_results,
                    out_png=Path(out_dir) / "burst_scan.png",
                    config=cfg.get("plotting", {}),
                )
            except Exception as e:
                logger.warning("Could not create burst scan plot -> %s", e)
    
        overlay = cfg.get("plotting", {}).get("overlay_isotopes", False)
        save_individual_ts = bool(
            cfg.get("plotting", {}).get("save_individual_time_series", False)
        )
        isotope_series_data: dict[str, list[dict[str, float]]] = {}

        if overlay:
            # Guarantee plotting payloads exist for both daughter isotopes even
            # when the fit skipped one of them so the overlay consistently
            # generates Po-214 and Po-218 images.
            tf_cfg = cfg.get("time_fit", {})
            ts_start = to_datetime_utc(t0_global)
            for iso in ("Po218", "Po214"):
                if iso in time_plot_data:
                    continue

                win_key = f"window_{iso.lower()}"
                win_range = tf_cfg.get(win_key) if isinstance(tf_cfg, Mapping) else None
                if win_range is None:
                    continue

                lo, hi = win_range
                try:
                    mask = (
                        (df_analysis["energy_MeV"] >= lo)
                        & (df_analysis["energy_MeV"] <= hi)
                        & (df_analysis["timestamp"] >= ts_start)
                        & (df_analysis["timestamp"] <= t_end_global)
                    )
                    iso_events = df_analysis.loc[mask]
                except Exception:
                    continue

                time_plot_data[iso] = {
                    "events_times": iso_events["timestamp"].values,
                    "events_energy": iso_events["energy_MeV"].values,
                }

        def _prepare_model_errors(
            ts_times: np.ndarray, plot_cfg: Mapping[str, Any], iso_names: list[str]
        ) -> dict[str, np.ndarray]:
            """Return model uncertainty arrays for the requested isotopes."""

            centers, widths = _ts_bin_centers_widths(
                ts_times, plot_cfg, t0_global.timestamp(), t_end_global_ts
            )
            normalise = bool(plot_cfg.get("plot_time_normalise_rate", False))
            model_errs: dict[str, np.ndarray] = {}
            for iso_key in iso_names:
                sigma_arr = _model_uncertainty(
                    centers,
                    widths,
                    time_fit_results.get(iso_key),
                    iso_key,
                    plot_cfg,
                    normalise,
                )
                if sigma_arr is not None:
                    model_errs[iso_key] = sigma_arr
            return model_errs

        for iso in ("Po218", "Po214", "Po210", "Po212"):
            pdata = time_plot_data.get(iso)
            if pdata is None:
                continue
            try:
                plot_cfg = dict(cfg.get("time_fit", {}))
                plot_cfg.update(cfg.get("plotting", {}))
                if run_periods_cfg:
                    plot_cfg["run_periods"] = run_periods_cfg
                # Per-isotope files always show only this isotope
                for other_iso in ("Po214", "Po218", "Po210", "Po212"):
                    if other_iso != iso:
                        plot_cfg[f"window_{other_iso.lower()}"] = None
                if cal_window_rel_unc:
                    plot_cfg["cal_window_rel_unc_per_iso"] = cal_window_rel_unc
                ts_times = pdata["events_times"]
                ts_energy = pdata["events_energy"]
                fit_obj = time_fit_results.get(iso)
                fit_dict = _fit_params(fit_obj)

                model_errs = _prepare_model_errors(ts_times, plot_cfg, [iso])

                ts_info = plot_time_series(
                    all_timestamps=ts_times,
                    all_energies=ts_energy,
                    fit_results=fit_dict,
                    t_start=t0_global.timestamp(),
                    t_end=t_end_global_ts,
                    config=plot_cfg,
                    out_png=Path(out_dir) / f"time_series_{iso}.png",
                    model_errors=model_errs,
                )
                if ts_info:
                    series_map = _segments_to_isotope_series(ts_info)
                    for iso_key, entries in series_map.items():
                        if not entries:
                            continue
                        existing = isotope_series_data.setdefault(iso_key, [])
                        existing.extend(entries)

            except Exception as e:
                logger.warning("Could not create time-series plot for %s -> %s", iso, e)

        # Combined overlay plot showing all isotopes together
        if overlay:
            try:
                overlay_cfg = dict(cfg.get("time_fit", {}))
                overlay_cfg.update(cfg.get("plotting", {}))
                if run_periods_cfg:
                    overlay_cfg["run_periods"] = run_periods_cfg
                if cal_window_rel_unc:
                    overlay_cfg["cal_window_rel_unc_per_iso"] = cal_window_rel_unc
                overlay_times = df_analysis["timestamp"].values
                overlay_energy = df_analysis["energy_MeV"].values
                overlay_fit_dict = {}
                for k in ("Po214", "Po218", "Po210"):
                    obj = time_fit_results.get(k)
                    if obj:
                        overlay_fit_dict.update(_fit_params(obj))
                overlay_iso_list = [
                    i for i in ("Po214", "Po218", "Po210") if time_fit_results.get(i)
                ]
                overlay_errs = _prepare_model_errors(
                    overlay_times, overlay_cfg, overlay_iso_list
                )
                _ = plot_time_series(
                    all_timestamps=overlay_times,
                    all_energies=overlay_energy,
                    fit_results=overlay_fit_dict,
                    t_start=t0_global.timestamp(),
                    t_end=t_end_global_ts,
                    config=overlay_cfg,
                    out_png=Path(out_dir) / "isotope_time_series.png",
                    model_errors=overlay_errs,
                )
            except Exception as e:
                logger.warning("Could not create overlay time-series plot -> %s", e)
    
        for iso_entries in isotope_series_data.values():
            iso_entries.sort(key=lambda row: row.get("t", 0.0))
    
        # Deduplicate isotope series data to prevent duplicate bins when overlay_isotopes is enabled
        isotope_series_data = dedupe_isotope_series(isotope_series_data)
    
        radon_inference_results = None
        radon_inference_cfg = cfg.get("radon_inference")
        if isotope_series_data and isinstance(radon_inference_cfg, Mapping):
            timestamps_for_external = sorted(
                {
                    float(entry.get("t"))
                    for entries in isotope_series_data.values()
                    for entry in entries
                    if entry.get("t") is not None
                }
            )
            external_series = None
            if timestamps_for_external and radon_inference_cfg.get("enabled", False):
                try:
                    raw_external = load_external_rn_series(
                        radon_inference_cfg.get("external_rn"), timestamps_for_external
                    )
                except Exception as exc:
                    logger.warning("Failed to load external radon series: %s", exc)
                    raw_external = []
    
                if raw_external:
                    external_series = []
                    for ts_obj, value in raw_external:
                        try:
                            if hasattr(ts_obj, "timestamp"):
                                t_val = float(ts_obj.timestamp())
                            else:
                                t_val = float(ts_obj)
                            val_float = float(value)
                        except (TypeError, ValueError):
                            continue
                        if not np.isfinite(t_val) or not np.isfinite(val_float):
                            continue
                        external_series.append({"t": t_val, "rn_bq_per_m3": val_float})
    
            radon_inference_results = run_radon_inference(
                isotope_series_data,
                cfg,
                external_series,
            )
            if radon_inference_results:
                summary["radon_inference"] = radon_inference_results
                try:
                    plot_rn_inferred_vs_time(radon_inference_results, Path(out_dir))
                    plot_ambient_rn_vs_time(radon_inference_results, Path(out_dir))
                    plot_volume_equiv_vs_time(radon_inference_results, Path(out_dir))
                except Exception as exc:
                    logger.warning("Failed to create radon inference plots: %s", exc)
                # Henry's law derived plots: radon-in-liquid and argon-from-leak
                try:
                    from radon.radon_plots import plot_radon_in_liquid, plot_argon_from_leak
                    plot_radon_in_liquid(radon_inference_results, Path(out_dir), cfg=cfg)
                    plot_argon_from_leak(radon_inference_results, Path(out_dir), cfg=cfg)
                except Exception as exc:
                    logger.warning("Failed to create Henry's law plots: %s", exc)

                # Keep ``radon_trend.png`` sourced from the radon concentration
                # series. ``rn_inferred`` is detector-cell activity in Bq and is
                # reported separately in ``radon_inferred.png``.

    # ── Lucas-cell assay bridge ──────────────────────────────────────
    with timer.section("lucas_bridge"):
        bridge_cfg = cfg.get("lucas_bridge")
        if isinstance(bridge_cfg, Mapping) and bridge_cfg.get("enabled", False):
            try:
                from assay_bridge import compute_bridge, get_bridge_detection_efficiency

                bridge_files = bridge_cfg.get("assay_files", [])
                bridge_results = compute_bridge(
                    bridge_files, summary, cfg,
                    isotope_series=isotope_series_data or None,
                    cal_window_rel_unc=cal_window_rel_unc or None,
                )
                if bridge_results:
                    summary.lucas_bridge = bridge_results
                    logger.info(
                        "Lucas bridge: %d assays processed",
                        bridge_results.get("n_assays", 0),
                    )

                    # ── Feed bridge efficiency back into radon inference ──
                    bridge_eff = get_bridge_detection_efficiency(bridge_results)
                    if bridge_eff and isotope_series_data:
                        # Separate the relative uncertainty from the isotope values
                        bridge_eff_rel_unc = bridge_eff.pop("rel_unc", 0.0)
                        radon_inference_cfg = cfg.get("radon_inference")
                        if isinstance(radon_inference_cfg, Mapping) and radon_inference_cfg.get("enabled", False):
                            # Store original values for reference
                            original_det_eff = dict(radon_inference_cfg.get("detection_efficiency", {}))
                            summary.lucas_bridge["original_detection_efficiency"] = original_det_eff
                            summary.lucas_bridge["bridge_eff_rel_unc"] = bridge_eff_rel_unc

                            # Override with bridge-derived values
                            radon_inference_cfg["detection_efficiency"] = bridge_eff
                            radon_inference_cfg["detection_efficiency_rel_unc"] = bridge_eff_rel_unc
                            logger.info(
                                "Re-running radon inference with bridge-derived "
                                "detection efficiency: %s (rel unc %.1f%%) (was: %s)",
                                bridge_eff, bridge_eff_rel_unc * 100, original_det_eff,
                            )

                            # external_series may not be in scope if inference didn't run
                            try:
                                _ext_series = external_series
                            except NameError:
                                _ext_series = None
                            try:
                                radon_inference_results_bridge = run_radon_inference(
                                    isotope_series_data,
                                    cfg,
                                    _ext_series,
                                )
                                if radon_inference_results_bridge:
                                    summary["radon_inference"] = radon_inference_results_bridge
                                    try:
                                        plot_rn_inferred_vs_time(radon_inference_results_bridge, Path(out_dir))
                                        plot_ambient_rn_vs_time(radon_inference_results_bridge, Path(out_dir))
                                        plot_volume_equiv_vs_time(radon_inference_results_bridge, Path(out_dir))
                                    except Exception as exc:
                                        logger.warning(
                                            "Failed to re-plot radon inference with bridge eff: %s", exc
                                        )
                                    try:
                                        from radon.radon_plots import plot_radon_in_liquid, plot_argon_from_leak
                                        plot_radon_in_liquid(radon_inference_results_bridge, Path(out_dir), cfg=cfg)
                                        plot_argon_from_leak(radon_inference_results_bridge, Path(out_dir), cfg=cfg)
                                    except Exception as exc:
                                        logger.warning(
                                            "Failed to re-plot Henry's law with bridge eff: %s", exc
                                        )
                            except Exception as exc:
                                logger.warning(
                                    "Failed to re-run radon inference with bridge eff: %s", exc
                                )
                                # Restore original values
                                radon_inference_cfg["detection_efficiency"] = original_det_eff

                    # ── Spike decay fitting (independent efficiency) ──
                    spike_fit_results = None
                    try:
                        from assay_bridge import fit_spike_periods
                        spike_fit_results = fit_spike_periods(
                            isotope_series_data or {}, cfg,
                            assay_results=summary.lucas_bridge.get("assays"),
                        )
                        if spike_fit_results.get("periods"):
                            summary.lucas_bridge["spike_fits"] = spike_fit_results
                            for sp in spike_fit_results["periods"]:
                                if sp.get("error"):
                                    logger.warning(
                                        "Spike fit %s: %s",
                                        sp.get("label", "?"), sp["error"],
                                    )
                            # Plot spike decay fits
                            try:
                                from plot_utils.diagnostics import plot_spike_decay_fits
                                plot_spike_decay_fits(
                                    spike_fit_results, isotope_series_data or {},
                                    out_dir,
                                )
                            except Exception as exc:
                                logger.warning("Failed to plot spike decay: %s", exc)
                    except Exception as exc:
                        logger.warning("Spike decay fitting failed: %s", exc)

                    # ── Bridge summary plot (after spike so we can overlay) ──
                    try:
                        from plot_utils.diagnostics import plot_bridge_summary
                        plot_bridge_summary(bridge_results, out_dir,
                                            spike_results=spike_fit_results)
                    except Exception as exc:
                        logger.warning("Failed to create bridge summary plot: %s", exc)

            except Exception as exc:
                logger.warning("Lucas-cell assay bridge failed: %s", exc)

    # Additional visualizations
    with timer.section("additional_visualizations"):
        if efficiency_results.get("sources"):
            try:
                errs_arr = np.array(
                    [
                        s.get("error", 0.0)
                        for s in efficiency_results["sources"].values()
                    ]
                )
                if errs_arr.size > 0:
                    cov = np.diag(errs_arr**2)
                    cov_heatmap(
                        cov,
                        Path(out_dir) / "eff_cov.png",
                        labels=list(efficiency_results["sources"].keys()),
                    )
                efficiency_bar(
                    efficiency_results,
                    Path(out_dir) / "efficiency.png",
                    config=cfg.get("plotting", {}),
                )
            except Exception as e:
                logger.warning("Could not create efficiency plots -> %s", e)

    # Radon activity and equivalent air plots
    summary_updated = False
    try:
        rad_summary = summary.get("radon", {}) if hasattr(summary, "get") else {}
        if not isinstance(rad_summary, Mapping):
            rad_summary = {}
        rad_ts_data = rad_summary.get("time_series", {}) if hasattr(rad_summary, "get") else {}
        if not isinstance(rad_ts_data, Mapping):
            rad_ts_data = {}

        window_start, window_end = _radon_time_window(
            t0_global, t_end_global_ts, radon_interval
        )
        if not math.isfinite(window_start):
            window_start = t0_global.timestamp()

        if math.isfinite(window_end) and window_end > window_start:
            time_grid = np.linspace(window_start, window_end, 100)
        else:
            time_grid = np.array([float(window_start)], dtype=float)

        fallback_times = np.asarray(rad_ts_data.get("time", []), dtype=float)
        fallback_activity_raw = np.asarray(
            rad_ts_data.get("activity", []), dtype=float
        )
        if fallback_activity_raw.size != fallback_times.size:
            fallback_activity_raw = None

        fallback_errs_raw = rad_ts_data.get("error") if hasattr(rad_ts_data, "get") else None
        fallback_errs_arr = (
            np.asarray(fallback_errs_raw, dtype=float)
            if fallback_errs_raw is not None
            else None
        )
        if fallback_errs_arr is not None and fallback_errs_arr.size != fallback_times.size:
            fallback_errs_arr = None

        try:
            fallback_val = float(rad_summary.get("Rn_activity_Bq", float("nan")))
        except (TypeError, ValueError):
            fallback_val = float("nan")

        stat_unc_val = rad_summary.get("stat_unc_Bq") if hasattr(rad_summary, "get") else None
        try:
            err_val = float(stat_unc_val) if stat_unc_val is not None else float("nan")
        except (TypeError, ValueError):
            err_val = float("nan")

        fallback_fill = fallback_val
        if (not math.isfinite(fallback_fill)) and radon_results:
            try:
                fallback_fill = float(
                    radon_results.get("radon_activity_Bq", {}).get("value", float("nan"))
                )
            except (TypeError, ValueError):
                fallback_fill = float("nan")

        err_fill = err_val
        if (not math.isfinite(err_fill)) and radon_results:
            try:
                err_fill = float(
                    radon_results.get("radon_activity_Bq", {}).get(
                        "uncertainty", float("nan")
                    )
                )
            except (TypeError, ValueError):
                err_fill = float("nan")

        radon_val = None
        radon_unc = None
        if radon_results:
            try:
                radon_val = float(
                    radon_results.get("radon_activity_Bq", {}).get("value", float("nan"))
                )
            except (TypeError, ValueError):
                radon_val = None
            try:
                radon_unc = float(
                    radon_results.get("radon_activity_Bq", {}).get(
                        "uncertainty", float("nan")
                    )
                )
            except (TypeError, ValueError):
                radon_unc = None

        valid_fits: dict[str, Mapping[str, Any]] = {}
        for iso in ("Po214", "Po218"):
            fit_obj = time_fit_results.get(iso)
            if fit_obj is None:
                continue
            fit_params = _fit_params(fit_obj)
            fit_ok = bool(fit_params.get("fit_valid", True))
            iso_flag = f"fit_valid_{iso}"
            if iso_flag in fit_params:
                fit_ok = fit_ok and bool(fit_params.get(iso_flag, True))
            if fit_ok:
                valid_fits[iso] = fit_params

        times_dt = to_datetime_utc(time_grid, unit="s")
        t_rel = (times_dt - analysis_start).total_seconds()
        activity_times = time_grid
        background_mode = _summary_radon_background_mode(
            summary,
            cfg,
            time_fit_results,
        )

        if radon_combined_info is not None:
            try:
                _ = plot_radon_activity(
                    [t0_global.timestamp(), t_end_global_ts],
                    [radon_combined_info["activity_Bq"]] * 2,
                    Path(out_dir) / "radon_activity_combined.png",
                    [radon_combined_info["unc_Bq"]] * 2,
                    config=cfg.get("plotting", {}),
                    sample_volume_l=sample_vol,
                    background_mode=background_mode,
                )
            except Exception as e:
                logger.warning("Could not create radon combined plot -> %s", e)

        A214 = dA214 = None
        if "Po214" in valid_fits:
            fit_result = time_fit_results["Po214"]
            fit = valid_fits["Po214"]
            A214, dA214 = _radon_activity_curve_from_fit(
                "Po214", fit_result, fit, t_rel, cfg
            )
            plot_radon_activity(
                time_grid,
                A214,
                Path(out_dir) / "radon_activity_po214.png",
                dA214,
                config=cfg.get("plotting", {}),
                sample_volume_l=sample_vol,
                background_mode=background_mode,
            )

        A218 = dA218 = None
        if "Po218" in valid_fits:
            fit_result = time_fit_results["Po218"]
            fit = valid_fits["Po218"]
            A218, dA218 = _radon_activity_curve_from_fit(
                "Po218", fit_result, fit, t_rel, cfg
            )

        activity_arr = err_arr = None
        if A214 is None and A218 is None:
            activity_arr = _regrid_series(
                fallback_times,
                fallback_activity_raw,
                activity_times,
                fallback_fill,
            )
            err_arr = _regrid_series(
                fallback_times,
                fallback_errs_arr,
                activity_times,
                err_fill,
            )
        else:
            activity_arr = np.zeros_like(time_grid, dtype=float)
            err_arr = np.zeros_like(time_grid, dtype=float)
            for i in range(time_grid.size):
                r214 = err214_i = None
                if A214 is not None:
                    r214 = A214[i]
                    err214_i = dA214[i]
                r218 = err218_i = None
                if A218 is not None:
                    r218 = A218[i]
                    err218_i = dA218[i]
                A, s = compute_radon_activity(
                    r218,
                    err218_i,
                    1.0,
                    r214,
                    err214_i,
                    1.0,
                )
                activity_arr[i] = A
                err_arr[i] = s

            if radon_val is not None and (
                not np.isfinite(activity_arr).any() or np.all(activity_arr == 0)
            ):
                activity_arr.fill(radon_val)
            if radon_unc is not None and (
                not np.isfinite(err_arr).any() or np.all(err_arr == 0)
            ):
                err_arr.fill(radon_unc)

        if A214 is None and A218 is None:
            if radon_val is not None and (
                not np.isfinite(activity_arr).any() or np.all(activity_arr == 0)
            ):
                activity_arr.fill(radon_val)
            if radon_unc is not None and (
                not np.isfinite(err_arr).any() or np.all(err_arr == 0)
            ):
                err_arr.fill(radon_unc)

        # Add systematic uncertainties in quadrature to radon activity errors.
        if activity_arr is not None and err_arr is not None:
            # 1. Calibration window efficiency systematic (energy scale shift
            #    changes the fraction of events captured in the fixed window).
            cal_rel = 0.0
            if cal_window_rel_unc:
                cal_po214 = cal_window_rel_unc.get("Po214", 0.0)
                cal_po218 = cal_window_rel_unc.get("Po218", 0.0)
                if A214 is not None and A218 is not None:
                    cal_rel = max(cal_po214, cal_po218)
                elif A214 is not None:
                    cal_rel = cal_po214
                elif A218 is not None:
                    cal_rel = cal_po218

            # 2. Bridge detection efficiency systematic (correlated across
            #    isotopes; does not average down).
            try:
                _bridge_rel = float(bridge_eff_rel_unc) if bridge_eff_rel_unc else 0.0
            except (NameError, TypeError):
                _bridge_rel = 0.0

            total_syst_rel2 = cal_rel**2 + _bridge_rel**2
            if total_syst_rel2 > 0:
                err_arr = np.sqrt(err_arr**2 + (activity_arr**2) * total_syst_rel2)

        if activity_arr is not None and err_arr is not None:
            times_list = [float(t) for t in np.asarray(activity_times, dtype=float)]
            plot_series = {
                "time": times_list,
                "activity": np.asarray(activity_arr, dtype=float).tolist(),
                "error": np.asarray(err_arr, dtype=float).tolist(),
            }
            if sample_vol is not None:
                try:
                    plot_series["sample_volume_l"] = float(sample_vol)
                except (TypeError, ValueError):
                    pass

            if background_mode is not None:
                plot_series["background_mode"] = background_mode

            total_vals, total_errs = _total_radon_series(
                activity_arr,
                err_arr,
                monitor_vol,
                sample_vol,
            )
            total_series = {
                "time": list(times_list),
                "activity": total_vals.tolist(),
            }
            if total_errs is not None:
                total_series["error"] = total_errs.tolist()
            if background_mode is not None:
                total_series["background_mode"] = background_mode

            summary_radon = summary.get("radon") if hasattr(summary, "get") else None
            if not isinstance(summary_radon, Mapping):
                summary_radon = {}
            else:
                summary_radon = dict(summary_radon)

            plot_payload = summary_radon.get("plot_series")
            if isinstance(plot_payload, Mapping):
                plot_payload = dict(plot_payload)
            else:
                plot_payload = {}
            plot_payload["time_series"] = plot_series
            plot_payload["total_time_series"] = total_series
            if background_mode is not None:
                plot_payload["background_mode"] = background_mode
            summary_radon["plot_series"] = plot_payload
            summary["radon"] = summary_radon

            plot_radon_activity(
                activity_times,
                activity_arr,
                Path(out_dir) / "radon_activity.png",
                err_arr,
                config=cfg.get("plotting", {}),
                sample_volume_l=sample_vol,
                background_mode=background_mode,
            )
            plot_total_radon(
                activity_times,
                total_vals,
                Path(out_dir) / "total_radon.png",
                total_errs,
                config=cfg.get("plotting", {}),
                background_mode=background_mode,
            )
            plot_radon_trend(
                activity_times,
                activity_arr,
                Path(out_dir) / "radon_trend.png",
                config=cfg.get("plotting", {}),
                sample_volume_l=sample_vol,
            )
            summary_updated = True

        if radon_interval is not None:
            times_trend = np.linspace(
                radon_interval[0].timestamp(),
                radon_interval[1].timestamp(),
                50,
            )
            times_trend_dt = to_datetime_utc(times_trend, unit="s")
            rel_trend = (times_trend_dt - analysis_start).total_seconds()
            A214_tr = None
            if "Po214" in time_fit_results:
                fit_result = time_fit_results["Po214"]
                fit = _fit_params(fit_result)
                if fit.get("fit_valid", True) and fit.get("fit_valid_Po214", True):
                    A214_tr, _ = _radon_activity_curve_from_fit(
                        "Po214", fit_result, fit, rel_trend, cfg
                    )
            A218_tr = None
            if "Po218" in time_fit_results:
                fit_result = time_fit_results["Po218"]
                fit = _fit_params(fit_result)
                if fit.get("fit_valid", True) and fit.get("fit_valid_Po218", True):
                    A218_tr, _ = _radon_activity_curve_from_fit(
                        "Po218", fit_result, fit, rel_trend, cfg
                    )
            if A214_tr is not None or A218_tr is not None:
                trend = np.zeros_like(times_trend)
                for i in range(times_trend.size):
                    r214 = A214_tr[i] if A214_tr is not None else None
                    r218 = A218_tr[i] if A218_tr is not None else None
                    A, _ = compute_radon_activity(r218, None, 1.0, r214, None, 1.0)
                    trend[i] = A
                plot_radon_trend(
                    times_trend,
                    trend,
                    Path(out_dir) / "radon_trend.png",
                    config=cfg.get("plotting", {}),
                    sample_volume_l=sample_vol,
                )

        ambient = cfg.get("analysis", {}).get("ambient_concentration")
        ambient_interp_m3 = None
        if args.ambient_file:
            try:
                dat = np.loadtxt(args.ambient_file, usecols=(0, 1))
                ambient_interp_m3 = np.interp(activity_times, dat[:, 0], dat[:, 1])
            except Exception as e:
                logger.warning(
                    "Could not read ambient file '%s': %s", args.ambient_file, e
                )

        if ambient_interp_m3 is not None:
            vol_arr = activity_arr / ambient_interp_m3
            vol_err = err_arr / ambient_interp_m3
            plot_equivalent_air(
                activity_times,
                vol_arr,
                vol_err,
                None,
                Path(out_dir) / "equivalent_air.png",
                config=cfg.get("plotting", {}),
            )
            if A214 is not None:
                plot_equivalent_air(
                    time_grid,
                    A214 / ambient_interp_m3,
                    dA214 / ambient_interp_m3,
                    None,
                    Path(out_dir) / "equivalent_air_po214.png",
                    config=cfg.get("plotting", {}),
                )
        elif ambient:
            vol_arr = activity_arr / float(ambient)
            vol_err = err_arr / float(ambient)
            plot_equivalent_air(
                activity_times,
                vol_arr,
                vol_err,
                float(ambient),
                Path(out_dir) / "equivalent_air.png",
                config=cfg.get("plotting", {}),
            )
            if A214 is not None:
                plot_equivalent_air(
                    time_grid,
                    A214 / float(ambient),
                    dA214 / float(ambient),
                    float(ambient),
                    Path(out_dir) / "equivalent_air_po214.png",
                    config=cfg.get("plotting", {}),
                )
    except Exception as e:
        logger.warning("Could not create radon activity plots -> %s", e)

    if summary_updated:
        try:
            write_summary(out_dir, summary)
        except Exception as e:
            logger.warning(
                "Could not refresh summary JSON with radon plot series -> %s", e
            )

    if args.hierarchical_summary:
        try:
            run_results = []
            for p in args.output_dir.glob("*/summary.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        dat = json.load(f)
                except Exception as e:
                    logger.warning("Skipping %s -> %s", p, e)
                    continue
                hl = dat.get("half_life")
                dhl = dat.get("dhalf_life")
                cal = dat.get("calibration", {})
                slope, dslope = cal.get("a", (None, None))
                intercept, dintercept = cal.get("c", (None, None))
                if hl is None or not np.isfinite(hl):
                    logger.warning("Skipping %s -> invalid half-life", p)
                    continue
                if dhl is None or not np.isfinite(dhl) or dhl < 0:
                    logger.warning("Skipping %s -> missing half-life uncertainty", p)
                    continue

                entry = {
                    "half_life": float(hl),
                    "dhalf_life": float(dhl),
                }

                if (
                    slope is not None
                    and dslope is not None
                    and np.isfinite(slope)
                    and np.isfinite(dslope)
                ):
                    entry["slope_MeV_per_ch"] = float(slope)
                    entry["dslope"] = float(dslope)

                if (
                    intercept is not None
                    and dintercept is not None
                    and np.isfinite(intercept)
                    and np.isfinite(dintercept)
                ):
                    entry["intercept"] = float(intercept)
                    entry["dintercept"] = float(dintercept)

                run_results.append(entry)
            if run_results:
                hier_out = fit_hierarchical_runs(run_results)
                with open(args.hierarchical_summary, "w", encoding="utf-8") as f:
                    json.dump(hier_out, f, indent=4)
        except Exception as e:
            logger.warning("hierarchical fit failed -> %s", e)

    timer.report()
    logger.info("Analysis complete. Results written to -> %s", out_dir)


if __name__ == "__main__":
    main()

