#!/usr/bin/env python3
"""
analyze.py

Full Radon Monitor Analysis Pipeline
====================================

Usage:
    python analyze.py \
        --config   config.yaml \
        [--input   merged_output.csv] \
        --output_dir  results \
        [--baseline_range ISO_START ISO_END]

This script performs the following steps:

  1. Load configuration (YAML or JSON).
  2. Load “merged” CSV of event data (timestamps, ADC, etc.).
  3. Perform energy calibration (either two‐point or auto, per config).
     -> Append `energy_MeV` to every event.
  4. (Optional) Extract a “baseline” interval for background estimation.
  5. (Optional) Spectral fit (Po‐210, Po‐218, Po‐214) using unbinned likelihood.
     -> Can bin either in “1 ADC‐channel per bin” or Freedman‐Diaconis (per config).
     -> Uses EMG tails for Po‐210/Po‐218 if requested.
     -> Overlays fit on the spectrum plot.
  6. Time‐series decay fit (Po‐218 and Po‐214 separately).
     -> Extract events in each isotope’s energy window.
     -> Subtract global t₀ so that model always starts at t=0.
     -> Fit unbinned decay (with efficiency, background, N₀, half‐life priors).
     -> Overlay fit curve on a time‐binned histogram (default 1 h bins), at 95% CL.
  7. (Optional) Systematics scan around user‐specified σ shifts.
  8. Produce a single JSON summary with:
       • calibration parameters
       • spectral‐fit outputs
       • time‐fit outputs (per isotope)
       • systematics deltas (if requested)
       • baseline info (if provided)
  9. Save plots (spectrum.png, time_series_Po214.png, time_series_Po218.png) under `output_dir/<timestamp>/`.

To run (without baseline) for a single merged CSV:

    python analyze.py \
       --config    config.yaml \
       --output_dir  results
"""


import argparse
import sys
import logging
import random
from datetime import datetime, timezone, timedelta
import subprocess
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping, cast

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from dateutil.tz import UTC, gettz

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
from calibration import (
    derive_calibration_constants,
    derive_calibration_constants_auto,
    apply_calibration,
)

from fitting import fit_spectrum, fit_time_series, FitResult, FitParams

from constants import (
    DEFAULT_NOISE_CUTOFF,
    PO210,
    PO214,
    PO218,
    DEFAULT_ADC_CENTROIDS,
)

NUCLIDES = {
    "Po210": PO210,
    "Po214": PO214,
    "Po218": PO218,
}


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



def _eff_prior(eff_cfg: Any) -> tuple[float, float]:
    """Return efficiency prior ``(mean, sigma)`` from configuration.

    ``None`` or the string ``"null"`` yields a flat prior ``(1.0, 1e6)``.
    Lists or tuples are returned as-is. Numeric values get a 5 % width.
    """
    if eff_cfg in (None, "null"):
        return (1.0, 1e6)
    if isinstance(eff_cfg, (list, tuple)):
        return tuple(eff_cfg)
    val = float(eff_cfg)
    return (val, 0.05 * val)

from plot_utils import (
    plot_spectrum,
    plot_time_series,
    plot_equivalent_air,
    plot_radon_activity_full,
    plot_radon_trend_full,
)

from plot_utils.radon import (
    plot_radon_activity as _plot_radon_activity,
    plot_radon_trend as _plot_radon_trend,
)


def plot_radon_activity(ts_dict, outdir, maybe_outdir=None, *_, **__):
    """Compatibility wrapper for tests expecting three arguments."""
    target = maybe_outdir or outdir
    Path(target).mkdir(parents=True, exist_ok=True)
    return _plot_radon_activity(ts_dict, target)


def plot_radon_trend(ts_dict, outdir, maybe_outdir=None, *_, **__):
    """Compatibility wrapper for tests expecting three arguments."""
    target = maybe_outdir or outdir
    Path(target).mkdir(parents=True, exist_ok=True)
    return _plot_radon_trend(ts_dict, target)

  
from systematics import scan_systematics, apply_linear_adc_shift
from visualize import cov_heatmap, efficiency_bar
from utils import (
    find_adc_bin_peaks,
    adc_hist_edges,
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


def plot_radon_activity(times, activity, out_png, errors=None, *, config=None):
    """Wrapper used by tests expecting output path as third argument."""
    return plot_radon_activity_full(times, activity, errors, out_png, config=config)


def plot_radon_trend(times, activity, out_png, *, config=None):
    """Wrapper used by tests expecting output path as third argument."""
    return plot_radon_trend_full(times, activity, out_png, config=config)


def _fit_params(obj: FitResult | Mapping[str, float] | None) -> FitParams:
    """Return fit parameters mapping from a ``FitResult`` or dictionary."""
    if isinstance(obj, FitResult):
        return cast(FitParams, obj.params)
    if isinstance(obj, Mapping):
        return obj  # type: ignore[return-value]
    return {}


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


def _ensure_events(events: pd.DataFrame, stage: str) -> None:
    """Exit if ``events`` is empty, printing a helpful message."""
    if len(events) == 0:
        print(f"No events remaining after {stage}. Exiting.")
        sys.exit(1)


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
    lo_val = float(lo) if np.isscalar(lo) else float(lo)
    hi_val = float(hi) if np.isscalar(hi) else float(hi)

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
    times_rel = arr - float(t_start)
    bin_mode = str(
        cfg.get("plot_time_binning_mode", cfg.get("time_bin_mode", "fixed"))
    ).lower()
    if bin_mode in ("fd", "auto"):
        data = times_rel[(times_rel >= 0) & (times_rel <= (t_end - t_start))]
        if len(data) < 2:
            n_bins = 1
        else:
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            if iqr <= 0:
                n_bins = int(cfg.get("time_bins_fallback", 1))
            else:
                bin_width = 2 * iqr / (len(data) ** (1.0 / 3.0))
                if isinstance(bin_width, np.timedelta64):
                    bin_width = bin_width / np.timedelta64(1, "s")
                    data_range = (data.max() - data.min()) / np.timedelta64(1, "s")
                else:
                    data_range = data.max() - data.min()
                n_bins = max(1, int(np.ceil(data_range / float(bin_width))))
    else:
        dt = int(cfg.get("plot_time_bin_width_s", cfg.get("time_bin_s", 3600)))
        n_bins = int(np.floor((t_end - t_start) / dt))
        if n_bins < 1:
            n_bins = 1

    if bin_mode not in ("fd", "auto"):
        edges = np.arange(0, (n_bins + 1) * dt, dt, dtype=float)
    else:
        edges = np.linspace(0, (t_end - t_start), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)
    return centers, widths


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


def parse_args(argv=None):
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Full Radon Monitor Analysis Pipeline")
    default_cfg = Path(__file__).resolve().with_name("config.yaml")
    p.add_argument(
        "--config",
        "-c",
        default=str(default_cfg),
        help="Path to YAML or JSON configuration file (default: config.yaml)",
    )
    default_input = Path.cwd() / "merged_output.csv"
    p.add_argument(
        "--input",
        "-i",
        default=str(default_input),
        help=(
            "CSV of merged event data (must contain at least: timestamp, adc) "
            f"(default: {default_input})"
        ),
    )
    p.add_argument(
        "--output_dir",
        "-o",
        default="results",
        help=(
            "Directory under which to create a timestamped analysis folder "
            "(override with --job-id; default: results)"
        ),
    )
    p.add_argument(
        "--timezone",
        default="UTC",
        help="Timezone for naive input timestamps (default: UTC)",
    )
    p.add_argument(
        "--baseline_range",
        nargs=2,
        metavar=("TSTART", "TEND"),
        type=str,
        help=(
            "Optional baseline-run interval. Providing this option overrides `baseline.range` in config.yaml. Provide two values (either ISO strings or epoch floats). If set, those events are extracted (same energy cuts) and listed in `baseline` of the summary."
        ),
    )
    p.add_argument(
        "--baseline-mode",
        choices=["none", "electronics", "radon", "all"],
        default="all",
        help="Background removal strategy (default: all)",
    )
    p.add_argument(
        "--iso",
        choices=["radon", "po218", "po214"],
        help="Analysis isotope mode (overrides analysis_isotope in config.yaml)",
    )
    p.add_argument(
        "--allow-negative-baseline",
        action="store_true",
        help="Allow negative baseline-corrected rates",
    )
    p.add_argument(
        "--allow-negative-activity",
        action="store_true",
        help="Continue if radon activity is negative",
    )
    p.add_argument(
        "--check-baseline-only",
        action="store_true",
        help="Exit after printing baseline diagnostics",
    )
    p.add_argument(
        "--burst-mode",
        choices=["none", "micro", "rate", "both"],
        help="Burst filtering mode to pass to apply_burst_filter. Providing this option overrides `burst_filter.burst_mode` in config.yaml",
    )
    p.add_argument(
        "--job-id",
        help="Optional identifier used for the results folder instead of the timestamp",
    )
    p.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing results folder if it already exists",
    )
    p.add_argument(
        "--efficiency-json",
        help="Path to a JSON file containing efficiency inputs to merge into the configuration",
    )
    p.add_argument(
        "--systematics-json",
        help="Path to a JSON file with systematics settings overriding the config",
    )
    p.add_argument(
        "--spike-count",
        type=float,
        help="Counts observed during a spike run for efficiency",
    )
    p.add_argument(
        "--spike-count-err",
        type=float,
        help="Uncertainty on spike counts",
    )
    p.add_argument(
        "--analysis-end-time",
        type=str,
        help="Ignore events occurring after this ISO timestamp. Providing this option overrides `analysis.analysis_end_time` in config.yaml",
    )
    p.add_argument(
        "--analysis-start-time",
        type=str,
        help="Reference start time of the analysis (ISO string or epoch). Overrides `analysis.analysis_start_time` in config.yaml",
    )
    p.add_argument(
        "--spike-start-time",
        help="Discard events after this ISO timestamp. Providing this option overrides `analysis.spike_start_time` in config.yaml",
    )
    p.add_argument(
        "--spike-end-time",
        help="Discard events before this ISO timestamp. Providing this option overrides `analysis.spike_end_time` in config.yaml",
    )
    p.add_argument(
        "--spike-period",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="Discard events between START and END (can be given multiple times). Providing this option overrides `analysis.spike_periods` in config.yaml",
    )
    p.add_argument(
        "--run-period",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="Keep events between START and END (can be given multiple times). Providing this option overrides `analysis.run_periods` in config.yaml",
    )
    p.add_argument(
        "--radon-interval",
        nargs=2,
        metavar=("START", "END"),
        help="Time interval to evaluate radon delta. Providing this option overrides `analysis.radon_interval` in config.yaml",
    )
    p.add_argument(
        "--slope",
        type=float,
        help="Apply a linear ADC drift correction with the given slope. Providing this option overrides `systematics.adc_drift_rate` in config.yaml",
    )
    p.add_argument(
        "--noise-cutoff",
        type=int,
        help=(
            "ADC threshold for the noise cut. Providing this option overrides "
            "`calibration.noise_cutoff` in config.yaml"
        ),
    )
    p.add_argument(
        "--calibration-method",
        choices=["two-point", "auto"],
        help=(
            "Energy calibration method. Providing this option overrides "
            "`calibration.method` in config.yaml"
        ),
    )
    p.add_argument(
        "--settle-s",
        type=float,
        help="Discard events occurring this many seconds after the start",
    )
    p.add_argument(
        "--hl-po214",
        type=float,
        help=(
            "Half-life to use for Po-214 in seconds. "
            "Providing this option overrides `time_fit.hl_po214` in config.yaml"
        ),
    )
    p.add_argument(
        "--hl-po218",
        type=float,
        help=(
            "Half-life to use for Po-218 in seconds. "
            "Providing this option overrides `time_fit.hl_po218` in config.yaml"
        ),
    )
    p.add_argument(
        "--eff-fixed",
        action="store_true",
        help="Fix all efficiencies to exactly 1.0 (no prior)",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging. Providing this option overrides `pipeline.log_level` in config.yaml",
    )
    p.add_argument(
        "--plot-time-binning-mode",
        dest="time_bin_mode_new",
        choices=["auto", "fd", "fixed"],
        help="Time-series binning mode. Providing this option overrides `plotting.plot_time_binning_mode` in config.yaml",
    )
    p.add_argument(
        "--time-bin-mode",
        dest="time_bin_mode_old",
        choices=["auto", "fd", "fixed"],
        help="DEPRECATED alias for --plot-time-binning-mode",
    )
    p.add_argument(
        "--plot-time-bin-width",
        dest="time_bin_width",
        type=float,
        help="Fixed time bin width in seconds. Providing this option overrides `plotting.plot_time_bin_width_s` in config.yaml",
    )
    p.add_argument(
        "--dump-ts-json",
        "--dump-time-series-json",
        dest="dump_ts_json",
        action="store_true",
        help="Write *_ts.json files containing binned time-series data",
    )
    p.add_argument(
        "--ambient-file",
        help=("Two-column text file of timestamp and ambient concentration in Bq/L"),
    )
    p.add_argument(
        "--ambient-concentration",
        type=float,
        help="Ambient radon concentration in Bq per liter for equivalent air plot. Providing this option overrides `analysis.ambient_concentration` in config.yaml",
    )
    p.add_argument(
        "--seed",
        type=int,
        help="Override random seed used by analysis algorithms. Providing this option overrides `pipeline.random_seed` in config.yaml",
    )
    p.add_argument(
        "--palette",
        help="Color palette for plots. Providing this option overrides `plotting.palette` in config.yaml",
    )
    p.add_argument(
        "--strict-covariance",
        action="store_true",
        help="Fail if fit covariance matrices are not positive definite",
    )
    p.add_argument(
        "--hierarchical-summary",
        metavar="OUTFILE",
        help=(
            "Combine half-life and calibration results from previous runs and "
            "write a hierarchical fit summary to OUTFILE"
        ),
    )

    args = p.parse_args(argv)

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

    # Resolve timezone for subsequent time parsing
    tzinfo = gettz(args.timezone)
    if tzinfo is None:
        print(f"ERROR: Unknown timezone '{args.timezone}'")
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
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Could not load config '{args.config}': {e}")
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
            print(
                f"ERROR: Could not load efficiency JSON '{args.efficiency_json}': {e}"
            )
            sys.exit(1)

    if args.systematics_json:
        try:
            with open(args.systematics_json, "r", encoding="utf-8") as f:
                cfg["systematics"] = json.load(f)
        except Exception as e:
            print(
                f"ERROR: Could not load systematics JSON '{args.systematics_json}': {e}"
            )
            sys.exit(1)

    if args.seed is not None:
        _log_override("pipeline", "random_seed", int(args.seed))
        cfg.setdefault("pipeline", {})["random_seed"] = int(args.seed)

    if args.ambient_concentration is not None:
        _log_override(
            "analysis",
            "ambient_concentration",
            float(args.ambient_concentration),
        )
        cfg.setdefault("analysis", {})["ambient_concentration"] = float(
            args.ambient_concentration
        )

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

    if args.spike_count is not None or args.spike_count_err is not None:
        eff_sec = cfg.setdefault("efficiency", {}).setdefault("spike", {})
        if args.spike_count is not None:
            eff_sec["counts"] = float(args.spike_count)
        if args.spike_count_err is not None:
            eff_sec["error"] = float(args.spike_count_err)

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

    # Configure logging as early as possible
    log_level = cfg.get("pipeline", {}).get("log_level", "INFO")
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=numeric_level,
        format="%(levelname)s:%(name)s:%(message)s",
    )

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

    # Timestamp for this analysis run
    now_str = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    # ────────────────────────────────────────────────────────────
    # 2. Load event data
    # ────────────────────────────────────────────────────────────
    try:
        events_all = load_events(args.input, column_map=cfg.get("columns"))

        # Parse timestamps to UTC ``Timestamp`` objects
        events_all["timestamp"] = events_all["timestamp"].map(parse_timestamp)

    except Exception as e:
        print(f"ERROR: Could not load events from '{args.input}': {e}")
        sys.exit(1)

    if events_all.empty:
        print("No events found in the input CSV. Exiting.")
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
    if noise_thr is not None:
        try:
            noise_thr_val = int(noise_thr)
        except (ValueError, TypeError):
            logging.warning(f"Invalid noise_cutoff '{noise_thr}' - skipping noise cut")
            noise_thr_val = None
        else:
            before = len(events_filtered)
            events_filtered = events_filtered[
                events_filtered["adc"] > noise_thr_val
            ].reset_index(drop=True)
            n_removed_noise = before - len(events_filtered)
            logging.info(f"Noise cut removed {n_removed_noise} events")

    _ensure_events(events_filtered, "noise cut")

    # Optional burst filter to remove high-rate clusters
    total_span = events_filtered["timestamp"].max() - events_filtered["timestamp"].min()
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
            print(f"WARNING: Could not apply ADC drift correction -> {e}")

    # ────────────────────────────────────────────────────────────
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
        logging.exception("calibration failed – using defaults")
        if not cfg.get("allow_fallback"):
            raise
        calibration_valid = False
        cal_params = {"a": (0.005, 0.001), "c": (0.02, 0.005), "sigma_E": (0.3, 0.1)}

    def _as_cal_result(obj):
        from calibration import CalibrationResult

        if isinstance(obj, CalibrationResult):
            return obj

        a, a_sig = obj.get("a", (0.0, 0.0))
        c, c_sig = obj.get("c", (0.0, 0.0))
        a2, a2_sig = obj.get("a2", (0.0, 0.0))

        coeffs = [c, a]
        cov = np.array([[c_sig**2, 0.0], [0.0, a_sig**2]])

        if "ac_covariance" in obj:
            cov_ac = float(np.asarray(obj["ac_covariance"], dtype=float)[0][1])
            cov[0, 1] = cov[1, 0] = cov_ac

        if "a2" in obj:
            coeffs.append(a2)
            cov = np.pad(cov, ((0, 1), (0, 1)), mode="constant", constant_values=0.0)
            cov[2, 2] = a2_sig**2
            cov[1, 2] = cov[2, 1] = float(obj.get("cov_a_a2", 0.0))
            cov[0, 2] = cov[2, 0] = float(obj.get("cov_a2_c", 0.0))

        return CalibrationResult(
            coeffs=coeffs,
            cov=cov,
            sigma_E=obj.get("sigma_E", (0.0, 0.0))[0],
            sigma_E_error=obj.get("sigma_E", (0.0, 0.0))[1],
            peaks=obj.get("peaks"),
        )

    cal_result = _as_cal_result(cal_params)

    # Save “a, c, sigma_E” so we can reconstruct energies
    if isinstance(cal_params, dict):
        a, a_sig = cal_params["a"]
        a2, a2_sig = cal_params.get("a2", (0.0, 0.0))
        c, c_sig = cal_params["c"]
        sigE_mean, sigE_sigma = cal_params["sigma_E"]
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
        c = cal_params.coeffs[idx.get(0, 0)] if 0 in idx else 0.0
        a = cal_params.coeffs[idx.get(1, 0)] if 1 in idx else 0.0
        a2 = cal_params.coeffs[idx.get(2, 0)] if 2 in idx else 0.0
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

    # Apply calibration -> new column “energy_MeV” and its uncertainty
    energies = cal_result.predict(df_analysis["adc"])
    df_analysis["energy_MeV"] = energies
    df_analysis["denergy_MeV"] = cal_result.uncertainty(df_analysis["adc"])

    # Derive default time-fit windows from calibration peaks when missing
    if getattr(cal_result, "peaks", None):
        tf_cfg = cfg.setdefault("time_fit", {})
        for iso in ("Po210", "Po218", "Po214"):
            key = f"window_{iso.lower()}"
            if tf_cfg.get(key) is None:
                peak_E = cal_result.peaks.get(iso, {}).get("centroid_mev")
                if peak_E is not None:
                    tf_cfg[key] = [float(peak_E - 0.08), float(peak_E + 0.08)]

    # ────────────────────────────────────────────────────────────
    # 4. Baseline run (optional)
    # ────────────────────────────────────────────────────────────
    baseline_info = {}
    baseline_counts = {}
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

    monitor_vol = float(baseline_cfg.get("monitor_volume_l", 605.0))
    sample_vol = float(baseline_cfg.get("sample_volume_l", 0.0))
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
                "Baseline interval outside data range – taking counts anyway"
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
            print(f"WARNING: Baseline noise estimation failed -> {e}")

        if noise_level is not None:
            # Store estimated noise peak amplitude in counts (not ADC units)
            baseline_info["noise_level"] = float(noise_level)

        # Record noise counts in ``baseline_counts``
        if "mask_noise" in locals():
            baseline_counts["noise"] = int(np.sum(mask_noise))

    _ensure_events(df_analysis, "baseline subtraction")

    if args.check_baseline_only:
        try:
            summary = summarize_baseline(cfg, isotopes_to_subtract)
        except BaselineError as e:
            print(f"BaselineError: {e}")
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
        print(table)
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
            print(f"WARNING: Baseline subtraction failed -> {e}")

    # ────────────────────────────────────────────────────────────
    # 5. Spectral fit (optional)
    # ────────────────────────────────────────────────────────────
    spectrum_results = {}
    spec_plot_data = None
    if cfg.get("spectral_fit", {}).get("do_spectral_fit", False):
        # Decide binning: new 'binning' dict or legacy keys
        bin_cfg = cfg["spectral_fit"].get("binning")
        if bin_cfg is not None:
            method = bin_cfg.get("method", "adc").lower()
            default_bins = bin_cfg.get("default_bins")
        else:
            method = str(
                cfg["spectral_fit"].get("spectral_binning_mode", "adc")
            ).lower()
            default_bins = cfg["spectral_fit"].get("fd_hist_bins")

        if method == "fd":
            E_all = df_analysis["energy_MeV"].values
            # Freedman‐Diaconis on energy array
            q25, q75 = np.percentile(E_all, [25, 75])
            iqr = q75 - q25
            n = E_all.size
            if (iqr > 0) and (n > 0):
                fd_width = 2 * iqr / (n ** (1 / 3))
                # fd_width is measured in MeV since energies are in MeV
                nbins = max(
                    1,
                    int(np.ceil((E_all.max() - E_all.min()) / float(fd_width))),
                )
            else:
                nbins = default_bins

            bins = nbins
            bin_edges = None
        else:
            # "ADC" binning mode -> fixed width in raw channels
            width = 1
            if bin_cfg is not None:
                width = bin_cfg.get("adc_bin_width", 1)
            else:
                width = cfg["spectral_fit"].get("adc_bin_width", 1)
            adc_min = df_analysis["adc"].min()
            adc_max = df_analysis["adc"].max()
            bins = int(np.ceil((adc_max - adc_min + 1) / width))

            # Build edges in ADC units then convert to energy for plotting
            bin_edges_adc = np.arange(adc_min, adc_min + bins * width + 1, width)
            bin_edges = apply_calibration(bin_edges_adc, a, c, quadratic_coeff=a2)

        # Find approximate ADC centroids for Po‐210, Po‐218, Po‐214

        expected_peaks = cfg.get("spectral_fit", {}).get("expected_peaks")
        if expected_peaks is None:
            expected_peaks = DEFAULT_ADC_CENTROIDS

        # `find_adc_bin_peaks` will return a dict: e.g. { "Po210": adc_centroid, … }
        adc_peaks = find_adc_bin_peaks(
            df_analysis["adc"].values,
            expected=expected_peaks,
            window=cfg["spectral_fit"].get("peak_search_width_adc", 50),
            prominence=cfg["spectral_fit"].get("peak_search_prominence", 0),
            width=cfg["spectral_fit"].get("peak_search_width_adc", None),
            method=cfg["spectral_fit"].get("peak_search_method", "prominence"),
            cwt_widths=cfg["spectral_fit"].get("peak_search_cwt_widths"),
        )

        # Build priors for the unbinned spectrum fit:
        priors_spec = {}
        # σ_E prior
        priors_spec["sigma_E"] = (
            sigE_mean,
            cfg["spectral_fit"].get("sigma_E_prior_source", sigE_sigma),
        )

        for peak, centroid_adc in adc_peaks.items():
            mu = apply_calibration(centroid_adc, a, c, quadratic_coeff=a2)
            bounds_cfg = cfg["spectral_fit"].get("mu_bounds", {})
            bounds = bounds_cfg.get(peak)
            if bounds is not None:
                lo, hi = bounds
                if not lo < hi:
                    raise ValueError(f"mu_bounds for {peak} require lower < upper")
                if not (lo <= mu <= hi):
                    mu = np.clip(mu, lo, hi)
            priors_spec[f"mu_{peak}"] = (mu, cfg["spectral_fit"].get("mu_sigma"))
            # Observed raw-counts around the expected energy window
            peak_tol = cfg["spectral_fit"].get("spectral_peak_tolerance_mev", 0.3)
            raw_count = float(
                (
                    (df_analysis["energy_MeV"] >= mu - peak_tol)
                    & (df_analysis["energy_MeV"] <= mu + peak_tol)
                ).sum()
            )
            mu_amp = max(raw_count, 1.0)
            sigma_amp = max(
                np.sqrt(mu_amp), cfg["spectral_fit"].get("amp_prior_scale") * mu_amp
            )
            priors_spec[f"S_{peak}"] = (mu_amp, sigma_amp)

            # If EMG tails are requested for this peak:
            if cfg["spectral_fit"].get("use_emg", {}).get(peak, False):
                priors_spec[f"tau_{peak}"] = (
                    cfg["spectral_fit"].get(f"tau_{peak}_prior_mean"),
                    cfg["spectral_fit"].get(f"tau_{peak}_prior_sigma"),
                )

        # Continuum priors
        bkg_mode = str(cfg["spectral_fit"].get("bkg_mode", "manual")).lower()
        if bkg_mode == "auto":
            from background import estimate_linear_background

            mu_map = {k: priors_spec[f"mu_{k}"][0] for k in adc_peaks.keys()}
            peak_tol = cfg["spectral_fit"].get("spectral_peak_tolerance_mev", 0.3)
            b0_est, b1_est = estimate_linear_background(
                df_analysis["energy_MeV"].values,
                mu_map,
                peak_width=peak_tol,
            )
            priors_spec["b0"] = (b0_est, abs(b0_est) * 0.1 + 1e-3)
            priors_spec["b1"] = (b1_est, abs(b1_est) * 0.1 + 1e-3)
        else:
            priors_spec["b0"] = tuple(cfg["spectral_fit"].get("b0_prior"))
            priors_spec["b1"] = tuple(cfg["spectral_fit"].get("b1_prior"))

        # Flags controlling the spectral fit
        spec_flags = cfg["spectral_fit"].get("flags", {}).copy()
        if not cfg["spectral_fit"].get("float_sigma_E", True):
            spec_flags["fix_sigma_E"] = True

        # Launch the spectral fit
        spec_fit_out = None
        try:
            fit_kwargs = {
                "energies": df_analysis["energy_MeV"].values,
                "priors": priors_spec,
                "flags": spec_flags,
            }
            if cfg["spectral_fit"].get("use_plot_bins_for_fit", False):
                fit_kwargs.update({"bins": bins, "bin_edges": bin_edges})
            if cfg["spectral_fit"].get("unbinned_likelihood", False):
                fit_kwargs["unbinned"] = True
            if args.strict_covariance:
                fit_kwargs["strict"] = True
            bounds_cfg = cfg["spectral_fit"].get("mu_bounds", {})
            if bounds_cfg:
                bounds_map = {}
                for iso, bnd in bounds_cfg.items():
                    if bnd is not None:
                        bounds_map[f"mu_{iso}"] = tuple(bnd)
                if bounds_map:
                    fit_kwargs["bounds"] = bounds_map
            spec_fit_out = fit_spectrum(**fit_kwargs)
            spectrum_results = spec_fit_out
        except Exception as e:
            print(f"WARNING: Spectral fit failed -> {e}")
            spectrum_results = {}

        # Store plotting inputs (bin_edges now in energy units)
        fit_vals = None
        if isinstance(spec_fit_out, FitResult):
            fit_vals = spec_fit_out.params
        elif isinstance(spec_fit_out, dict):
            fit_vals = spec_fit_out
        spec_plot_data = {
            "energies": df_analysis["energy_MeV"].values,
            "fit_vals": fit_vals,
            "bins": bins,
            "bin_edges": bin_edges,
        }

    # ────────────────────────────────────────────────────────────
    # 6. Time‐series decay fits for Po‐218 and Po‐214
    # ────────────────────────────────────────────────────────────
    time_fit_results = {}
    priors_time_all = {}
    time_plot_data = {}
    iso_live_time = {}
    t_start_map = {}
    iso_counts = {}
    iso_counts_raw = {}
    radon_estimate_info = None
    po214_estimate_info = None
    po218_estimate_info = None
    if cfg.get("time_fit", {}).get("do_time_fit", False):
        for iso in ("Po218", "Po214"):
            win_key = f"window_{iso.lower()}"

            # Missing energy window for this isotope -> skip gracefully
            win_range = cfg.get("time_fit", {}).get(win_key)
            if win_range is None:
                print(
                    f"INFO: Config key '{win_key}' not found. Skipping time fit for {iso}."
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
            if iso_events.empty:
                print(f"WARNING: No events found for {iso} in [{lo}, {hi}] MeV.")
                continue

            first_ts = to_datetime_utc(iso_events["timestamp"].iloc[0])
            t0_dt = to_utc_datetime(t0_global)
            settle = timedelta(seconds=float(args.settle_s or 0))
            t_start_fit_dt = max(first_ts, t0_dt + settle)
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

        # Half-life prior (user must supply [T₁/₂, σ(T₁/₂)] in seconds)
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
            baseline_info.setdefault("corrected_activity", {})[iso] = {
                "value": c_rate,
                "uncertainty": c_sigma,
            }
            weight_factor = 1.0 / (c_sigma**2) if c_sigma > 0 else 1.0
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
            baseline_info.setdefault("corrected_activity", {})[iso] = {
                "value": c_rate,
                "uncertainty": c_sigma,
            }
            weight_factor = 1.0 / (c_sigma**2) if c_sigma > 0 else 1.0
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
        eff_cfg_val = cfg["time_fit"].get(f"eff_{iso.lower()}")
        if args.eff_fixed:
            eff_value = 1.0
        else:
            eff_value = None
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
            "fit_initial": not cfg["time_fit"]["flags"].get(
                f"fix_N0_{iso.lower()}", False
            ),
            "background_guess": cfg["time_fit"].get("background_guess", 0.0),
            "n0_guess_fraction": cfg["time_fit"].get("n0_guess_fraction", 0.1),
        }

        # Run time-series fit
        decay_out = None  # fresh variable each iteration
        try:
            t_start_val = t_start_map.get(iso)
            if isinstance(t_start_val, datetime):
                t_start_fit = t_start_val.timestamp()
            else:
                t_start_fit = to_utc_datetime(
                    t_start_val if t_start_val is not None else t0_global
                ).timestamp()
            try:
                decay_out = fit_time_series(
                    times_dict,
                    t_start_fit,
                    t_end_global_ts,
                    fit_cfg,
                    weights=weights_map,
                    strict=args.strict_covariance,
                )
            except TypeError:
                decay_out = fit_time_series(
                    times_dict,
                    t_start_fit,
                    t_end_global_ts,
                    fit_cfg,
                    strict=args.strict_covariance,
                )
            time_fit_results[iso] = decay_out
        except Exception as e:
            print(f"WARNING: Decay‐curve fit for {iso} failed -> {e}")
            time_fit_results[iso] = {}

        # Store inputs for plotting later
        time_plot_data[iso] = {
            "events_times": iso_events["timestamp"].values,
            "events_energy": iso_events["energy_MeV"].values,
        }

    # --- Radon combination ---
    from radon_joint_estimator import estimate_radon_activity
    from types import SimpleNamespace

    fit214_obj = time_fit_results.get("Po214")
    fit218_obj = time_fit_results.get("Po218")
    fit214 = fit218 = None
    if fit214_obj:
        p = _fit_params(fit214_obj)
        fit214 = SimpleNamespace(
            rate=p.get("E_corrected", p.get("E_Po214")),
            err=p.get("dE_corrected", p.get("dE_Po214")),
            counts=getattr(fit214_obj, "counts", None),
            params=p,
        )
    if fit218_obj:
        p = _fit_params(fit218_obj)
        fit218 = SimpleNamespace(
            rate=p.get("E_corrected", p.get("E_Po218")),
            err=p.get("dE_corrected", p.get("dE_Po218")),
            counts=getattr(fit218_obj, "counts", None),
            params=p,
        )

    iso_mode = cfg.get("analysis_isotope", "radon").lower()

    if iso_mode == "radon":
        have_218 = (
            fit218
            and fit218.counts is not None
            and ("eff" in fit218.params or "eff_Po218" in fit218.params)
        )
        have_214 = (
            fit214
            and fit214.counts is not None
            and ("eff" in fit214.params or "eff_Po214" in fit214.params)
        )
        if have_218 or have_214:
            N218 = fit218.counts if have_218 else None
            N214 = fit214.counts if have_214 else None
            eps218 = (
                fit218.params.get("eff", fit218.params.get("eff_Po218", 1.0))
                if fit218
                else 1.0
            )
            eps214 = (
                fit214.params.get("eff", fit214.params.get("eff_Po214", 1.0))
                if fit214
                else 1.0
            )
            radon_estimate_info = estimate_radon_activity(
                N218=N218,
                epsilon218=eps218,
                f218=1.0,
                N214=N214,
                epsilon214=eps214,
                f214=1.0,
            )
        elif (fit214 and fit214.rate is not None) or (fit218 and fit218.rate is not None):
            radon_estimate_info = estimate_radon_activity(
                rate214=fit214.rate if fit214 else None,
                err214=fit214.err if fit214 else None,
                rate218=fit218.rate if fit218 else None,
                err218=fit218.err if fit218 else None,
            )
    elif iso_mode == "po218":
        if fit218:
            po218_estimate_info = {"activity_Bq": fit218.rate, "stat_unc_Bq": fit218.err}
    elif iso_mode == "po214":
        if fit214:
            po214_estimate_info = {"activity_Bq": fit214.rate, "stat_unc_Bq": fit214.err}
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

    # ────────────────────────────────────────────────────────────
    # 7. Systematics scan (optional)
    # ────────────────────────────────────────────────────────────
    systematics_results = {}
    if cfg.get("systematics", {}).get("enable", False):
        sys_cfg = cfg.get("systematics", {})

        for iso, fit_out in time_fit_results.items():
            if not fit_out:
                continue

            sigma_dict = {}
            for name in ("sigma_E_frac", "tail_fraction", "energy_shift_keV"):
                if name in sys_cfg:
                    base = name.replace("_frac", "").replace("_keV", "")
                    if base in priors_time_all.get(iso, {}):
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
                    "fit_initial": not cfg["time_fit"]["flags"].get(
                        f"fix_N0_{iso.lower()}", False
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
                print(f"WARNING: Systematics scan for {iso} -> {e}")

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
                try:
                    val = get_spike_efficiency(scfg)
                    err = float(scfg.get("error", 0.0))
                    key = "spike" if isinstance(scfg_raw, dict) else f"spike_{idx}"
                    sources[key] = {"value": val, "error": err}
                    vals.append(val)
                    errs.append(err)
                except Exception as e:
                    print(f"WARNING: Spike efficiency -> {e}")

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
                    print(f"WARNING: Assay efficiency -> {e}")

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
                print(f"WARNING: Decay efficiency -> {e}")

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
                print(f"WARNING: BLUE combination failed -> {e}")

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
    if baseline_live_time > 0:
        for iso, n in baseline_counts.items():
            eff_cfg = cfg["time_fit"].get(f"eff_{iso.lower()}")
            if isinstance(eff_cfg, list):
                eff = eff_cfg[0]
            else:
                eff = eff_cfg if eff_cfg is not None else 1.0
            if eff > 0:
                baseline_rates[iso] = n / (baseline_live_time * eff)
                baseline_unc[iso] = np.sqrt(n) / (baseline_live_time * eff)
            else:
                baseline_rates[iso] = 0.0
                baseline_unc[iso] = 0.0

    dilution_factor = compute_dilution_factor(monitor_vol, sample_vol)
    scales = {
        "Po214": dilution_factor,
        "Po218": dilution_factor,
        "Po210": 1.0,
        "noise": 1.0,
    }
    baseline_info["scales"] = scales
    baseline_info["analysis_counts"] = iso_counts_raw

    corrected_rates = {}
    corrected_unc = {}
    activity_rows = []

    for iso, fit in time_fit_results.items():
        params = _fit_params(fit)
        if not params or f"E_{iso}" not in params:
            continue

        if iso not in isotopes_to_subtract or baseline_live_time <= 0:
            continue

        err_fit = params.get(f"dE_{iso}", 0.0)
        live_time_iso = iso_live_time.get(iso, 0.0)
        count = iso_counts_raw.get(iso, baseline_counts.get(iso, 0.0))
        eff_cfg = cfg["time_fit"].get(f"eff_{iso.lower()}")
        if isinstance(eff_cfg, list):
            eff = eff_cfg[0]
        else:
            eff = eff_cfg if eff_cfg is not None else 1.0
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

        params["E_corrected"] = corr_rate
        params["dE_corrected"] = corr_sigma
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
        print(f"ERROR: {e}")
        sys.exit(1)

    # ────────────────────────────────────────────────────────────
    # Radon activity extrapolation
    # ────────────────────────────────────────────────────────────
    from radon_activity import compute_radon_activity, compute_total_radon

    radon_results = {}
    radon_combined_info = None

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
        rate214 = fit_dict.get("E_corrected", fit_dict.get("E_Po214"))
        err214 = fit_dict.get("dE_corrected", fit_dict.get("dE_Po214"))

    rate218 = None
    err218 = None
    if "Po218" in time_fit_results:
        fit_dict = _fit_params(time_fit_results["Po218"])
        rate218 = fit_dict.get("E_corrected", fit_dict.get("E_Po218"))
        err218 = fit_dict.get("dE_corrected", fit_dict.get("dE_Po218"))

    A_radon, dA_radon = compute_radon_activity(
        rate218, err218, eff_po218, rate214, err214, eff_po214
    )

    if cfg.get("analysis_isotope", "radon") == "radon":
        radon_combined_info = {
            "activity_Bq": A_radon,
            "unc_Bq": dA_radon,
        }

    # Convert activity to a concentration per liter of monitor volume and the
    # total amount of radon present in just the assay sample.
    try:
        conc, dconc, total_bq, dtotal_bq = compute_total_radon(
            A_radon,
            dA_radon,
            monitor_vol,
            sample_vol,
            allow_negative_activity=args.allow_negative_activity,
        )
    except RuntimeError as e:
        print(f"ERROR: {e}")
        sys.exit(1)

    radon_results["radon_activity_Bq"] = {"value": A_radon, "uncertainty": dA_radon}
    radon_results["radon_concentration_Bq_per_L"] = {
        "value": conc,
        "uncertainty": dconc,
    }
    radon_results["total_radon_in_sample_Bq"] = {
        "value": total_bq,
        "uncertainty": dtotal_bq,
    }

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
            E = fit.get("E_corrected", fit.get("E_Po214"))
            dE = fit.get("dE_corrected", fit.get("dE_Po214", 0.0))
            N0 = fit.get("N0_Po214", 0.0)
            dN0 = fit.get("dN0_Po214", 0.0)
            hl = _hl_value(cfg, "Po214")
            cov = _cov_lookup(fit_result, "E_Po214", "N0_Po214")
            delta214, err_delta214 = radon_delta(
                t_start_rel,
                t_end_rel,
                E,
                dE,
                N0,
                dN0,
                hl,
                cov,
            )

        delta218 = err_delta218 = None
        if "Po218" in time_fit_results:
            fit_result = time_fit_results["Po218"]
            fit = _fit_params(fit_result)
            E = fit.get("E_corrected", fit.get("E_Po218"))
            dE = fit.get("dE_corrected", fit.get("dE_Po218", 0.0))
            N0 = fit.get("N0_Po218", 0.0)
            dN0 = fit.get("dN0_Po218", 0.0)
            hl = _hl_value(cfg, "Po218")
            cov = _cov_lookup(fit_result, "E_Po218", "N0_Po218")
            delta218, err_delta218 = radon_delta(
                t_start_rel,
                t_end_rel,
                E,
                dE,
                N0,
                dN0,
                hl,
                cov,
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
    # 8. Assemble and write out the summary JSON
    # ────────────────────────────────────────────────────────────
    spec_dict = {}
    if isinstance(spectrum_results, FitResult):
        spec_dict = dict(spectrum_results.params)
        spec_dict["cov"] = spectrum_results.cov.tolist()
        spec_dict["ndf"] = spectrum_results.ndf
    elif isinstance(spectrum_results, dict):
        spec_dict = spectrum_results

    time_fit_serializable = {}
    for iso, fit in time_fit_results.items():
        if isinstance(fit, FitResult):
            d = dict(fit.params)
            d["cov"] = fit.cov.tolist()
            d["ndf"] = fit.ndf
        elif isinstance(fit, dict):
            d = fit
        else:
            d = {}
        time_fit_serializable[iso] = d

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
            "ambient_concentration": cfg.get("analysis", {}).get(
                "ambient_concentration"
            ),
            "settle_s": cfg.get("analysis", {}).get("settle_s"),
        },
    )

    if radon_combined_info is not None:
        summary.radon_combined = radon_combined_info

    from radon_joint_estimator import estimate_radon_activity

    iso_mode = cfg.get("analysis_isotope", "radon").lower()
    if iso_mode == "radon":
        radon = estimate_radon_activity(
            N218       = fit218.counts if fit218 else 0,
            epsilon218 = fit218.params.get("eff", 1.0) if fit218 else 1.0,
            N214       = fit214.counts if fit214 else 0,
            epsilon214 = fit214.params.get("eff", 1.0) if fit214 else 1.0,
            f218 = 1.0,
            f214 = 1.0,
        )

        # ── Construct a one-point time-series so the plotters don’t crash ──
        run_midpoint = 0.5 * (t0_cfg.timestamp() + t_end_cfg.timestamp())
        radon["time_series"] = {
            "time":     [run_midpoint],
            "activity": [radon["Rn_activity_Bq"]],
            "error":    [radon["stat_unc_Bq"]],
        }

        summary["radon"] = radon
    elif iso_mode == "po218":
        if fit218:
            summary["po218"] = {"activity_Bq": fit218.rate, "stat_unc_Bq": fit218.err}
    elif iso_mode == "po214":
        if fit214:
            summary["po214"] = {"activity_Bq": fit214.rate, "stat_unc_Bq": fit214.err}
    else:
        raise ValueError(f"Unknown analysis_isotope {iso_mode!r}")

    if weights is not None:
        summary.efficiency = summary.efficiency or {}
        summary.efficiency["blue_weights"] = list(weights)

    results_dir = Path(args.output_dir) / (args.job_id or now_str)
    if results_dir.exists():
        if args.overwrite:
            shutil.rmtree(results_dir)
        else:
            raise FileExistsError(f"Results folder already exists: {results_dir}")

    copy_config(results_dir, cfg, exist_ok=args.overwrite)
    out_dir = Path(write_summary(results_dir, summary))
    out_dir.mkdir(parents=True, exist_ok=True)

    if iso_mode == "radon" and "radon" in summary:
        rad_ts = summary["radon"]["time_series"]


        plot_radon_activity(
            rad_ts["time"],
            rad_ts["activity"],
            Path(out_dir) / "radon_activity.png",
            rad_ts.get("error"),
            config=cfg.get("plotting", {}),
        )
        plot_radon_trend(
            rad_ts["time"],
            rad_ts["activity"],
            Path(out_dir) / "radon_trend.png",
            config=cfg.get("plotting", {}),
        )



    # Generate plots now that the output directory exists
    if spec_plot_data:
        try:
            _ = plot_spectrum(
                energies=spec_plot_data["energies"],
                fit_vals=spec_plot_data["fit_vals"],
                out_png=Path(out_dir) / "spectrum.png",
                bins=spec_plot_data["bins"],
                bin_edges=spec_plot_data["bin_edges"],
                config=cfg.get("plotting", {}),
            )
        except Exception as e:
            print(f"WARNING: Could not create spectrum plot: {e}")

    overlay = cfg.get("plotting", {}).get("overlay_isotopes", False)

    for iso, pdata in time_plot_data.items():
        try:
            plot_cfg = dict(cfg.get("time_fit", {}))
            plot_cfg.update(cfg.get("plotting", {}))
            if not overlay:
                for other_iso in ("Po214", "Po218", "Po210"):
                    if other_iso != iso:
                        plot_cfg[f"window_{other_iso.lower()}"] = None
                ts_times = pdata["events_times"]
                ts_energy = pdata["events_energy"]
                fit_obj = time_fit_results.get(iso)
                fit_dict = _fit_params(fit_obj)
            else:
                ts_times = df_analysis["timestamp"].values
                ts_energy = df_analysis["energy_MeV"].values
                fit_dict = {}
                for k in ("Po214", "Po218", "Po210"):
                    obj = time_fit_results.get(k)
                    if obj:
                        fit_dict.update(_fit_params(obj))

            centers, widths = _ts_bin_centers_widths(
                ts_times, plot_cfg, t0_global.timestamp(), t_end_global_ts
            )
            normalise = bool(plot_cfg.get("plot_time_normalise_rate", False))
            model_errs = {}
            iso_list_err = (
                [iso]
                if not overlay
                else [i for i in ("Po214", "Po218", "Po210") if time_fit_results.get(i)]
            )
            for iso_key in iso_list_err:
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
            _ = plot_time_series(
                all_timestamps=ts_times,
                all_energies=ts_energy,
                fit_results=fit_dict,
                t_start=t0_global.timestamp(),
                t_end=t_end_global_ts,
                config=plot_cfg,
                out_png=Path(out_dir) / f"time_series_{iso}.png",
                model_errors=model_errs,
            )
        except Exception as e:
            print(f"WARNING: Could not create time-series plot for {iso} -> {e}")

    # Additional visualizations
    if efficiency_results.get("sources"):
        try:
            errs_arr = np.array(
                [s.get("error", 0.0) for s in efficiency_results["sources"].values()]
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
            print(f"WARNING: Could not create efficiency plots -> {e}")

    # Radon activity and equivalent air plots
    try:
        from radon_activity import radon_activity_curve

        times = np.linspace(t0_global.timestamp(), t_end_global_ts, 100)
        times_dt = to_datetime_utc(times, unit="s")
        t_rel = (times_dt - analysis_start).total_seconds()

        if radon_combined_info is not None:
            try:
                _ = plot_radon_activity(
                    [t0_global.timestamp(), t_end_global_ts],
                    [radon_combined_info["activity_Bq"]] * 2,
                    Path(out_dir) / "radon_activity_combined.png",
                    [radon_combined_info["unc_Bq"]] * 2,
                    config=cfg.get("plotting", {}),
                )
            except Exception as e:
                print(f"WARNING: Could not create radon combined plot -> {e}")

        A214 = dA214 = None
        if "Po214" in time_fit_results:
            fit_result = time_fit_results["Po214"]
            fit = _fit_params(fit_result)
            E = fit.get("E_corrected", fit.get("E_Po214"))
            dE = fit.get("dE_corrected", fit.get("dE_Po214", 0.0))
            N0 = fit.get("N0_Po214", 0.0)
            dN0 = fit.get("dN0_Po214", 0.0)
            hl = _hl_value(cfg, "Po214")
            cov = _cov_lookup(fit_result, "E_Po214", "N0_Po214")
            A214, dA214 = radon_activity_curve(t_rel, E, dE, N0, dN0, hl, cov)
            plot_radon_activity(
                times,
                A214,
                Path(out_dir) / "radon_activity_po214.png",
                dA214,
                config=cfg.get("plotting", {}),
            )

        A218 = dA218 = None
        if "Po218" in time_fit_results:
            fit_result = time_fit_results["Po218"]
            fit = _fit_params(fit_result)
            E = fit.get("E_corrected", fit.get("E_Po218"))
            dE = fit.get("dE_corrected", fit.get("dE_Po218", 0.0))
            N0 = fit.get("N0_Po218", 0.0)
            dN0 = fit.get("dN0_Po218", 0.0)
            hl = _hl_value(cfg, "Po218")
            cov = _cov_lookup(fit_result, "E_Po218", "N0_Po218")
            A218, dA218 = radon_activity_curve(t_rel, E, dE, N0, dN0, hl, cov)

        activity_arr = np.zeros_like(times, dtype=float)
        err_arr = np.zeros_like(times, dtype=float)
        for i in range(times.size):
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

        if np.all(activity_arr == 0):
            activity_arr.fill(radon_results["radon_activity_Bq"]["value"])
            err_arr.fill(radon_results["radon_activity_Bq"]["uncertainty"])

        plot_radon_activity(
            times,
            activity_arr,
            Path(out_dir) / "radon_activity.png",
            err_arr,
            config=cfg.get("plotting", {}),
        )

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
                E214 = fit.get("E_corrected", fit.get("E_Po214"))
                dE214 = fit.get("dE_corrected", fit.get("dE_Po214", 0.0))
                N0214 = fit.get("N0_Po214", 0.0)
                dN0214 = fit.get("dN0_Po214", 0.0)
                hl214 = _hl_value(cfg, "Po214")
                cov214 = _cov_lookup(fit_result, "E_Po214", "N0_Po214")
                A214_tr, _ = radon_activity_curve(
                    rel_trend, E214, dE214, N0214, dN0214, hl214, cov214
                )
            A218_tr = None
            if "Po218" in time_fit_results:
                fit_result = time_fit_results["Po218"]
                fit = _fit_params(fit_result)
                E218 = fit.get("E_corrected", fit.get("E_Po218"))
                dE218 = fit.get("dE_corrected", fit.get("dE_Po218", 0.0))
                N0218 = fit.get("N0_Po218", 0.0)
                dN0218 = fit.get("dN0_Po218", 0.0)
                hl218 = _hl_value(cfg, "Po218")
                cov218 = _cov_lookup(fit_result, "E_Po218", "N0_Po218")
                A218_tr, _ = radon_activity_curve(
                    rel_trend, E218, dE218, N0218, dN0218, hl218, cov218
                )
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
            )

        ambient = cfg.get("analysis", {}).get("ambient_concentration")
        ambient_interp = None
        if args.ambient_file:
            try:
                dat = np.loadtxt(args.ambient_file, usecols=(0, 1))
                ambient_interp = np.interp(times, dat[:, 0], dat[:, 1])
            except Exception as e:
                print(
                    f"WARNING: Could not read ambient file '{args.ambient_file}': {e}"
                )

        if ambient_interp is not None:
            vol_arr = activity_arr / ambient_interp
            vol_err = err_arr / ambient_interp
            plot_equivalent_air(
                times,
                vol_arr,
                vol_err,
                None,
                Path(out_dir) / "equivalent_air.png",
                config=cfg.get("plotting", {}),
            )
            if A214 is not None:
                plot_equivalent_air(
                    times,
                    A214 / ambient_interp,
                    dA214 / ambient_interp,
                    None,
                    Path(out_dir) / "equivalent_air_po214.png",
                    config=cfg.get("plotting", {}),
                )
        elif ambient:
            vol_arr = activity_arr / float(ambient)
            vol_err = err_arr / float(ambient)
            plot_equivalent_air(
                times,
                vol_arr,
                vol_err,
                float(ambient),
                Path(out_dir) / "equivalent_air.png",
                config=cfg.get("plotting", {}),
            )
            if A214 is not None:
                plot_equivalent_air(
                    times,
                    A214 / float(ambient),
                    dA214 / float(ambient),
                    float(ambient),
                    Path(out_dir) / "equivalent_air_po214.png",
                    config=cfg.get("plotting", {}),
                )
    except Exception as e:
        print(f"WARNING: Could not create radon activity plots -> {e}")

    if args.hierarchical_summary:
        try:
            run_results = []
            for p in args.output_dir.glob("*/summary.json"):
                try:
                    with open(p, "r", encoding="utf-8") as f:
                        dat = json.load(f)
                except Exception as e:
                    print(f"WARNING: Skipping {p} -> {e}")
                    continue
                hl = dat.get("half_life")
                dhl = dat.get("dhalf_life")
                cal = dat.get("calibration", {})
                slope, dslope = cal.get("a", (None, None))
                intercept, dintercept = cal.get("c", (None, None))
                if hl is not None:
                    run_results.append(
                        {
                            "half_life": hl,
                            "dhalf_life": dhl,
                            "slope_MeV_per_ch": slope,
                            "dslope": dslope,
                            "intercept": intercept,
                            "dintercept": dintercept,
                        }
                    )
            if run_results:
                hier_out = fit_hierarchical_runs(run_results)
                with open(args.hierarchical_summary, "w", encoding="utf-8") as f:
                    json.dump(hier_out, f, indent=4)
        except Exception as e:
            print(f"WARNING: hierarchical fit failed -> {e}")

    print(f"Analysis complete. Results written to -> {out_dir}")


if __name__ == "__main__":
    main()
