#!/usr/bin/env python3
"""
analyze.py

Full Radon Monitor Analysis Pipeline
====================================

Usage:
    python analyze.py \
        --config   config.json \
        --input    merged_output.csv \
        --output_dir  results \
        [--baseline_range ISO_START ISO_END]

This script performs the following steps:

  1. Load JSON configuration.
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
       --config    config.json \
       --input     merged_output.csv \
       --output_dir  results
"""


import argparse
import sys
import logging
import random
from datetime import datetime, timezone
import subprocess
import hashlib
import json
from pathlib import Path
import shutil

import math
import numpy as np
import pandas as pd
from scipy.stats import norm
from dateutil import parser as date_parser
from dateutil.tz import UTC

from hierarchical import fit_hierarchical_runs

# ‣ Import our supporting modules (all must live in the same folder).
from io_utils import (
    load_config,
    copy_config,
    load_events,
    write_summary,
    apply_burst_filter,
)
from calibration import (
    derive_calibration_constants,
    derive_calibration_constants_auto,
    apply_calibration,
)

from fitting import fit_spectrum, fit_time_series, FitResult

from constants import (
    DEFAULT_NOISE_CUTOFF,
    PO210,
    PO214,
    PO218,
    DEFAULT_ADC_CENTROIDS,
)

from plot_utils import (
    plot_spectrum,
    plot_time_series,
    plot_radon_activity,
    plot_equivalent_air,
    plot_radon_trend,
)
from systematics import scan_systematics, apply_linear_adc_shift
from visualize import cov_heatmap, efficiency_bar
from utils import (
    find_adc_bin_peaks,
    adc_hist_edges,
    cps_to_bq,
    parse_time,
    parse_time_arg,
)
from radmon.baseline import subtract_baseline
from radon.baseline import subtract_baseline as subtract_radon_baseline


def _fit_params(obj):
    """Return fit parameters dictionary from either a FitResult or mapping."""
    if isinstance(obj, FitResult):
        return obj.params
    if isinstance(obj, dict):
        return obj
    return {}


def _cov_entry(fit: FitResult | dict, p1: str, p2: str) -> float:
    """Return covariance between two parameters from a ``FitResult``.

    Raises
    ------
    KeyError
        If either ``p1`` or ``p2`` is not present in the covariance matrix.
    """

    if not isinstance(fit, FitResult) or fit.cov is None:
        return 0.0

    ordered = [
        k for k in fit.params.keys() if k != "fit_valid" and not k.startswith("d")
    ]

    if p1 not in ordered or p2 not in ordered:
        raise KeyError(f"Parameter(s) missing in covariance: {p1}, {p2}")

    i1 = ordered.index(p1)
    i2 = ordered.index(p2)
    cov = np.asarray(fit.cov, dtype=float)
    if cov.ndim >= 2 and i1 < cov.shape[0] and i2 < cov.shape[1]:
        return float(cov[i1, i2])

    raise KeyError(f"Parameter(s) missing in covariance: {p1}, {p2}")


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


def parse_args(argv=None):
    """Parse command line arguments."""
    p = argparse.ArgumentParser(description="Full Radon Monitor Analysis Pipeline")
    p.add_argument(
        "--config", "-c", required=True, help="Path to JSON configuration file"
    )
    p.add_argument(
        "--input",
        "-i",
        required=True,
        help="CSV of merged event data (must contain at least: timestamp, adc)",
    )
    p.add_argument(
        "--output_dir",
        "-o",
        required=True,
        help="Directory under which to create a timestamped analysis folder (override with --job-id)",
    )
    p.add_argument(
        "--baseline_range",
        nargs=2,
        metavar=("TSTART", "TEND"),
        type=parse_time_arg,
        help=(
            "Optional baseline-run interval. Providing this option overrides `baseline.range` in config.json. Provide two values (either ISO strings or epoch floats). If set, those events are extracted (same energy cuts) and listed in `baseline` of the summary."
        ),
    )
    p.add_argument(
        "--baseline-mode",
        choices=["none", "electronics", "radon", "all"],
        default="all",
        help="Background removal strategy (default: all)",
    )
    p.add_argument(
        "--burst-mode",
        choices=["none", "micro", "rate", "both"],
        help="Burst filtering mode to pass to apply_burst_filter. Providing this option overrides `burst_filter.burst_mode` in config.json",
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
        type=parse_time_arg,
        help="Ignore events occurring after this ISO timestamp. Providing this option overrides `analysis.analysis_end_time` in config.json",
    )
    p.add_argument(
        "--analysis-start-time",
        type=parse_time_arg,
        help="Reference start time of the analysis (ISO string or epoch). Overrides `analysis.analysis_start_time` in config.json",
    )
    p.add_argument(
        "--spike-end-time",
        help="Discard events before this ISO timestamp. Providing this option overrides `analysis.spike_end_time` in config.json",
    )
    p.add_argument(
        "--spike-period",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="Discard events between START and END (can be given multiple times). Providing this option overrides `analysis.spike_periods` in config.json",
    )
    p.add_argument(
        "--run-period",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="Keep events between START and END (can be given multiple times). Providing this option overrides `analysis.run_periods` in config.json",
    )
    p.add_argument(
        "--radon-interval",
        nargs=2,
        metavar=("START", "END"),
        help="Time interval to evaluate radon delta. Providing this option overrides `analysis.radon_interval` in config.json",
    )
    p.add_argument(
        "--slope",
        type=float,
        help="Apply a linear ADC drift correction with the given slope. Providing this option overrides `systematics.adc_drift_rate` in config.json",
    )
    p.add_argument(
        "--noise-cutoff",
        type=int,
        help=(
            "ADC threshold for the noise cut. Providing this option overrides "
            "`calibration.noise_cutoff` in config.json"

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
            "Providing this option overrides `time_fit.hl_po214` in config.json"
        ),
    )
    p.add_argument(
        "--hl-po218",
        type=float,
        help=(
            "Half-life to use for Po-218 in seconds. "
            "Providing this option overrides `time_fit.hl_po218` in config.json"
        ),
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging. Providing this option overrides `pipeline.log_level` in config.json",
    )
    p.add_argument(
        "--plot-time-binning-mode",
        dest="time_bin_mode_new",
        choices=["auto", "fd", "fixed"],
        help="Time-series binning mode. Providing this option overrides `plotting.plot_time_binning_mode` in config.json",
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
        help="Fixed time bin width in seconds. Providing this option overrides `plotting.plot_time_bin_width_s` in config.json",
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
        help="Ambient radon concentration in Bq per liter for equivalent air plot. Providing this option overrides `analysis.ambient_concentration` in config.json",
    )
    p.add_argument(
        "--seed",
        type=int,
        help="Override random seed used by analysis algorithms. Providing this option overrides `pipeline.random_seed` in config.json",
    )
    p.add_argument(
        "--palette",
        help="Color palette for plots. Providing this option overrides `plotting.palette` in config.json",
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
        cfg.setdefault("analysis", {})["analysis_end_time"] = args.analysis_end_time.timestamp()

    if args.analysis_start_time is not None:
        _log_override("analysis", "analysis_start_time", args.analysis_start_time)
        cfg.setdefault("analysis", {})["analysis_start_time"] = args.analysis_start_time.timestamp()

    if args.spike_end_time is not None:
        _log_override("analysis", "spike_end_time", args.spike_end_time)
        cfg.setdefault("analysis", {})["spike_end_time"] = args.spike_end_time

    if args.spike_period:
        _log_override("analysis", "spike_periods", args.spike_period)
        cfg.setdefault("analysis", {})["spike_periods"] = args.spike_period

    if args.run_period:
        _log_override("analysis", "run_periods", args.run_period)
        cfg.setdefault("analysis", {})["run_periods"] = args.run_period

    if args.radon_interval:
        _log_override("analysis", "radon_interval", args.radon_interval)
        cfg.setdefault("analysis", {})["radon_interval"] = args.radon_interval

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
        if pd.api.types.is_datetime64_any_dtype(events_all["timestamp"]):
            events_all["timestamp"] = events_all["timestamp"].view("int64") / 1e9
    except Exception as e:
        print(f"ERROR: Could not load events from '{args.input}': {e}")
        sys.exit(1)

    if events_all.empty:
        print("No events found in the input CSV. Exiting.")
        sys.exit(0)

    # ``load_events()`` now returns timezone-aware datetimes; convert to epoch
    # seconds for internal calculations.

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
            logging.warning(
                f"Invalid noise_cutoff '{noise_thr}' - skipping noise cut"
            )
            noise_thr_val = None
        else:
            before = len(events_filtered)
            events_filtered = events_filtered[events_filtered["adc"] > noise_thr_val].reset_index(drop=True)
            n_removed_noise = before - len(events_filtered)
            logging.info(f"Noise cut removed {n_removed_noise} events")

    _ensure_events(events_filtered, "noise cut")

    # Optional burst filter to remove high-rate clusters
    total_span = events_filtered["timestamp"].max() - events_filtered["timestamp"].min()
    rate_cps = len(events_filtered) / max(total_span, 1e-9)
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
    events_filtered, n_removed_burst = apply_burst_filter(events_filtered, cfg, mode=burst_mode)
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
            t0_global = parse_time(t0_cfg)
            cfg.setdefault("analysis", {})["analysis_start_time"] = t0_global
        except Exception:
            logging.warning(
                f"Invalid analysis_start_time '{t0_cfg}' - using first event"
            )
            t0_global = events_filtered["timestamp"].min()
    else:
        t0_global = events_filtered["timestamp"].min()

    t_end_cfg = cfg.get("analysis", {}).get("analysis_end_time")
    t_end_global = None
    if t_end_cfg is not None:
        try:
            t_end_global = parse_time(t_end_cfg)
            cfg.setdefault("analysis", {})["analysis_end_time"] = t_end_global
        except Exception:
            logging.warning(
                f"Invalid analysis_end_time '{t_end_cfg}' - using last event"
            )
            t_end_global = None

    spike_end_cfg = cfg.get("analysis", {}).get("spike_end_time")
    t_spike_end = None
    if spike_end_cfg is not None:
        try:
            t_spike_end = parse_time(spike_end_cfg)
            cfg.setdefault("analysis", {})["spike_end_time"] = t_spike_end
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
            start_ts = parse_time(start)
            end_ts = parse_time(end)
            if end_ts <= start_ts:
                raise ValueError("end <= start")
            spike_periods.append((start_ts, end_ts))
        except Exception as e:
            logging.warning(f"Invalid spike_period {period} -> {e}")
    if spike_periods:
        cfg.setdefault("analysis", {})["spike_periods"] = [
            [s, e] for s, e in spike_periods
        ]

    run_periods_cfg = cfg.get("analysis", {}).get("run_periods", [])
    if run_periods_cfg is None:
        run_periods_cfg = []
    run_periods = []
    for period in run_periods_cfg:
        try:
            start, end = period
            start_ts = parse_time(start)
            end_ts = parse_time(end)
            if end_ts <= start_ts:
                raise ValueError("end <= start")
            run_periods.append((start_ts, end_ts))
        except Exception as e:
            logging.warning(f"Invalid run_period {period} -> {e}")
    if run_periods:
        cfg.setdefault("analysis", {})["run_periods"] = [
            [s, e] for s, e in run_periods
        ]

    radon_interval_cfg = cfg.get("analysis", {}).get("radon_interval")
    radon_interval = None
    if radon_interval_cfg:
        try:
            start_r, end_r = radon_interval_cfg
            start_r_ts = parse_time(start_r)
            end_r_ts = parse_time(end_r)
            if end_r_ts <= start_r_ts:
                raise ValueError("end <= start")
            radon_interval = (start_r_ts, end_r_ts)
            cfg.setdefault("analysis", {})["radon_interval"] = [start_r_ts, end_r_ts]
        except Exception as e:
            logging.warning(f"Invalid radon_interval {radon_interval_cfg} -> {e}")
            radon_interval = None

    # Apply optional time window cuts before any baseline or fit operations
    df_analysis = events_filtered.copy()
    if t_spike_end is not None:
        df_analysis = df_analysis[df_analysis["timestamp"] >= t_spike_end].reset_index(drop=True)
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
        if t_end_cfg is None and len(df_analysis) > 0:
            t_end_global = df_analysis["timestamp"].max()
    if t_end_global is not None:
        df_analysis = df_analysis[df_analysis["timestamp"] <= t_end_global].reset_index(drop=True)
    else:
        t_end_global = df_analysis["timestamp"].max()

    _ensure_events(df_analysis, "time-window selection")

    analysis_start = datetime.fromtimestamp(t0_global, tz=timezone.utc)
    analysis_end = datetime.fromtimestamp(t_end_global, tz=timezone.utc)

    # Optional ADC drift correction before calibration
    # Applied once using either the CLI value or the config default.
    drift_cfg = cfg.get("systematics", {})
    drift_rate = (
        float(args.slope)
        if args.slope is not None
        else float(drift_cfg.get("adc_drift_rate", 0.0))
    )
    drift_mode = "linear" if args.slope is not None else drift_cfg.get(
        "adc_drift_mode", "linear"
    )
    drift_params = drift_cfg.get("adc_drift_params")

    if drift_rate != 0.0 or drift_mode != "linear" or drift_params is not None:
        try:
            df_analysis["adc"] = apply_linear_adc_shift(
                df_analysis["adc"].values,
                df_analysis["timestamp"].values,
                float(drift_rate),
                t_ref=t0_global,
                mode=drift_mode,
                params=drift_params,
            )
        except Exception as e:
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
        calibration_valid = False
        cal_params = {"a": (0.005, 0.001), "c": (0.02, 0.005), "sigma_E": (0.3, 0.1)}

    # Save “a, c, sigma_E” so we can reconstruct energies
    a, a_sig = cal_params["a"]
    a2, a2_sig = cal_params.get("a2", (0.0, 0.0))
    c, c_sig = cal_params["c"]
    sigE_mean, sigE_sigma = cal_params["sigma_E"]
    cov_mat = np.asarray(cal_params.get("ac_covariance", [[0.0, 0.0], [0.0, 0.0]]), dtype=float)
    cov_ac = float(cov_mat[0, 1])
    cov_a_a2 = float(cal_params.get("cov_a_a2", 0.0))

    # Apply calibration -> new column “energy_MeV” and its uncertainty
    df_analysis["energy_MeV"] = apply_calibration(df_analysis["adc"], a, c, quadratic_coeff=a2)

    adc_vals = df_analysis["adc"].astype(float)
    var_energy = (
        (adc_vals * a_sig) ** 2
        + (adc_vals ** 2 * a2_sig) ** 2
        + c_sig ** 2
        + 2 * adc_vals * cov_ac
        + 2 * adc_vals ** 3 * cov_a_a2
    )
    df_analysis["denergy_MeV"] = np.sqrt(np.clip(var_energy, 0, None))

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
        t0_epoch = args.baseline_range[0].timestamp()
        t1_epoch = args.baseline_range[1].timestamp()
        logging.info(
            f"Baseline window (epoch seconds): {t0_epoch} \u2192 {t1_epoch}"
        )
        cfg.setdefault("baseline", {})["range"] = [t0_epoch, t1_epoch]
        baseline_range = [t0_epoch, t1_epoch]
    elif "range" in baseline_cfg:
        baseline_range = baseline_cfg.get("range")

    monitor_vol = float(baseline_cfg.get("monitor_volume_l", 605.0))
    sample_vol = float(baseline_cfg.get("sample_volume_l", 0.0))
    base_events = pd.DataFrame()
    baseline_live_time = 0.0
    mask_base = None

    if baseline_range:
        t_start_base = parse_time(baseline_range[0])
        t_end_base = parse_time(baseline_range[1])
        if t_end_base <= t_start_base:
            raise ValueError("baseline_range end time must be greater than start time")
        mask_base_full = (events_all["timestamp"] >= t_start_base) & (
            events_all["timestamp"] < t_end_base
        )
        mask_base = (df_analysis["timestamp"] >= t_start_base) & (
            df_analysis["timestamp"] < t_end_base
        )
        base_events = events_all[mask_base_full].copy()
        # Apply calibration to the baseline events
        if not base_events.empty:
            base_events["energy_MeV"] = apply_calibration(
                base_events["adc"], a, c, quadratic_coeff=a2
            )
            adc_b = base_events["adc"].astype(float)
            var_base = (
                (adc_b * a_sig) ** 2
                + (adc_b ** 2 * a2_sig) ** 2
                + c_sig ** 2
                + 2 * adc_b * cov_ac
                + 2 * adc_b ** 3 * cov_a_a2
            )
            base_events["denergy_MeV"] = np.sqrt(np.clip(var_base, 0, None))
        else:
            base_events["energy_MeV"] = np.array([], dtype=float)
            base_events["denergy_MeV"] = np.array([], dtype=float)
        if len(base_events) == 0:
            logging.warning(
                "baseline_range yielded zero events \u2013 skipping baseline subtraction"
            )
            baseline_live_time = 0.0
        else:
            baseline_live_time = float(t_end_base - t_start_base)
        cfg.setdefault("baseline", {})["range"] = [t_start_base, t_end_base]
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

            peak_adc = cal_params.get("peaks", {}).get("Po210", {}).get("centroid_adc")
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
            print(f"WARNING: Baseline noise estimation failed -> {e}")

        if noise_level is not None:
            # Store estimated noise peak amplitude in counts (not ADC units)
            baseline_info["noise_level"] = float(noise_level)

        # Record noise counts in ``baseline_counts``
        if "mask_noise" in locals():
            baseline_counts["noise"] = int(np.sum(mask_noise))

    _ensure_events(df_analysis, "baseline subtraction")

    if args.baseline_range:
        t_base0 = args.baseline_range[0]
        t_base1 = args.baseline_range[1]
        edges = adc_hist_edges(df_analysis["adc"].values, hist_bins)
        df_analysis = subtract_baseline(
            df_analysis,
            events_all,
            bins=edges,
            t_base0=t_base0,
            t_base1=t_base1,
            mode=args.baseline_mode,
            live_time_analysis=(analysis_end - analysis_start).total_seconds(),
        )

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
                    int(
                        np.ceil(
                            (E_all.max() - E_all.min()) / float(fd_width)
                        )
                    ),
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
                df_analysis["energy_MeV"].values, df_analysis["denergy_MeV"].values, lo, hi
            )
            iso_mask = probs > 0
            iso_events = df_analysis[iso_mask].copy()
            iso_events["weight"] = probs[iso_mask]
            if iso_events.empty:
                print(f"WARNING: No events found for {iso} in [{lo}, {hi}] MeV.")
                continue

        # Build priors for time fit
        priors_time = {}

        # Efficiency prior per isotope
        eff_val = cfg["time_fit"].get(
            f"eff_{iso.lower()}", [1.0, 0.0]
        )
        priors_time["eff"] = tuple(eff_val)

        # Half-life prior (user must supply [T₁/₂, σ(T₁/₂)] in seconds)
        hl_key = f"hl_{iso.lower()}"
        if hl_key in cfg["time_fit"]:
            T12, T12sig = cfg["time_fit"][hl_key]
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

            eff = cfg["time_fit"].get(
                f"eff_{iso.lower()}", [1.0]
            )[0]
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
            live_time_analysis = (analysis_end - analysis_start).total_seconds()
            if iso in isotopes_to_subtract and baseline_live_time > 0 and eff > 0:
                c_rate, c_sigma = subtract_radon_baseline(
                    analysis_counts,
                    eff,
                    live_time_analysis,
                    baseline_counts.get(iso, 0.0),
                    baseline_live_time,
                )
            else:
                if eff > 0 and live_time_analysis > 0:
                    c_rate = analysis_counts / (live_time_analysis * eff)
                    c_sigma = math.sqrt(analysis_counts) / (
                        live_time_analysis * eff
                    )
                else:
                    c_rate = 0.0
                    c_sigma = 0.0
            baseline_info.setdefault("corrected_activity", {})[iso] = {
                "value": c_rate,
                "uncertainty": c_sigma,
            }
        else:
            priors_time["N0"] = (
                0.0,
                cfg["time_fit"].get(
                    f"sig_n0_{iso.lower()}",
                    cfg["time_fit"].get(f"sig_N0_{iso}", 1.0),
                ),
            )

            analysis_counts = float(np.sum(iso_events["weight"]))
            live_time_analysis = (analysis_end - analysis_start).total_seconds()
            eff = cfg["time_fit"].get(
                f"eff_{iso.lower()}", [1.0]
            )[0]
            if eff > 0 and live_time_analysis > 0:
                c_rate = analysis_counts / (live_time_analysis * eff)
                c_sigma = math.sqrt(analysis_counts) / (
                    live_time_analysis * eff
                )
            else:
                c_rate = 0.0
                c_sigma = 0.0
            baseline_info.setdefault("corrected_activity", {})[iso] = {
                "value": c_rate,
                "uncertainty": c_sigma,
            }

        # Store priors for use in systematics scanning
        priors_time_all[iso] = priors_time

        # Build configuration for fit_time_series
        if args.settle_s is not None:
            cut = t0_global + float(args.settle_s)
            iso_events = iso_events[iso_events["timestamp"] >= cut]
        times_dict = {iso: iso_events["timestamp"].values}
        weights_map = {iso: iso_events["weight"].values}
        fit_cfg = {
            "isotopes": {
                iso: {
                    "half_life_s": cfg["time_fit"].get(
                        f"hl_{iso.lower()}", [np.nan]
                    )[0],
                    "efficiency": cfg["time_fit"].get(
                        f"eff_{iso.lower()}", [1.0]
                    )[0],
                }
            },
            "fit_background": not cfg["time_fit"]["flags"].get(
                "fix_background_b", False
            ),
            "fit_initial": not cfg["time_fit"]["flags"].get(f"fix_N0_{iso.lower()}", False),
            "background_guess": cfg["time_fit"].get("background_guess", 0.0),
            "n0_guess_fraction": cfg["time_fit"].get("n0_guess_fraction", 0.1),
        }

        # Run time-series fit
        decay_out = None  # fresh variable each iteration
        try:
            t_start_fit = t0_global
            if args.settle_s is not None:
                t_start_fit = t0_global + float(args.settle_s)
            try:
                decay_out = fit_time_series(
                    times_dict,
                    t_start_fit,
                    t_end_global,
                    fit_cfg,
                    weights=weights_map,
                    strict=args.strict_covariance,
                )
            except TypeError:
                decay_out = fit_time_series(
                    times_dict,
                    t_start_fit,
                    t_end_global,
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

    # Also extract Po-210 events for plotting if a window is provided
    win_p210 = cfg.get("time_fit", {}).get("window_po210")
    if win_p210 is not None:
        lo, hi = win_p210
        mask210 = (
            (df_analysis["energy_MeV"] >= lo)
            & (df_analysis["energy_MeV"] <= hi)
            & (df_analysis["timestamp"] >= t0_global)
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
                times_dict = {iso: filtered_df["timestamp"].values}
                weights_local = {iso: probs[mask]}
                cfg_fit = {
                    "isotopes": {
                        iso: {
                            "half_life_s": cfg["time_fit"].get(f"hl_{iso.lower()}", [np.nan])[0],
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
                        t0_global,
                        t_end_global,
                        cfg_fit,
                        weights=weights_local,
                        strict=args.strict_covariance,
                    )
                except TypeError:
                    out = fit_time_series(
                        times_dict,
                        t0_global,
                        t_end_global,
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
    baseline_rates = {}
    baseline_unc = {}
    if baseline_live_time > 0:
        for iso, count in baseline_counts.items():
            eff = cfg["time_fit"].get(
                f"eff_{iso.lower()}", [1.0]
            )[0]
            if eff > 0:
                rate = count / (baseline_live_time * eff)
                err = np.sqrt(count) / (baseline_live_time * eff)
            else:
                rate = 0.0
                err = 0.0
            baseline_rates[iso] = rate  # Bq
            baseline_unc[iso] = err

    dilution_factor = 0.0
    if monitor_vol + sample_vol > 0:
        dilution_factor = monitor_vol / (monitor_vol + sample_vol)

    scales = {
        "Po214": dilution_factor,
        "Po218": dilution_factor,
        "Po210": 1.0,
        "noise": 1.0,
    }
    baseline_info["scales"] = scales

    for iso, rate in baseline_rates.items():
        fit = time_fit_results.get(iso)
        params = _fit_params(fit)
        if params and (f"E_{iso}" in params):
            s = scales.get(iso, 1.0)
            params["E_corrected"] = params[f"E_{iso}"] - s * rate
            err_fit = params.get(f"dE_{iso}", 0.0)
            sigma_rate = 0.0
            if baseline_live_time > 0:
                count = baseline_counts.get(iso, 0.0)
                eff = cfg["time_fit"].get(
                    f"eff_{iso.lower()}", [1.0]
                )[0]
                if eff > 0:
                    sigma_rate = math.sqrt(count) / (baseline_live_time * eff)
            params["dE_corrected"] = float(math.hypot(err_fit, sigma_rate * s))

    if baseline_rates:
        baseline_info["rate_Bq"] = baseline_rates
        baseline_info["rate_unc_Bq"] = baseline_unc
        baseline_info["dilution_factor"] = dilution_factor

    # ────────────────────────────────────────────────────────────
    # Radon activity extrapolation
    # ────────────────────────────────────────────────────────────
    from radon_activity import compute_radon_activity, compute_total_radon

    radon_results = {}
    eff_po214 = cfg.get("time_fit", {}).get("eff_po214", [1.0])[0]
    eff_po218 = cfg.get("time_fit", {}).get("eff_po218", [1.0])[0]

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

    # Convert activity to a concentration per liter of monitor volume and the
    # total amount of radon present in just the assay sample.
    conc, dconc, total_bq, dtotal_bq = compute_total_radon(
        A_radon,
        dA_radon,
        monitor_vol,
        sample_vol,
    )

    radon_results["radon_activity_Bq"] = {"value": A_radon, "uncertainty": dA_radon}
    radon_results["radon_concentration_Bq_per_L"] = {
        "value": conc,
        "uncertainty": dconc,
    }
    radon_results["total_radon_in_sample_Bq"] = {
        "value": total_bq,
        "uncertainty": dtotal_bq,
    }

    if radon_interval is not None:
        from radon_activity import radon_delta

        t_start_rel = radon_interval[0] - t0_global
        t_end_rel = radon_interval[1] - t0_global

        delta214 = err_delta214 = None
        if "Po214" in time_fit_results:
            fit_result = time_fit_results["Po214"]
            fit = _fit_params(fit_result)
            E = fit.get("E_corrected", fit.get("E_Po214"))
            dE = fit.get("dE_corrected", fit.get("dE_Po214", 0.0))
            N0 = fit.get("N0_Po214", 0.0)
            dN0 = fit.get("dN0_Po214", 0.0)
            default_const = cfg.get("nuclide_constants", {})
            default_hl = default_const.get("Po214", PO214).half_life_s
            hl = cfg.get("time_fit", {}).get("hl_po214", [default_hl])[0]
            cov = _cov_entry(fit_result, "E_Po214", "N0_Po214")
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
            default_const = cfg.get("nuclide_constants", {})
            default_hl = default_const.get("Po218", PO218).half_life_s
            hl = cfg.get("time_fit", {}).get("hl_po218", [default_hl])[0]
            cov = _cov_entry(fit_result, "E_Po218", "N0_Po218")
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

    summary = {
        "timestamp": now_str,
        "config_used": args.config.name,
        "calibration": cal_params,
        "calibration_valid": calibration_valid,
        "spectral_fit": spec_dict,
        "time_fit": time_fit_serializable,
        "systematics": systematics_results,
        "baseline": baseline_info,
        "radon_results": radon_results,
        "noise_cut": {"removed_events": int(n_removed_noise)},
        "burst_filter": {
            "removed_events": int(n_removed_burst),
            "burst_mode": burst_mode,
        },
        "adc_drift_rate": drift_rate,
        "adc_drift_mode": drift_mode,
        "adc_drift_params": drift_params,
        "efficiency": efficiency_results,
        "random_seed": seed_used,
        "git_commit": commit,
        "requirements_sha256": requirements_sha256,
        "cli_sha256": cli_sha256,
        "cli_args": cli_args,
        "analysis": {
            "analysis_start_time": t0_cfg,
            "analysis_end_time": t_end_cfg,
            "spike_end_time": spike_end_cfg,
            "spike_periods": spike_periods_cfg,
            "run_periods": run_periods_cfg,
            "radon_interval": radon_interval_cfg,
            "ambient_concentration": cfg.get("analysis", {}).get(
                "ambient_concentration"
            ),
            "settle_s": cfg.get("analysis", {}).get("settle_s"),
        },
    }

    if weights is not None:
        summary["efficiency"]["blue_weights"] = list(weights)

    results_dir = Path(args.output_dir) / (args.job_id or now_str)
    if results_dir.exists():
        if args.overwrite:
            shutil.rmtree(results_dir)
        else:
            raise FileExistsError(f"Results folder already exists: {results_dir}")

    copy_config(results_dir, cfg, exist_ok=args.overwrite)
    out_dir = write_summary(results_dir, summary)

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
            _ = plot_time_series(
                all_timestamps=ts_times,
                all_energies=ts_energy,
                fit_results=fit_dict,
                t_start=t0_global,
                t_end=t_end_global,
                config=plot_cfg,
                out_png=Path(out_dir) / f"time_series_{iso}.png",
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

        times = np.linspace(t0_global, t_end_global, 100)
        t_rel = times - t0_global

        A214 = dA214 = None
        if "Po214" in time_fit_results:
            fit_result = time_fit_results["Po214"]
            fit = _fit_params(fit_result)
            E = fit.get("E_corrected", fit.get("E_Po214"))
            dE = fit.get("dE_corrected", fit.get("dE_Po214", 0.0))
            N0 = fit.get("N0_Po214", 0.0)
            dN0 = fit.get("dN0_Po214", 0.0)
            default_const = cfg.get("nuclide_constants", {})
            default_hl = default_const.get("Po214", PO214).half_life_s
            hl = cfg.get("time_fit", {}).get("hl_po214", [default_hl])[0]
            cov = _cov_entry(fit_result, "E_Po214", "N0_Po214")
            A214, dA214 = radon_activity_curve(t_rel, E, dE, N0, dN0, hl, cov)
            plot_radon_activity(
                times,
                A214,
                dA214,
                Path(out_dir) / "radon_activity_po214.png",
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
            default_const = cfg.get("nuclide_constants", {})
            default_hl = default_const.get("Po218", PO218).half_life_s
            hl = cfg.get("time_fit", {}).get("hl_po218", [default_hl])[0]
            cov = _cov_entry(fit_result, "E_Po218", "N0_Po218")
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
            err_arr,
            Path(out_dir) / "radon_activity.png",
            config=cfg.get("plotting", {}),
        )

        if radon_interval is not None:
            times_trend = np.linspace(radon_interval[0], radon_interval[1], 50)
            rel_trend = times_trend - t0_global
            A214_tr = None
            if "Po214" in time_fit_results:
                fit_result = time_fit_results["Po214"]
                fit = _fit_params(fit_result)
                E214 = fit.get("E_corrected", fit.get("E_Po214"))
                dE214 = fit.get("dE_corrected", fit.get("dE_Po214", 0.0))
                N0214 = fit.get("N0_Po214", 0.0)
                dN0214 = fit.get("dN0_Po214", 0.0)
                default_const = cfg.get("nuclide_constants", {})
                default_hl = default_const.get("Po214", PO214).half_life_s
                hl214 = cfg.get("time_fit", {}).get("hl_po214", [default_hl])[0]
                cov214 = _cov_entry(fit_result, "E_Po214", "N0_Po214")
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
                default_const = cfg.get("nuclide_constants", {})
                default_hl = default_const.get("Po218", PO218).half_life_s
                hl218 = cfg.get("time_fit", {}).get("hl_po218", [default_hl])[0]
                cov218 = _cov_entry(fit_result, "E_Po218", "N0_Po218")
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
