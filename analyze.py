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
import os
import sys
import logging
import random
from datetime import datetime, timezone
import subprocess
import hashlib
import json

import numpy as np
import pandas as pd
from scipy.stats import norm

# ‣ Import our supporting modules (all must live in the same folder).
from io_utils import (
    load_config,
    copy_config,
    load_events,
    write_summary,
    apply_burst_filter,
)
from calibration import derive_calibration_constants, derive_calibration_constants_auto
from fitting import fit_spectrum, fit_time_series
from plot_utils import (
    plot_spectrum,
    plot_time_series,
    plot_radon_activity,
    plot_equivalent_air,
)
from systematics import scan_systematics, apply_linear_adc_shift
from visualize import cov_heatmap, efficiency_bar
from utils import find_adc_peaks, cps_to_bq


def window_prob(E, sigma, lo, hi):
    """Return probability that each ``E`` lies in [lo, hi].

    Elements with ``sigma == 0`` are evaluated via a simple range check instead
    of calling :func:`scipy.stats.norm.cdf` with ``scale=0``.
    Parameters may be scalar or array-like and are broadcast element-wise.
    """

    E = np.asarray(E, dtype=float)
    sigma = np.asarray(sigma, dtype=float)
    E, sigma = np.broadcast_arrays(E, sigma)
    lo_val = float(lo) if np.isscalar(lo) else float(lo)
    hi_val = float(hi) if np.isscalar(hi) else float(hi)

    prob = np.empty_like(E, dtype=float)
    zero_mask = sigma == 0

    if np.any(zero_mask):
        prob[zero_mask] = (
            (E[zero_mask] >= lo_val) & (E[zero_mask] <= hi_val)
        ).astype(float)

    if np.any(~zero_mask):
        nz = ~zero_mask
        prob[nz] = norm.cdf(hi_val, loc=E[nz], scale=sigma[nz]) - norm.cdf(
            lo_val, loc=E[nz], scale=sigma[nz]
        )

    if prob.ndim == 0:
        return float(prob)
    return prob


def parse_args():
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
        help=(
            "Optional baseline-run interval. "
            "Provide two values (either ISO strings or epoch floats). "
            "If set, those events are extracted (same energy cuts) and "
            "listed in `baseline` of the summary."
        ),
    )
    p.add_argument(
        "--burst-mode",
        choices=["none", "micro", "rate", "both"],
        help="Burst filtering mode to pass to apply_burst_filter (overrides config)",
    )
    p.add_argument(
        "--job-id",
        help="Optional identifier used for the results folder instead of the timestamp",
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
        help="Ignore events occurring after this ISO timestamp",
    )
    p.add_argument(
        "--spike-end-time",
        help="Discard events before this ISO timestamp",
    )
    p.add_argument(
        "--spike-period",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="Discard events between START and END (can be given multiple times)",
    )
    p.add_argument(
        "--run-period",
        nargs=2,
        action="append",
        metavar=("START", "END"),
        help="Keep events between START and END (can be given multiple times)",
    )
    p.add_argument(
        "--radon-interval",
        nargs=2,
        metavar=("START", "END"),
        help="Time interval to evaluate radon delta",
    )
    p.add_argument(
        "--slope",
        type=float,
        help="Apply a linear ADC drift correction with the given slope",
    )
    p.add_argument(
        "--settle-s",
        type=float,
        help="Discard events occurring this many seconds after the start",
    )
    p.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging",
    )
    p.add_argument(
        "--time-bin-mode",
        choices=["auto", "fd", "fixed"],
        help="Time-series binning mode (overrides plotting.plot_time_binning_mode)",
    )
    p.add_argument(
        "--time-bin-width",
        type=float,
        help="Fixed time bin width in seconds (overrides plotting.plot_time_bin_width_s)",
    )
    p.add_argument(
        "--dump-ts-json",
        action="store_true",
        help="Write *_ts.json files containing binned time-series data",
    )
    p.add_argument(
        "--ambient-file",
        help=(
            "Two-column text file of timestamp and ambient concentration in Bq/L"
        ),
    )
    p.add_argument(
        "--ambient-concentration",
        type=float,
        help="Ambient radon concentration in Bq per liter for equivalent air plot",
    )
    p.add_argument(
        "--seed",
        type=int,
        help="Override random seed used by analysis algorithms",
    )
    return p.parse_args()


def main():
    cli_args = sys.argv[:]
    cli_sha256 = hashlib.sha256(" ".join(cli_args).encode("utf-8")).hexdigest()
    try:
        commit = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], encoding="utf-8")
            .strip()
        )
    except Exception:
        commit = "unknown"

    args = parse_args()

    # ────────────────────────────────────────────────────────────
    # 1. Load configuration
    # ────────────────────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Could not load config '{args.config}': {e}")
        sys.exit(1)

    # Apply optional overrides from command-line arguments
    if args.efficiency_json:
        try:
            with open(args.efficiency_json, "r", encoding="utf-8") as f:
                cfg["efficiency"] = json.load(f)
        except Exception as e:
            print(f"ERROR: Could not load efficiency JSON '{args.efficiency_json}': {e}")
            sys.exit(1)

    if args.systematics_json:
        try:
            with open(args.systematics_json, "r", encoding="utf-8") as f:
                cfg["systematics"] = json.load(f)
        except Exception as e:
            print(f"ERROR: Could not load systematics JSON '{args.systematics_json}': {e}")
            sys.exit(1)

    if args.seed is not None:
        cfg.setdefault("pipeline", {})["random_seed"] = int(args.seed)

    if args.ambient_concentration is not None:
        cfg.setdefault("analysis", {})["ambient_concentration"] = float(
            args.ambient_concentration
        )

    if args.analysis_end_time is not None:
        cfg.setdefault("analysis", {})["analysis_end_time"] = args.analysis_end_time

    if args.spike_end_time is not None:
        cfg.setdefault("analysis", {})["spike_end_time"] = args.spike_end_time

    if args.spike_period:
        cfg.setdefault("analysis", {})["spike_periods"] = args.spike_period

    if args.run_period:
        cfg.setdefault("analysis", {})["run_periods"] = args.run_period

    if args.radon_interval:
        cfg.setdefault("analysis", {})["radon_interval"] = args.radon_interval


    if args.time_bin_mode:
        cfg.setdefault("plotting", {})["plot_time_binning_mode"] = args.time_bin_mode
    if args.time_bin_width is not None:
        cfg.setdefault("plotting", {})["plot_time_bin_width_s"] = float(args.time_bin_width)
    if args.dump_ts_json:
        cfg.setdefault("plotting", {})["dump_time_series_json"] = True

    if args.spike_count is not None or args.spike_count_err is not None:
        eff_sec = cfg.setdefault("efficiency", {}).setdefault("spike", {})
        if args.spike_count is not None:
            eff_sec["counts"] = float(args.spike_count)
        if args.spike_count_err is not None:
            eff_sec["error"] = float(args.spike_count_err)

    if args.slope is not None:
        cfg.setdefault("systematics", {})["adc_drift_rate"] = float(args.slope)

    if args.debug:
        cfg.setdefault("pipeline", {})["log_level"] = "DEBUG"

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
        events = load_events(args.input)
    except Exception as e:
        print(f"ERROR: Could not load events from '{args.input}': {e}")
        sys.exit(1)

    if events.empty:
        print("No events found in the input CSV. Exiting.")
        sys.exit(0)

    # Ensure “timestamp” column is float‐seconds since epoch
    # If user provided ISO‐strings, convert to epoch:
    if events["timestamp"].dtype == object:
        events["timestamp"] = (
            pd.to_datetime(events["timestamp"], utc=True).astype(np.int64) / 1e9
        )

    events["timestamp"] = events["timestamp"].astype(float)

    # Optional burst filter to remove high-rate clusters
    burst_mode = (
        args.burst_mode
        if args.burst_mode is not None
        else cfg.get("burst_filter", {}).get("burst_mode", "rate")
    )
    events, n_removed_burst = apply_burst_filter(events, cfg, mode=burst_mode)

    # Global t₀ reference
    t0_cfg = cfg.get("analysis", {}).get("analysis_start_time")
    if t0_cfg is not None:
        try:
            t0_global = pd.to_datetime(t0_cfg, utc=True).timestamp()
        except Exception:
            logging.warning(
                f"Invalid analysis_start_time '{t0_cfg}' - using first event"
            )
            t0_global = events["timestamp"].min()
    else:
        t0_global = events["timestamp"].min()

    def _to_epoch(val):
        try:
            return float(val)
        except Exception:
            return pd.to_datetime(val, utc=True).timestamp()

    t_end_cfg = cfg.get("analysis", {}).get("analysis_end_time")
    t_end_global = None
    if t_end_cfg is not None:
        try:
            t_end_global = _to_epoch(t_end_cfg)
        except Exception:
            logging.warning(
                f"Invalid analysis_end_time '{t_end_cfg}' - using last event"
            )
            t_end_global = None

    spike_end_cfg = cfg.get("analysis", {}).get("spike_end_time")
    t_spike_end = None
    if spike_end_cfg is not None:
        try:
            t_spike_end = _to_epoch(spike_end_cfg)
        except Exception:
            logging.warning(
                f"Invalid spike_end_time '{spike_end_cfg}' - ignoring"
            )
            t_spike_end = None

    spike_periods_cfg = cfg.get("analysis", {}).get("spike_periods", [])
    if spike_periods_cfg is None:
        spike_periods_cfg = []
    spike_periods = []
    for period in spike_periods_cfg:
        try:
            start, end = period
            start_ts = _to_epoch(start)
            end_ts = _to_epoch(end)
            if end_ts <= start_ts:
                raise ValueError("end <= start")
            spike_periods.append((start_ts, end_ts))
        except Exception as e:
            logging.warning(f"Invalid spike_period {period} -> {e}")

    run_periods_cfg = cfg.get("analysis", {}).get("run_periods", [])
    if run_periods_cfg is None:
        run_periods_cfg = []
    run_periods = []
    for period in run_periods_cfg:
        try:
            start, end = period
            start_ts = _to_epoch(start)
            end_ts = _to_epoch(end)
            if end_ts <= start_ts:
                raise ValueError("end <= start")
            run_periods.append((start_ts, end_ts))
        except Exception as e:
            logging.warning(f"Invalid run_period {period} -> {e}")

    radon_interval_cfg = cfg.get("analysis", {}).get("radon_interval")
    radon_interval = None
    if radon_interval_cfg:
        try:
            start_r, end_r = radon_interval_cfg
            start_r_ts = _to_epoch(start_r)
            end_r_ts = _to_epoch(end_r)
            if end_r_ts <= start_r_ts:
                raise ValueError("end <= start")
            radon_interval = (start_r_ts, end_r_ts)
        except Exception as e:
            logging.warning(f"Invalid radon_interval {radon_interval_cfg} -> {e}")
            radon_interval = None

    # Apply optional time window cuts before any baseline or fit operations
    if t_spike_end is not None:
        events = events[events["timestamp"] >= t_spike_end].reset_index(drop=True)
    for start_ts, end_ts in spike_periods:
        mask = (events["timestamp"] >= start_ts) & (events["timestamp"] < end_ts)
        if mask.any():
            events = events[~mask].reset_index(drop=True)
    if run_periods:
        keep_mask = np.zeros(len(events), dtype=bool)
        for start_ts, end_ts in run_periods:
            keep_mask |= (events["timestamp"] >= start_ts) & (events["timestamp"] < end_ts)
        events = events[keep_mask].reset_index(drop=True)
        if t_end_cfg is None and len(events) > 0:
            t_end_global = events["timestamp"].max()
    if t_end_global is not None:
        events = events[events["timestamp"] <= t_end_global].reset_index(drop=True)
    else:
        t_end_global = events["timestamp"].max()

    # Optional ADC drift correction before calibration
    # Applied once using either the CLI value or the config default.
    drift_rate = float(args.slope) if args.slope is not None else float(
        cfg.get("systematics", {}).get("adc_drift_rate", 0.0)
    )

    if drift_rate != 0.0:
        try:
            events["adc"] = apply_linear_adc_shift(
                events["adc"].values,
                events["timestamp"].values,
                float(drift_rate),
                t_ref=t0_global,
            )
        except Exception as e:
            print(f"WARNING: Could not apply ADC drift correction -> {e}")


    # ────────────────────────────────────────────────────────────
    # 3. Energy calibration
    # ────────────────────────────────────────────────────────────
    adc_vals = events["adc"].values

    try:
        if cfg.get("calibration", {}).get("method", "two-point") == "auto":
            # Auto‐cal using Freedman‐Diaconis histogram + peak detection
            cal_params = derive_calibration_constants_auto(
                adc_vals,
                noise_cutoff=cfg["calibration"].get("noise_cutoff", 300),
                hist_bins=cfg["calibration"].get("hist_bins", 2000),
                peak_search_radius=cfg["calibration"].get("peak_search_radius", 200),
                nominal_adc=cfg["calibration"].get("nominal_adc"),
            )
        else:
            # Two‐point calibration as given in config
            cal_params = derive_calibration_constants(adc_vals, config=cfg)
    except RuntimeError as e:
        print(f"WARNING: calibration failed – {e}. Using defaults.")
        cal_params = {"a": (0.005, 0.001), "c": (0.02, 0.005), "sigma_E": (0.3, 0.1)}

    # Save “a, c, sigma_E” so we can reconstruct energies
    a, a_sig = cal_params["a"]
    c, c_sig = cal_params["c"]
    sigE_mean, sigE_sigma = cal_params["sigma_E"]

    # Apply linear calibration -> new column “energy_MeV” and its uncertainty
    events["energy_MeV"] = events["adc"] * a + c
    events["denergy_MeV"] = np.sqrt((events["adc"] * a_sig) ** 2 + c_sig ** 2)

    # ────────────────────────────────────────────────────────────
    # 4. Baseline run (optional)
    # ────────────────────────────────────────────────────────────
    baseline_info = {}
    baseline_cfg = cfg.get("baseline", {})
    baseline_range = None
    if args.baseline_range:
        baseline_range = args.baseline_range
    elif "range" in baseline_cfg:
        baseline_range = baseline_cfg.get("range")

    monitor_vol = float(baseline_cfg.get("monitor_volume_l", 605.0))
    sample_vol = float(baseline_cfg.get("sample_volume_l", 0.0))
    base_events = pd.DataFrame()
    baseline_live_time = 0.0
    mask_base = None

    if baseline_range:

        def to_epoch(x):
            try:
                return float(x)
            except Exception:
                return pd.to_datetime(x, utc=True).timestamp()

        t_start_base = to_epoch(baseline_range[0])
        t_end_base = to_epoch(baseline_range[1])
        if t_end_base <= t_start_base:
            raise ValueError(
                "baseline_range end time must be greater than start time"
            )
        mask_base = (events["timestamp"] >= t_start_base) & (
            events["timestamp"] < t_end_base
        )
        base_events = events[mask_base].copy()
        baseline_live_time = float(t_end_base - t_start_base)
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

            peak_adc = (
                cal_params.get("peaks", {})
                .get("Po210", {})
                .get("centroid_adc")
            )
            if peak_adc is not None:
                noise_level, _ = estimate_baseline_noise(
                    base_events["adc"].values,
                    peak_adc=peak_adc,
                )
        except Exception as e:
            print(f"WARNING: Baseline noise estimation failed -> {e}")

        if noise_level is not None:
            baseline_info["noise_level"] = float(noise_level)



        # Remove baseline events from the main dataset before any fits.
        # This is done once here to avoid accidentally discarding data twice
        # which previously left an empty DataFrame for the time fits.


    # After creating ``base_events``, drop them from the dataset
    if baseline_range:
        # Remove rows where ``mask_base`` is True
        events = events[~mask_base].reset_index(drop=True)

        if t_end_cfg is None:
            t_end_global = events["timestamp"].max()




        # Baseline events were already removed above. Avoid reapplying the mask
        # here since it may be misaligned after ``events`` has been
        # reindexed, which can inadvertently drop all remaining rows on
        # newer pandas versions.




    baseline_counts = {}
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
            E_all = events["energy_MeV"].values
            # Freedman‐Diaconis on energy array
            q25, q75 = np.percentile(E_all, [25, 75])
            iqr = q75 - q25
            n = E_all.size
            if (iqr > 0) and (n > 0):
                fd_width = 2 * iqr / (n ** (1 / 3))
                emin, emax = E_all.min(), E_all.max()
                nbins = max(1, int(np.ceil((emax - emin) / fd_width)))
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
            adc_min = events["adc"].min()
            adc_max = events["adc"].max()
            bins = int(np.ceil((adc_max - adc_min + 1) / width))

            # Build edges in ADC units then convert to energy for plotting
            bin_edges_adc = np.arange(adc_min, adc_min + bins * width + 1, width)
            bin_edges = bin_edges_adc * a + c

        # Find approximate ADC centroids for Po‐210, Po‐218, Po‐214

        if "expected_peaks" not in cfg.get("spectral_fit", {}):
            raise KeyError("'spectral_fit.expected_peaks' must be provided in the configuration")

        expected_peaks = cfg["spectral_fit"]["expected_peaks"]

        # `find_adc_peaks` will return a dict: e.g. { "Po210": adc_centroid, … }
        adc_peaks = find_adc_peaks(
            events["adc"].values,
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
            mu = centroid_adc * a + c  # convert to MeV
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
                    (events["energy_MeV"] >= mu - peak_tol)
                    & (events["energy_MeV"] <= mu + peak_tol)
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
                events["energy_MeV"].values,
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
                "energies": events["energy_MeV"].values,
                "priors": priors_spec,
                "flags": spec_flags,
            }
            if cfg["spectral_fit"].get("use_plot_bins_for_fit", False):
                fit_kwargs.update({"bins": bins, "bin_edges": bin_edges})
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
        spec_plot_data = {
            "energies": events["energy_MeV"].values,
            "fit_vals": spec_fit_out if spectrum_results else None,
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
            win_key = f"window_{iso}"

            # Missing energy window for this isotope -> skip gracefully
            win_range = cfg.get("time_fit", {}).get(win_key)
            if win_range is None:
                print(
                    f"INFO: Config key '{win_key}' not found. Skipping time fit for {iso}."
                )
                continue

            lo, hi = win_range
            probs = window_prob(events["energy_MeV"].values, events["denergy_MeV"].values, lo, hi)
            iso_mask = probs > 0
            iso_events = events[iso_mask].copy()
            iso_events["weight"] = probs[iso_mask]
            if iso_events.empty:
                print(f"WARNING: No events found for {iso} in [{lo}, {hi}] MeV.")
                continue

        # Build priors for time fit
        priors_time = {}

        # Efficiency prior per isotope
        eff_val = cfg["time_fit"].get(f"eff_{iso}", [1.0, 0.0])
        priors_time["eff"] = tuple(eff_val)

        # Half-life prior (user must supply [T₁/₂, σ(T₁/₂)] in seconds)
        hl_key = f"hl_{iso}"
        if hl_key in cfg["time_fit"]:
            T12, T12sig = cfg["time_fit"][hl_key]
            priors_time["tau"] = (T12 / np.log(2), T12sig / np.log(2))

        # Background‐rate prior
        if f"bkg_{iso}" in cfg["time_fit"]:
            priors_time["B0"] = tuple(cfg["time_fit"][f"bkg_{iso}"])

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
            baseline_counts[iso] = n0_count

            eff = cfg["time_fit"].get(f"eff_{iso}", [1.0])[0]
            if baseline_live_time > 0 and eff > 0:
                n0_activity = n0_count / (baseline_live_time * eff)
                n0_sigma = np.sqrt(n0_count) / (baseline_live_time * eff)
            else:
                n0_activity = 0.0
                n0_sigma = 1.0

            priors_time["N0"] = (
                n0_activity,
                cfg["time_fit"].get(f"sig_N0_{iso}", n0_sigma),
            )
        else:
            priors_time["N0"] = (
                0.0,
                cfg["time_fit"].get(f"sig_N0_{iso}", 1.0),
            )

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
                    "half_life_s": cfg["time_fit"][f"hl_{iso}"][0],
                    "efficiency": cfg["time_fit"][f"eff_{iso}"][0],
                }
            },
            "fit_background": not cfg["time_fit"]["flags"].get(
                "fix_background_b", False
            ),
            "fit_initial": not cfg["time_fit"]["flags"].get(f"fix_N0_{iso}", False),
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
                )
            except TypeError:
                decay_out = fit_time_series(
                    times_dict,
                    t_start_fit,
                    t_end_global,
                    fit_cfg,
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

    # ────────────────────────────────────────────────────────────
    # 7. Systematics scan (optional)
    # ────────────────────────────────────────────────────────────
    systematics_results = {}
    if cfg.get("systematics", {}).get("enable", False):
        sigma_dict = cfg["systematics"].get("sigma_shifts", {})
        keys = cfg["systematics"].get("scan_keys", [])

        for iso, fit_out in time_fit_results.items():
            if not fit_out:
                continue

            # Build a wrapper to re‐run fit_time_series with modified priors
            def fit_wrapper(priors_mod):
                win_range = cfg.get("time_fit", {}).get(f"window_{iso}")
                if win_range is None:
                    raise ValueError(
                        f"Missing window for {iso} during systematics scan"
                    )
                probs = window_prob(
                    events["energy_MeV"].values,
                    events["denergy_MeV"].values,
                    win_range[0],
                    win_range[1],
                )
                mask = probs > 0
                filtered_df = events[mask]
                times_dict = {iso: filtered_df["timestamp"].values}
                weights_local = {iso: probs[mask]}
                cfg_fit = {
                    "isotopes": {
                        iso: {
                            "half_life_s": cfg["time_fit"][f"hl_{iso}"][0],
                            "efficiency": priors_mod["eff"][0],
                        }
                    },
                    "fit_background": not cfg["time_fit"]["flags"].get(
                        "fix_background_b", False
                    ),
                    "fit_initial": not cfg["time_fit"]["flags"].get(
                        f"fix_N0_{iso}", False
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
                    )
                except TypeError:
                    out = fit_time_series(
                        times_dict,
                        t0_global,
                        t_end_global,
                        cfg_fit,
                    )
                # Return the full fit result so scan_systematics can
                # extract whichever parameter it needs.
                return out

            try:
                deltas, total_unc = scan_systematics(
                    fit_wrapper, priors_time_all.get(iso, {}), sigma_dict, keys
                )
                systematics_results[iso] = {"deltas": deltas, "total_unc": total_unc}
            except Exception as e:
                print(f"WARNING: Systematics scan for {iso} -> {e}")

    # ────────────────────────────────────────────────────────────
    # 7b. Optional efficiency calculations
    # ────────────────────────────────────────────────────────────
    efficiency_results = {}
    eff_cfg = cfg.get("efficiency", {})
    if eff_cfg:
        from efficiency import (
            calc_spike_efficiency,
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
                    val = calc_spike_efficiency(
                        scfg["counts"], scfg["activity_bq"], scfg["live_time_s"]
                    )
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
    if baseline_live_time > 0:
        for iso, count in baseline_counts.items():
            eff = cfg["time_fit"].get(f"eff_{iso}", [1.0])[0]
            if eff > 0:
                rate = count / (baseline_live_time * eff)
            else:
                rate = 0.0
            baseline_rates[iso] = rate  # Bq

    dilution_factor = 0.0
    if monitor_vol + sample_vol > 0:
        dilution_factor = monitor_vol / (monitor_vol + sample_vol)

    for iso, rate in baseline_rates.items():
        fit = time_fit_results.get(iso)
        if fit and (f"E_{iso}" in fit):
            fit["E_corrected"] = fit[f"E_{iso}"] - rate * dilution_factor

    if baseline_rates:
        baseline_info["rate_Bq"] = baseline_rates
        baseline_info["dilution_factor"] = dilution_factor

    # ────────────────────────────────────────────────────────────
    # Radon activity extrapolation
    # ────────────────────────────────────────────────────────────
    from radon_activity import compute_radon_activity, compute_total_radon

    radon_results = {}
    eff_Po214 = cfg.get("time_fit", {}).get("eff_Po214", [1.0])[0]
    eff_Po218 = cfg.get("time_fit", {}).get("eff_Po218", [1.0])[0]

    rate214 = None
    err214 = None
    if "Po214" in time_fit_results:
        rate214 = time_fit_results["Po214"].get("E_corrected", time_fit_results["Po214"].get("E_Po214"))
        err214 = time_fit_results["Po214"].get("dE_Po214")

    rate218 = None
    err218 = None
    if "Po218" in time_fit_results:
        rate218 = time_fit_results["Po218"].get("E_corrected", time_fit_results["Po218"].get("E_Po218"))
        err218 = time_fit_results["Po218"].get("dE_Po218")

    A_radon, dA_radon = compute_radon_activity(
        rate218, err218, eff_Po218, rate214, err214, eff_Po214
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
            fit = time_fit_results["Po214"]
            E = fit.get("E_corrected", fit.get("E_Po214"))
            dE = fit.get("dE_Po214", 0.0)
            N0 = fit.get("N0_Po214", 0.0)
            dN0 = fit.get("dN0_Po214", 0.0)
            hl = cfg.get("time_fit", {}).get("hl_Po214", [328320])[0]
            delta214, err_delta214 = radon_delta(
                t_start_rel,
                t_end_rel,
                E,
                dE,
                N0,
                dN0,
                hl,
            )

        delta218 = err_delta218 = None
        if "Po218" in time_fit_results:
            fit = time_fit_results["Po218"]
            E = fit.get("E_corrected", fit.get("E_Po218"))
            dE = fit.get("dE_Po218", 0.0)
            N0 = fit.get("N0_Po218", 0.0)
            dN0 = fit.get("dN0_Po218", 0.0)
            hl = cfg.get("time_fit", {}).get("hl_Po218", [328320])[0]
            delta218, err_delta218 = radon_delta(
                t_start_rel,
                t_end_rel,
                E,
                dE,
                N0,
                dN0,
                hl,
            )

        d_radon, d_err = compute_radon_activity(
            delta218,
            err_delta218,
            eff_Po218,
            delta214,
            err_delta214,
            eff_Po214,
        )
        radon_results["radon_delta_Bq"] = {"value": d_radon, "uncertainty": d_err}

    # ────────────────────────────────────────────────────────────
    # 8. Assemble and write out the summary JSON
    # ────────────────────────────────────────────────────────────
    summary = {
        "timestamp": now_str,
        "config_used": os.path.basename(args.config),
        "calibration": cal_params,
        "spectral_fit": spectrum_results,
        "time_fit": time_fit_results,
        "systematics": systematics_results,
        "baseline": baseline_info,
        "radon_results": radon_results,
        "burst_filter": {"removed_events": int(n_removed_burst)},
        "adc_drift_rate": drift_rate,
        "efficiency": efficiency_results,
        "random_seed": seed_used,
        "git_commit": commit,
        "cli_sha256": cli_sha256,
        "cli_args": cli_args,
        "analysis": {
            "analysis_start_time": t0_cfg,
            "analysis_end_time": t_end_cfg,
            "spike_end_time": spike_end_cfg,
            "spike_periods": spike_periods_cfg,
            "run_periods": run_periods_cfg,
            "radon_interval": radon_interval_cfg,
            "ambient_concentration": cfg.get("analysis", {}).get("ambient_concentration"),
        },
    }

    out_dir = write_summary(args.output_dir, summary, args.job_id or now_str)
    copy_config(out_dir, args.config)

    # Generate plots now that the output directory exists
    if spec_plot_data:
        try:
            _ = plot_spectrum(
                energies=spec_plot_data["energies"],
                fit_vals=spec_plot_data["fit_vals"],
                out_png=os.path.join(out_dir, "spectrum.png"),
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
            other = "Po214" if iso == "Po218" else "Po218"
            if not overlay:
                plot_cfg[f"window_{other}"] = None
                ts_times = pdata["events_times"]
                ts_energy = pdata["events_energy"]
                fit_dict = time_fit_results.get(iso, {})
            else:
                ts_times = events["timestamp"].values
                ts_energy = events["energy_MeV"].values
                fit_dict = {}
                fit_dict.update(time_fit_results.get("Po214", {}))
                fit_dict.update(time_fit_results.get("Po218", {}))
            _ = plot_time_series(
                all_timestamps=ts_times,
                all_energies=ts_energy,
                fit_results=fit_dict,
                t_start=t0_global,
                t_end=t_end_global,
                config=plot_cfg,
                out_png=os.path.join(out_dir, f"time_series_{iso}.png"),
            )
        except Exception as e:
            print(f"WARNING: Could not create time-series plot for {iso} -> {e}")

    # Additional visualizations
    if efficiency_results.get("sources"):
        try:
            errs_arr = np.array([s.get("error", 0.0) for s in efficiency_results["sources"].values()])
            if errs_arr.size > 0:
                cov = np.diag(errs_arr ** 2)
                cov_heatmap(
                    cov,
                    os.path.join(out_dir, "eff_cov.png"),
                    labels=list(efficiency_results["sources"].keys()),
                )
            efficiency_bar(
                efficiency_results,
                os.path.join(out_dir, "efficiency.png"),
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
            fit = time_fit_results["Po214"]
            E = fit.get("E_corrected", fit.get("E_Po214"))
            dE = fit.get("dE_Po214", 0.0)
            N0 = fit.get("N0_Po214", 0.0)
            dN0 = fit.get("dN0_Po214", 0.0)
            hl = cfg.get("time_fit", {}).get("hl_Po214", [328320])[0]
            A214, dA214 = radon_activity_curve(t_rel, E, dE, N0, dN0, hl)

        A218 = dA218 = None
        if "Po218" in time_fit_results:
            fit = time_fit_results["Po218"]
            E = fit.get("E_corrected", fit.get("E_Po218"))
            dE = fit.get("dE_Po218", 0.0)
            N0 = fit.get("N0_Po218", 0.0)
            dN0 = fit.get("dN0_Po218", 0.0)
            hl = cfg.get("time_fit", {}).get("hl_Po218", [328320])[0]
            A218, dA218 = radon_activity_curve(t_rel, E, dE, N0, dN0, hl)

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
            os.path.join(out_dir, "radon_activity.png"),
            config=cfg.get("plotting", {}),
        )

        if radon_interval is not None:
            times_trend = np.linspace(radon_interval[0], radon_interval[1], 50)
            rel_trend = times_trend - t0_global
            A214_tr = None
            if "Po214" in time_fit_results:
                fit = time_fit_results["Po214"]
                E214 = fit.get("E_corrected", fit.get("E_Po214"))
                dE214 = fit.get("dE_Po214", 0.0)
                N0214 = fit.get("N0_Po214", 0.0)
                dN0214 = fit.get("dN0_Po214", 0.0)
                hl214 = cfg.get("time_fit", {}).get("hl_Po214", [328320])[0]
                A214_tr, _ = radon_activity_curve(rel_trend, E214, dE214, N0214, dN0214, hl214)
            A218_tr = None
            if "Po218" in time_fit_results:
                fit = time_fit_results["Po218"]
                E218 = fit.get("E_corrected", fit.get("E_Po218"))
                dE218 = fit.get("dE_Po218", 0.0)
                N0218 = fit.get("N0_Po218", 0.0)
                dN0218 = fit.get("dN0_Po218", 0.0)
                hl218 = cfg.get("time_fit", {}).get("hl_Po218", [328320])[0]
                A218_tr, _ = radon_activity_curve(rel_trend, E218, dE218, N0218, dN0218, hl218)
            trend = np.zeros_like(times_trend)
            for i in range(times_trend.size):
                r214 = A214_tr[i] if A214_tr is not None else None
                r218 = A218_tr[i] if A218_tr is not None else None
                A, _ = compute_radon_activity(r218, None, 1.0, r214, None, 1.0)
                trend[i] = A
            plot_radon_trend(
                times_trend,
                trend,
                os.path.join(out_dir, "radon_trend.png"),
                config=cfg.get("plotting", {}),
            )

        ambient = cfg.get("analysis", {}).get("ambient_concentration")
        ambient_interp = None
        if args.ambient_file:
            try:
                dat = np.loadtxt(args.ambient_file, usecols=(0, 1))
                ambient_interp = np.interp(times, dat[:, 0], dat[:, 1])
            except Exception as e:
                print(f"WARNING: Could not read ambient file '{args.ambient_file}': {e}")

        if ambient_interp is not None:
            vol_arr = activity_arr / ambient_interp
            vol_err = err_arr / ambient_interp
            plot_equivalent_air(
                times,
                vol_arr,
                vol_err,
                None,
                os.path.join(out_dir, "equivalent_air.png"),
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
                os.path.join(out_dir, "equivalent_air.png"),
                config=cfg.get("plotting", {}),
            )
    except Exception as e:
        print(f"WARNING: Could not create radon activity plots -> {e}")

    print(f"Analysis complete. Results written to -> {out_dir}")


if __name__ == "__main__":
    main()
