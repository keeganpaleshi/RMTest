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
  7. (Optional) Systematics scan around user-specified sigma shifts.
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


import sys
import logging
import random

logger = logging.getLogger(__name__)
from datetime import datetime, timezone, timedelta
import subprocess
import hashlib
import json
from pathlib import Path
import shutil
from typing import Any, Mapping

import math
import numpy as np
import pandas as pd
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

from fitting import fit_spectrum, fit_time_series, FitResult
from reporting import build_diagnostics, start_warning_capture

from constants import (
    DEFAULT_NOISE_CUTOFF,
    NEGATIVE_ACTIVITY_CLAMP_UNCERTAINTY_BQ,
    DEFAULT_ADC_CENTROIDS,
    DEFAULT_KNOWN_ENERGIES,
)

from analysis_helpers import (
    PipelineTimer,
    _burst_sensitivity_scan,
    _config_efficiency,
    _cov_lookup,
    _eff_prior,
    _ensure_events,
    _fallback_uncertainty,
    _fit_efficiency,
    _fit_params,
    _float_with_default,
    _hl_value,
    _model_uncertainty,
    _normalise_mu_bounds,
    _radon_activity_curve_from_fit,
    _radon_background_mode,
    _radon_time_window,
    _regrid_series,
    _resolved_efficiency,
    _roi_diff,
    _save_stub_spectrum_plot,
    _spectral_fit_with_check,
    _total_radon_series,
    _ts_bin_centers_widths,
    _segments_to_isotope_series,
    _safe_float,
    auto_expand_window,
    dedupe_isotope_series,
    get_spike_efficiency,
    prepare_analysis_df,
    window_prob,
)


from plot_utils import (
    plot_spectrum,
    plot_time_series,
    plot_equivalent_air,
    plot_spectrum_comparison,
    plot_activity_grid,
)

from plotting_wrappers import (
    plot_radon_activity_dict,
    plot_radon_trend_dict,
    plot_radon_activity,
    plot_total_radon,
    plot_radon_trend,
)

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
import baseline_handling
from time_fitting import two_pass_time_fit
from config.validation import validate_baseline_window
from pipeline_init import init_pipeline
from data_loading import load_and_filter_events, setup_time_windows
from calibration_stage import run_energy_calibration
from baseline_stage import run_baseline_stage
from spectral_fit_stage import run_spectral_fit_stage
from time_series_stage import run_time_series_stage
from systematics_stage import run_systematics_stage
from summary_stage import build_and_write_summary

# Re-export for test compatibility
from analysis_helpers import _spike_eff_cache


def main(argv=None):
    # Initialize pipeline: parse args, load config, apply overrides, setup logging & random seed
    args, cfg, timer, tzinfo, metadata = init_pipeline(argv)

    # Store cli_args for test compatibility
    cli_args = args

    # Unpack metadata
    cli_sha256 = metadata["cli_sha256"]
    commit = metadata["commit"]
    requirements_sha256 = metadata["requirements_sha256"]
    cfg_sha256 = metadata["cfg_sha256"]
    seed_used = metadata["seed_used"]
    now_str = metadata["timestamp"]

    # Initialize variables for pipeline
    pre_spec_energies = np.array([])
    post_spec_energies = np.array([])
    roi_diff = {}
    scan_results = {}
    best_params = None

    # ────────────────────────────────────────────────────────────
    # 2. Load event data and apply filters
    # ────────────────────────────────────────────────────────────
    try:
        events_all, events_filtered, n_removed_noise, n_removed_burst = load_and_filter_events(
            args.input, cfg, args, timer
        )
    except Exception:
        sys.exit(1)

    if events_all.empty:
        sys.exit(0)

    events_after_noise = events_filtered.copy()
    events_after_burst = events_filtered.copy()

    # Determine burst mode for summary
    burst_mode = (
        args.burst_mode
        if args.burst_mode is not None
        else cfg.get("burst_filter", {}).get("burst_mode", "rate")
    )

    # Setup time windows and parse time periods
    time_windows = setup_time_windows(cfg, events_filtered)
    t0_global = time_windows["t0_global"]
    t0_cfg = time_windows["t0_cfg"]
    t_end_global = time_windows["t_end_global"]
    t_end_global_ts = time_windows["t_end_global_ts"]
    t_end_cfg = time_windows["t_end_cfg"]
    t_spike_start = time_windows["t_spike_start"]
    spike_start_cfg = time_windows["spike_start_cfg"]
    t_spike_end = time_windows["t_spike_end"]
    spike_end_cfg = time_windows["spike_end_cfg"]
    spike_periods = time_windows["spike_periods"]
    spike_periods_cfg = time_windows["spike_periods_cfg"]
    run_periods = time_windows["run_periods"]
    run_periods_cfg = time_windows["run_periods_cfg"]
    radon_interval = time_windows["radon_interval"]
    radon_interval_cfg = time_windows["radon_interval_cfg"]

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
            logging.exception("calibration failed – using defaults")
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
            sigma_E, sigma_E_error = _value_sigma(obj.get("sigma_E", 0.0))
    
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
                sigma_E=sigma_E,
                sigma_E_error=sigma_E_error,
                peaks=obj.get("peaks"),
            )
    
        cal_result = _as_cal_result(cal_params)
    
        # Save “a, c, sigma_E” so we can reconstruct energies
        if isinstance(cal_params, dict):
            a, a_sig = _value_sigma(cal_params.get("a", 0.0))
            a2, a2_sig = _value_sigma(cal_params.get("a2", 0.0))
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
    
        # ────────────────────────────────────────────────────────────
    with timer.section("baseline"):
        # 4. Baseline run (optional)
        # ────────────────────────────────────────────────────────────
        baseline_result = run_baseline_stage(
            df_analysis=df_analysis,
            events_all=events_all,
            cal_result=cal_result,
            cfg=cfg,
            args=args,
            analysis_start=analysis_start,
            analysis_end=analysis_end,
            hist_bins=hist_bins,
        )

        # Unpack results
        baseline_info = baseline_result["baseline_info"]
        baseline_counts = baseline_result["baseline_counts"]
        baseline_record = baseline_result["baseline_record"]
        dilution_factor = baseline_result["dilution_factor"]
        base_events = baseline_result["base_events"]
        baseline_range = baseline_result["baseline_range"]
        isotopes_to_subtract = baseline_result["isotopes_to_subtract"]
        df_analysis = baseline_result["df_analysis"]

        # Extract baseline_live_time for later use
        baseline_live_time = baseline_info.get("live_time", 0.0)

        # Initialize other baseline-related variables
        baseline_background_provenance: dict[str, dict[str, Any]] = {}

        # ────────────────────────────────────────────────────────────
    with timer.section("spectral_fit"):
        # 5. Spectral fit (optional)
        # ────────────────────────────────────────────────────────────
        spectrum_results = {}
        spec_plot_data = None
        peak_deviation = {}
        if cfg.get("spectral_fit", {}).get("do_spectral_fit", False):
            # Decide binning: new 'binning' dict or legacy keys
            spectral_cfg = cfg["spectral_fit"]
    
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
            elif method == "energy":
                width = 0.02
                if bin_cfg is not None:
                    width = bin_cfg.get("energy_bin_width", width)
                else:
                    width = cfg["spectral_fit"].get("energy_bin_width", width)
                width = float(width)
                if width <= 0:
                    raise ValueError("energy_bin_width must be positive")
                E_all = df_analysis["energy_MeV"].values
                if E_all.size == 0:
                    bins = 1
                    bin_edges = np.array([0.0, width], dtype=float)
                else:
                    e_min = float(np.min(E_all))
                    e_max = float(np.max(E_all))
                    # Guard against a single-point spectrum
                    if np.isclose(e_min, e_max):
                        e_max = e_min + width
                    n_steps = int(np.ceil((e_max - e_min) / width))
                    # np.arange is exclusive of the stop value -> pad by one step
                    stop = e_min + (n_steps + 1) * width
                    bin_edges = np.arange(e_min, stop + 0.5 * width, width, dtype=float)
                    bins = bin_edges.size - 1
            else:
                # "ADC" binning mode -> fixed width in raw channels
                width = 1
                if bin_cfg is not None:
                    width = bin_cfg.get("adc_bin_width", 1)
                else:
                    width = spectral_cfg.get("adc_bin_width", 1)
                adc_min = df_analysis["adc"].min()
                adc_max = df_analysis["adc"].max()
                bins = int(np.ceil((adc_max - adc_min + 1) / width))
    
                # Build edges in ADC units then convert to energy for plotting
                bin_edges_adc = np.arange(adc_min, adc_min + bins * width + 1, width)
                bin_edges = apply_calibration(bin_edges_adc, a, c, quadratic_coeff=a2)
    
            # Find approximate ADC centroids for Po‐210, Po‐218, Po‐214
    
            expected_peaks = spectral_cfg.get("expected_peaks")
            if expected_peaks is None:
                expected_peaks = DEFAULT_ADC_CENTROIDS
    
            # `find_adc_bin_peaks` will return a dict: e.g. { "Po210": adc_centroid, … }
            adc_peaks = find_adc_bin_peaks(
                df_analysis["adc"].values,
                expected=expected_peaks,
                window=spectral_cfg.get("peak_search_width_adc", 50),
                prominence=spectral_cfg.get("peak_search_prominence", 0),
                width=spectral_cfg.get("peak_search_width_adc", None),
                method=spectral_cfg.get("peak_search_method", "prominence"),
                cwt_widths=spectral_cfg.get("peak_search_cwt_widths"),
            )
    
            # Build priors for the unbinned spectrum fit:
            priors_spec = {}
            # Resolution prior: map calibrated sigma_E -> sigma0 parameter
            sigma_prior_source = spectral_cfg.get("sigma_E_prior_source")
            sigma_prior_sigma = spectral_cfg.get("sigma_E_prior_sigma", sigE_sigma)
    
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
    
            float_sigma_E = bool(spectral_cfg.get("float_sigma_E", True))
    
            priors_spec["sigma_E"] = (sigE_mean, sigma_E_prior)
            # Fit_spectrum expects separate ``sigma0`` and ``F`` resolution terms.
            # Initialise sigma0 from the calibration-derived resolution.  Allow it
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
            )
    
            for peak, centroid_adc in adc_peaks.items():
                mu = apply_calibration(centroid_adc, a, c, quadratic_coeff=a2)
                bounds = mu_bounds_fit.get(peak)
                if bounds is not None:
                    lo, hi = bounds
                    if not (lo <= mu <= hi):
                        mu = float(np.clip(mu, lo, hi))
                priors_spec[f"mu_{peak}"] = (mu, spectral_cfg.get("mu_sigma"))
                # Observed raw-counts around the expected energy window
                peak_tol = spectral_cfg.get("spectral_peak_tolerance_mev", 0.3)
                raw_count = float(
                    (
                        (df_analysis["energy_MeV"] >= mu - peak_tol)
                        & (df_analysis["energy_MeV"] <= mu + peak_tol)
                    ).sum()
                )
                mu_amp = max(raw_count, 1.0)
                sigma_amp = max(
                    np.sqrt(mu_amp), spectral_cfg.get("amp_prior_scale") * mu_amp
                )
                priors_spec[f"S_{peak}"] = (mu_amp, sigma_amp)
    
                # If EMG tails are requested for this peak:
                if spectral_cfg.get("use_emg", {}).get(peak, False):
                    priors_spec[f"tau_{peak}"] = (
                        spectral_cfg.get(f"tau_{peak}_prior_mean"),
                        spectral_cfg.get(f"tau_{peak}_prior_sigma"),
                    )
    
            # Continuum priors
            bkg_mode = str(spectral_cfg.get("bkg_mode", "manual")).lower()
            if bkg_mode == "auto":
                from background import estimate_linear_background
    
                mu_map = {k: priors_spec[f"mu_{k}"][0] for k in adc_peaks.keys()}
                peak_tol = spectral_cfg.get("spectral_peak_tolerance_mev", 0.3)
                b0_est, b1_est = estimate_linear_background(
                    df_analysis["energy_MeV"].values,
                    mu_map,
                    peak_width=peak_tol,
                )
                priors_spec["b0"] = (b0_est, abs(b0_est) * 0.1 + 1e-3)
                priors_spec["b1"] = (b1_est, abs(b1_est) * 0.1 + 1e-3)
            elif bkg_mode.startswith("auto_poly"):
                from background import estimate_polynomial_background_auto
    
                mu_map = {k: priors_spec[f"mu_{k}"][0] for k in adc_peaks.keys()}
                peak_tol = spectral_cfg.get("spectral_peak_tolerance_mev", 0.3)
                try:
                    max_n = int(bkg_mode.split("auto_poly")[-1])
                except ValueError:
                    max_n = 2
                coeffs, order = estimate_polynomial_background_auto(
                    df_analysis["energy_MeV"].values,
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
    
            # Flags controlling the spectral fit
            spec_flags = spectral_cfg.get("flags", {}).copy()
            analysis_cfg = cfg.get("analysis", {})
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
    
            # Launch the spectral fit
            spec_fit_out = None
            peak_deviation = {}
            try:
                fit_kwargs = {
                    "energies": df_analysis["energy_MeV"].values,
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
    
                spec_fit_out, peak_deviation = _spectral_fit_with_check(
                    df_analysis["energy_MeV"].values,
                    priors_spec,
                    spec_flags,
                    cfg,
                    bins=fit_kwargs.get("bins"),
                    bin_edges=fit_kwargs.get("bin_edges"),
                    bounds=fit_kwargs.get("bounds"),
                    unbinned=fit_kwargs.get("unbinned", False),
                    strict=fit_kwargs.get("strict", False),
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
                            df_analysis["energy_MeV"].values,
                            priors_shrunk,
                            flags=flags_fix,
                            bins=fit_kwargs.get("bins"),
                            bin_edges=fit_kwargs.get("bin_edges"),
                            bounds=fit_kwargs.get("bounds"),
                            unbinned=fit_kwargs.get("unbinned", False),
                            strict=fit_kwargs.get("strict", False),
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
                                    df_analysis["energy_MeV"].values,
                                    priors_shrunk,
                                    flags=spec_flags,
                                    bins=fit_kwargs.get("bins"),
                                    bin_edges=fit_kwargs.get("bin_edges"),
                                    bounds=fit_kwargs.get("bounds"),
                                    unbinned=fit_kwargs.get("unbinned", False),
                                    strict=fit_kwargs.get("strict", False),
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
            except Exception as e:
                logger.warning("Spectral fit failed -> %s", e)
                spectrum_results = {}
    
            # Store plotting inputs (bin_edges now in energy units)
            fit_vals = None
            if isinstance(spec_fit_out, FitResult):
                fit_vals = spec_fit_out
            elif isinstance(spec_fit_out, dict):
                fit_vals = spec_fit_out
            spec_plot_data = {
                "energies": df_analysis["energy_MeV"].values,
                "fit_vals": fit_vals,
                "bins": bins,
                "bin_edges": bin_edges,
                "flags": dict(spec_flags),
            }
    
        # ────────────────────────────────────────────────────────────
    with timer.section("time_series"):
        # 6. Time‐series decay fits for Po‐218 and Po‐214
        # ────────────────────────────────────────────────────────────
        time_series_results = run_time_series_stage(
            df_analysis=df_analysis,
            cfg=cfg,
            args=args,
            t0_global=t0_global,
            t_end_global=t_end_global,
            t_end_global_ts=t_end_global_ts,
            baseline_range=baseline_range,
            base_events=base_events,
            baseline_record=baseline_record,
            baseline_live_time=baseline_live_time,
            baseline_counts=baseline_counts,
            isotopes_to_subtract=isotopes_to_subtract,
            baseline_info=baseline_info,
            baseline_background_provenance=baseline_background_provenance,
        )

        # Unpack results
        time_fit_results = time_series_results["time_fit_results"]
        time_fit_background_meta = time_series_results["time_fit_background_meta"]
        priors_time_all = time_series_results["priors_time_all"]
        time_plot_data = time_series_results["time_plot_data"]
        iso_live_time = time_series_results["iso_live_time"]
        t_start_map = time_series_results["t_start_map"]
        iso_counts = time_series_results["iso_counts"]
        iso_counts_raw = time_series_results["iso_counts_raw"]
        radon_estimate_info = time_series_results["radon_estimate_info"]
        po214_estimate_info = time_series_results["po214_estimate_info"]
        po218_estimate_info = time_series_results["po218_estimate_info"]

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
                        logger.warning("%s – clamping to non-negative values", msg)
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
        summary, out_dir, isotope_series_data, radon_inference_results = build_and_write_summary(
            args=args,
            cfg=cfg,
            spectrum_results=spectrum_results,
            peak_deviation=peak_deviation,
            time_fit_results=time_fit_results,
            time_fit_background_meta=time_fit_background_meta,
            baseline_background_provenance=baseline_background_provenance,
            baseline_record=baseline_record,
            cal_params=cal_params,
            cal_result=cal_result,
            now_str=now_str,
            cfg_sha256=cfg_sha256,
            calibration_valid=calibration_valid,
            systematics_results=systematics_results,
            baseline_info=baseline_info,
            radon_results=radon_results,
            n_removed_noise=n_removed_noise,
            n_removed_burst=n_removed_burst,
            burst_mode=burst_mode,
            roi_diff=roi_diff,
            scan_results=scan_results,
            best_params=best_params,
            drift_rate=drift_rate,
            drift_mode=drift_mode,
            drift_params=drift_params,
            efficiency_results=efficiency_results,
            seed_used=seed_used,
            commit=commit,
            requirements_sha256=requirements_sha256,
            cli_sha256=cli_sha256,
            cli_args=cli_args,
            t0_cfg=t0_cfg,
            t_end_cfg=t_end_cfg,
            spike_start_cfg=spike_start_cfg,
            spike_end_cfg=spike_end_cfg,
            spike_periods_cfg=spike_periods_cfg,
            run_periods_cfg=run_periods_cfg,
            radon_interval_cfg=radon_interval_cfg,
            radon_combined_info=radon_combined_info,
            corrected_rates=corrected_rates,
            corrected_unc=corrected_unc,
            fit214=fit214,
            fit218=fit218,
            iso_live_time=iso_live_time,
            iso_mode=iso_mode,
            monitor_vol=monitor_vol,
            sample_vol=sample_vol,
            weights=weights,
            spec_plot_data=spec_plot_data,
            df_analysis=df_analysis,
            pre_spec_energies=pre_spec_energies,
            post_spec_energies=post_spec_energies,
            t0_global=t0_global,
            t_end_global=t_end_global,
            t_end_global_ts=t_end_global_ts,
            analysis_start=analysis_start,
            radon_interval=radon_interval,
            time_plot_data=time_plot_data,
        )

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
        background_mode = _radon_background_mode(cfg, time_fit_results)
        background_mode = baseline_handling.normalize_background_mode(background_mode)

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
