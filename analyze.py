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
     → Append `energy_MeV` to every event.
  4. (Optional) Extract a “baseline” interval for background estimation.
  5. (Optional) Spectral fit (Po‐210, Po‐218, Po‐214) using unbinned likelihood.
     → Can bin either in “1 ADC‐channel per bin” or Freedman‐Diaconis (per config).
     → Uses EMG tails for Po‐210/Po‐218 if requested.
     → Overlays fit on the spectrum plot.
  6. Time‐series decay fit (Po‐218 and Po‐214 separately).
     → Extract events in each isotope’s energy window.
     → Subtract global t₀ so that model always starts at t=0.
     → Fit unbinned decay (with efficiency, background, N₀, half‐life priors).
     → Overlay fit curve on a time‐binned histogram (default 1 h bins), at 95% CL.
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
import json
from datetime import datetime

import numpy as np
import pandas as pd

# ‣ Import our supporting modules (all must live in the same folder).
from io_utils     import load_config, copy_config, load_events, write_summary
from calibration  import derive_calibration_constants, derive_calibration_constants_auto
from fitting      import fit_spectrum, fit_decay
from plot_utils   import plot_spectrum, plot_time_series
from systematics  import scan_systematics
from utils        import find_adc_peaks


def parse_args():
    p = argparse.ArgumentParser(
        description="Full Radon Monitor Analysis Pipeline"
    )
    p.add_argument(
        "--config", "-c", required=True,
        help="Path to JSON configuration file"
    )
    p.add_argument(
        "--input", "-i", required=True,
        help="CSV of merged event data (must contain at least: timestamp, adc)"
    )
    p.add_argument(
        "--output_dir", "-o", required=True,
        help="Directory under which to create a timestamped analysis folder"
    )
    p.add_argument(
        "--baseline_range", nargs=2, metavar=("TSTART", "TEND"),
        help=(
            "Optional baseline-run interval. "
            "Provide two values (either ISO strings or epoch floats). "
            "If set, those events are extracted (same energy cuts) and "
            "listed in `baseline` of the summary."
        )
    )
    return p.parse_args()


def main():
    args = parse_args()

    # ────────────────────────────────────────────────────────────
    # 1. Load configuration
    # ────────────────────────────────────────────────────────────
    try:
        cfg = load_config(args.config)
    except Exception as e:
        print(f"ERROR: Could not load config '{args.config}': {e}")
        sys.exit(1)

    # Timestamp for this analysis run
    now_str = datetime.utcnow().strftime("%Y%m%dT%H%M%SZ")



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
            pd.to_datetime(events["timestamp"])
              .astype(np.int64) / 1e9
        )

    events["timestamp"] = events["timestamp"].astype(float)

    # Global t₀ reference = earliest event
    t0_global = events["timestamp"].min()

    # ────────────────────────────────────────────────────────────
    # 3. Energy calibration
    # ────────────────────────────────────────────────────────────
    adc_vals = events["adc"].values

    if cfg.get("calibration", {}).get("method", "two-point") == "auto":
        # Auto‐cal using Freedman‐Diaconis histogram + peak detection
        cal_params = derive_calibration_constants_auto(
            adc_vals,
            noise_cutoff = cfg["calibration"].get("noise_cutoff", 300),
            hist_bins    = cfg["calibration"].get("hist_bins", 2000),
            peak_search_radius = cfg["calibration"].get("peak_search_radius", 200)
        )
    else:
        # Two‐point calibration as given in config
        cal_params = derive_calibration_constants(
            adc_vals,
            config = cfg
        )

    # Save “a, c, sigma_E” so we can reconstruct energies
    a,  a_sig  = cal_params["a"]
    c,  c_sig  = cal_params["c"]
    sigE_mean, sigE_sigma = cal_params["sigma_E"]

    # Apply linear calibration → new column “energy_MeV”
    events["energy_MeV"] = events["adc"] * a + c

    # ────────────────────────────────────────────────────────────
    # 4. Baseline run (optional)
    # ────────────────────────────────────────────────────────────
    baseline_info = {}
    if args.baseline_range:
        def to_epoch(x):
            try:
                return float(x)
            except:
                return pd.to_datetime(x).astype(np.int64) / 1e9

        t_start_base = to_epoch(args.baseline_range[0])
        t_end_base   = to_epoch(args.baseline_range[1])
        mask_base = (
            (events["timestamp"] >= t_start_base) &
            (events["timestamp"] <  t_end_base  )
        )
        base_events = events[mask_base].copy()
        baseline_info = {
            "start": t_start_base,
            "end":   t_end_base,
            "n_events": len(base_events)
        }
    else:
        base_events = pd.DataFrame()

    # ────────────────────────────────────────────────────────────
    # 5. Spectral fit (optional)
    # ────────────────────────────────────────────────────────────
    spectrum_results = {}
    spec_plot_data = None
    if cfg.get("spectral_fit", {}).get("do_spectral_fit", False):
        # Decide binning: “adc” or “fd”
        bin_cfg = cfg["spectral_fit"].get("binning", {})
        if bin_cfg.get("method", "adc").lower() == "fd":
            E_all = events["energy_MeV"].values
            # Freedman‐Diaconis on energy array
            q25, q75 = np.percentile(E_all, [25, 75])
            iqr = q75 - q25
            n = E_all.size
            if (iqr > 0) and (n > 0):
                fd_width = 2 * iqr / (n ** (1/3))
                emin, emax = E_all.min(), E_all.max()
                nbins = max(1, int(np.ceil((emax - emin) / fd_width)))
            else:
                nbins = bin_cfg.get("default_bins", 400)

            bins = nbins
            bin_edges = None
        else:
            # “1 ADC channel per bin”
            adc_min = events["adc"].min()
            adc_max = events["adc"].max()
            bins = int(adc_max - adc_min + 1)
            bin_edges = np.arange(adc_min, adc_max + 2)

        # Find approximate ADC centroids for Po‐210, Po‐218, Po‐214
        expected_peaks = cfg["spectral_fit"].get(
            "expected_peaks",
            {"Po210": 5300, "Po218": 6000, "Po214": 7690}
        )
        # `find_adc_peaks` will return a dict: e.g. { "Po210": adc_centroid, … }
        adc_peaks = find_adc_peaks(
            events["adc"].values,
            expected=expected_peaks,
            window=cfg["spectral_fit"].get("peak_window_adc", 50)
        )

        # Build priors for the unbinned spectrum fit:
        priors_spec = {}
        # σ_E prior
        priors_spec["sigma_E"] = (
            sigE_mean,
            sigE_sigma
        )

        for peak, centroid_adc in adc_peaks.items():
            mu = centroid_adc * a + c  # convert to MeV
            priors_spec[f"mu_{peak}"] = (
                mu,
                cfg["spectral_fit"].get("mu_sigma", 0.05)
            )
            # Observed raw‐counts in ±0.3 MeV window as seed
            raw_count = float(
                ((events["energy_MeV"] >= mu - 0.3) &
                 (events["energy_MeV"] <= mu + 0.3)).sum()
            )
            mu_amp = max(raw_count, 1.0)
            sigma_amp = max(
                np.sqrt(mu_amp),
                cfg["spectral_fit"].get("amp_prior_scale", 1.0) * mu_amp
            )
            priors_spec[f"S_{peak}"] = (mu_amp, sigma_amp)

            # If EMG tails are requested for this peak:
            if cfg["spectral_fit"].get("use_emg", {}).get(peak, False):
                priors_spec[f"tau_{peak}"] = (
                    cfg["spectral_fit"].get(f"tau_{peak}_prior_mean", 0.0),
                    cfg["spectral_fit"].get(f"tau_{peak}_prior_sigma", 0.0)
                )

        # Continuum priors
        priors_spec["b0"] = tuple(cfg["spectral_fit"].get("b0_prior", (0.0, 1.0)))
        priors_spec["b1"] = tuple(cfg["spectral_fit"].get("b1_prior", (0.0, 1.0)))

        # Launch the spectral fit
        try:
            spec_fit_out = fit_spectrum(
                energies=events["energy_MeV"].values,
                priors=priors_spec,
                flags=cfg["spectral_fit"].get("flags", {})
            )
            spectrum_results = spec_fit_out
        except Exception as e:
            print(f"WARNING: Spectral fit failed → {e}")
            spectrum_results = {}

        # Store plotting inputs to generate the figure after summary writing
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
            if win_key not in cfg["time_fit"]:
                continue

        lo, hi = cfg["time_fit"][win_key]
        iso_mask = (
            (events["energy_MeV"] >= lo) &
            (events["energy_MeV"] <= hi)
        )
        iso_events = events[iso_mask].copy()
        if iso_events.empty:
            print(f"WARNING: No events found for {iso} in [{lo}, {hi}] MeV.")
            continue

        # Relative times for fitting: subtract t0_global
        times_rel = (iso_events["timestamp"].values - t0_global).astype(float)

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
        if args.baseline_range:
            # Count baseline events in this energy window
            n0_count = float(
                ((base_events["energy_MeV"] >= lo) &
                 (base_events["energy_MeV"] <= hi)).sum()
            )
            priors_time["N0"] = (
                n0_count,
                cfg["time_fit"].get(f"sig_N0_{iso}", np.sqrt(n0_count) if n0_count > 0 else 1.0)
            )
        else:
            priors_time["N0"] = (0.0, 0.0)

        # Store priors for use in systematics scanning
        priors_time_all[iso] = priors_time

        # Any extra flags (e.g. fix N0=0 or fix B0=0)
        flags_time = cfg["time_fit"].get("flags", {})

        # Run decay fit
        decay_out = None  # fresh variable each iteration
        try:
            decay_out = fit_decay(
                times=times_rel,
                priors=priors_time,
                t0=0.0,
                t_end=(events["timestamp"].max() - t0_global),
                flags=flags_time
            )
            time_fit_results[iso] = decay_out
        except Exception as e:
            print(f"WARNING: Decay‐curve fit for {iso} failed → {e}")
            time_fit_results[iso] = {}

        # Store inputs for plotting later
        time_plot_data[iso] = {
            "events_times": iso_events["timestamp"].values,
            "fit_dict": decay_out,
        }

    # ────────────────────────────────────────────────────────────
    # 7. Systematics scan (optional)
    # ────────────────────────────────────────────────────────────
    systematics_results = {}
    if cfg.get("systematics", {}).get("enable", False):
        sigma_dict = cfg["systematics"].get("sigma_shifts", {})
        keys       = cfg["systematics"].get("scan_keys", [])

        for iso, fit_out in time_fit_results.items():
            if not fit_out:
                continue

            # Build a wrapper to re‐run fit_decay with modified priors
            def fit_wrapper(priors_mod):
                filtered_times = (
                    events[
                        (events["energy_MeV"] >= cfg["time_fit"][f"window_{iso}"][0]) &
                        (events["energy_MeV"] <= cfg["time_fit"][f"window_{iso}"][1])
                    ]["timestamp"].values - t0_global
                )
                out = fit_decay(
                    times=filtered_times,
                    priors=priors_mod,
                    t0=0.0,
                    t_end=(events["timestamp"].max() - t0_global)
                )
                return out.get("eff", np.nan)

            try:
                deltas, total_unc = scan_systematics(
                    fit_wrapper,
                    priors_time_all.get(iso, {}),
                    sigma_dict,
                    keys
                )
                systematics_results[iso] = {"deltas": deltas, "total_unc": total_unc}
            except Exception as e:
                print(f"WARNING: Systematics scan for {iso} → {e}")

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
        "baseline": baseline_info
    }

    out_dir = write_summary(args.output_dir, summary)
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
            )
        except Exception as e:
            print(f"WARNING: Could not create spectrum plot: {e}")

    for iso, pdata in time_plot_data.items():
        try:
            _ = plot_time_series(
                all_timestamps=pdata["events_times"],
                all_energies=events["energy_MeV"].values,
                fit_results=pdata["fit_dict"],
                t_start=t0_global,
                t_end=events["timestamp"].max(),
                config=cfg["time_fit"],
                out_png=os.path.join(out_dir, f"time_series_{iso}.png"),
            )
        except Exception as e:
            print(f"WARNING: Could not create time-series plot for {iso} → {e}")

    print(f"Analysis complete. Results written to → {out_dir}")


if __name__ == "__main__":
    main()
