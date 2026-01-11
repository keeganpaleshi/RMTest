"""
baseline_stage.py

Baseline analysis stage for the radon monitor pipeline.
Handles baseline range extraction, dilution factor computation, and baseline subtraction.
"""

import sys
import logging
from typing import Any, Optional, Tuple
from datetime import datetime

import numpy as np
import pandas as pd

from baseline_utils import (
    compute_dilution_factor,
    summarize_baseline,
    BaselineError,
)
from utils import adc_hist_edges, to_utc_datetime
from utils.time_utils import to_datetime_utc
from config.validation import validate_baseline_window
from constants import DEFAULT_NOISE_CUTOFF
import baseline
import baseline_handling

logger = logging.getLogger(__name__)


def run_baseline_stage(
    df_analysis: pd.DataFrame,
    events_all: pd.DataFrame,
    cal_result,
    cfg: dict,
    args,
    analysis_start: datetime,
    analysis_end: datetime,
    hist_bins: int,
) -> dict[str, Any]:
    """
    Run the baseline analysis stage.

    Parameters
    ----------
    df_analysis : pd.DataFrame
        Analysis dataframe with calibrated energies
    events_all : pd.DataFrame
        All events (unfiltered)
    cal_result : CalibrationResult
        Calibration result object
    cfg : dict
        Configuration dictionary
    args : argparse.Namespace
        Command-line arguments
    analysis_start : datetime
        Start time of analysis period
    analysis_end : datetime
        End time of analysis period
    hist_bins : int
        Number of histogram bins for spectrum

    Returns
    -------
    dict
        Dictionary containing:
        - baseline_info: dict with baseline metadata
        - baseline_counts: dict with isotope counts
        - baseline_record: baseline record object
        - dilution_factor: dilution factor value
        - base_events: baseline events dataframe
        - baseline_range: tuple of (start, end) times
        - isotopes_to_subtract: list of isotope names
        - df_analysis: updated analysis dataframe (if baseline subtraction was applied)
    """
    baseline_info = {}
    baseline_counts = {}
    baseline_record = None
    dilution_factor = None
    baseline_cfg = cfg.get("baseline", {})
    isotopes_to_subtract = baseline_cfg.get("isotopes_to_subtract", ["Po214", "Po218"])
    baseline_range = None

    # Parse baseline range from args or config
    if args.baseline_range:
        baseline_range = (args.baseline_range[0], args.baseline_range[1])
        logging.info(
            "Baseline window: %s → %s",
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
    except ValueError:
        raise

    # Parse baseline volumes
    try:
        monitor_vol = float(baseline_cfg.get("monitor_volume_l", 605.0))
        sample_vol = float(baseline_cfg.get("sample_volume_l", 0.0))
    except (TypeError, ValueError) as exc:
        raise ValueError("baseline volumes must be numeric") from exc

    base_events = pd.DataFrame()
    baseline_live_time = 0.0
    mask_base = None

    # Extract baseline events if range is specified
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

        # Compute dilution factor
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

        # Set up scales for different isotopes
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
                # Use noise cutoff from config, fallback to default
                noise_thr_val = cfg.get("calibration", {}).get(
                    "noise_cutoff", DEFAULT_NOISE_CUTOFF
                )
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

    # Check if we should exit early (baseline check mode)
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

    # Apply baseline subtraction if requested
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

    return {
        "baseline_info": baseline_info,
        "baseline_counts": baseline_counts,
        "baseline_record": baseline_record,
        "dilution_factor": dilution_factor,
        "base_events": base_events,
        "baseline_range": baseline_range,
        "isotopes_to_subtract": isotopes_to_subtract,
        "df_analysis": df_analysis,
    }
