"""
data_loading.py

Data loading and initial preprocessing for the analysis pipeline.

This module handles:
- Loading event data from CSV
- Applying noise cutoffs
- Applying burst filters
- Setting up time windows and global time references
"""

import logging
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Tuple, Optional, Any, Dict

from io_utils import load_events, apply_burst_filter
from analysis_helpers import _ensure_events
from utils.time_utils import parse_timestamp
from utils import to_utc_datetime


logger = logging.getLogger(__name__)


def load_and_filter_events(
    input_path: Path,
    cfg: Dict[str, Any],
    args: Any,
    timer: Any,
) -> Tuple[pd.DataFrame, pd.DataFrame, int, int]:
    """
    Load events from CSV and apply noise and burst filters.

    Returns:
        tuple: (events_all, events_filtered, n_removed_noise, n_removed_burst)
    """
    # Load events
    with timer.section("load_events"):
        try:
            events_all = load_events(input_path, column_map=cfg.get("columns"))
            events_all["timestamp"] = events_all["timestamp"].map(parse_timestamp)
        except Exception as e:
            logger.error("Could not load events from '%s': %s", input_path, e)
            raise

    if events_all.empty:
        logger.info("No events found in the input CSV. Exiting.")
        return events_all, events_all, 0, 0

    # Apply noise cut
    noise_thr = cfg.get("calibration", {}).get("noise_cutoff")
    n_removed_noise = 0
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
                before_noise = len(events_filtered)
                events_filtered = events_filtered[
                    events_filtered["adc"] >= noise_thr_val
                ].copy()
                n_removed_noise = before_noise - len(events_filtered)
                if n_removed_noise > 0:
                    logging.info(
                        f"Removed {n_removed_noise} events below ADC threshold {noise_thr_val}"
                    )

    # Apply burst filter
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

    return events_all, events_filtered, n_removed_noise, n_removed_burst


def setup_time_windows(cfg: Dict[str, Any], events_filtered: pd.DataFrame) -> Dict[str, Any]:
    """
    Setup global time windows and parse time periods from config.

    Returns:
        dict: Contains t0_global, t_end_global, t_end_global_ts, spike_start, spike_end,
              spike_periods, run_periods, radon_interval, and updated config entries
    """
    result = {}

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

    result["t0_global"] = t0_global
    result["t0_cfg"] = t0_cfg

    # Global end time
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

    result["t_end_global"] = t_end_global
    result["t_end_global_ts"] = t_end_global_ts
    result["t_end_cfg"] = t_end_cfg

    # Spike start/end times
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

    result["t_spike_start"] = t_spike_start
    result["spike_start_cfg"] = spike_start_cfg

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

    result["t_spike_end"] = t_spike_end
    result["spike_end_cfg"] = spike_end_cfg

    # Spike periods
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

    result["spike_periods"] = spike_periods
    result["spike_periods_cfg"] = spike_periods_cfg

    # Run periods
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

    result["run_periods"] = run_periods
    result["run_periods_cfg"] = run_periods_cfg

    # Radon interval
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

    result["radon_interval"] = radon_interval
    result["radon_interval_cfg"] = radon_interval_cfg

    return result
