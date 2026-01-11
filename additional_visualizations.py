"""
additional_visualizations.py

Additional visualization generation for the radon monitor analysis pipeline.
This module handles:
- Efficiency plots (covariance heatmap, efficiency bar charts)
- Radon activity time series plots
- Total radon plots
- Radon trend plots
- Equivalent air volume plots
- Hierarchical summary generation
"""

import logging
import math
import json
from pathlib import Path
from typing import Any, Mapping, Optional

import numpy as np

from visualize import cov_heatmap, efficiency_bar
from plotting_wrappers import (
    plot_radon_activity,
    plot_total_radon,
    plot_radon_trend,
)
from plot_utils import plot_equivalent_air
from analysis_helpers import (
    _fit_params,
    _radon_time_window,
    _radon_activity_curve_from_fit,
    _radon_background_mode,
    _regrid_series,
    _total_radon_series,
)
import baseline_handling
from utils.time_utils import to_datetime_utc
from hierarchical import fit_hierarchical_runs
import radon_activity
from radon_activity import compute_radon_activity

logger = logging.getLogger(__name__)


def generate_additional_visualizations(
    timer,
    efficiency_results: dict,
    cfg: dict,
    out_dir: Path,
    summary: dict,
    t0_global,
    t_end_global_ts: float,
    radon_interval,
    time_fit_results: dict,
    analysis_start,
    radon_combined_info: Optional[dict],
    sample_vol: float,
    monitor_vol: float,
    radon_results: Optional[dict],
    args,
):
    """
    Generate additional visualizations for the radon monitor analysis.

    Parameters
    ----------
    timer : PipelineTimer
        Timer for tracking performance
    efficiency_results : dict
        Efficiency calculation results with 'sources' key
    cfg : dict
        Configuration dictionary
    out_dir : Path
        Output directory for plots
    summary : dict
        Summary dictionary (modified in-place)
    t0_global : datetime
        Global start time
    t_end_global_ts : float
        Global end timestamp in seconds
    radon_interval : tuple or None
        Radon analysis interval (start_dt, end_dt)
    time_fit_results : dict
        Time series fit results for isotopes
    analysis_start : datetime
        Analysis start time
    radon_combined_info : dict or None
        Combined radon activity information
    sample_vol : float
        Sample volume in liters
    monitor_vol : float
        Monitor volume in liters
    radon_results : dict or None
        Radon estimation results
    args : argparse.Namespace
        Command-line arguments
    """
    with timer.section("additional_visualizations"):
        # Efficiency visualization
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

    # Hierarchical summary
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
