"""
time_series_stage.py

Time-series decay fit pipeline stage.

Handles time-series decay fitting for Po-218 and Po-214 isotopes:
- Extracts events in energy windows for each isotope
- Performs baseline correction when applicable
- Runs two-pass time-series fits with efficiency and background modeling
- Combines results to estimate radon activity
- Prepares data for time-series plotting
"""

import logging
import math
from datetime import datetime, timedelta
from typing import Any, Mapping, Dict, Tuple, Optional
import numpy as np
import pandas as pd
from types import SimpleNamespace

from fitting import fit_time_series, FitResult
from analysis_helpers import (
    window_prob,
    auto_expand_window,
    _eff_prior,
    _config_efficiency,
    _hl_value,
    _fit_params,
    _fallback_uncertainty,
    _resolved_efficiency,
    _fit_efficiency,
)
from baseline_utils import subtract_baseline_counts
from time_fitting import two_pass_time_fit
from utils.time_utils import to_datetime_utc, to_epoch_seconds, to_utc_datetime
import baseline_handling
from radon_joint_estimator import estimate_radon_activity

logger = logging.getLogger(__name__)


def run_time_series_stage(
    df_analysis: pd.DataFrame,
    cfg: Dict[str, Any],
    args: Any,
    t0_global: datetime,
    t_end_global: datetime,
    t_end_global_ts: float,
    baseline_range: Optional[Tuple[datetime, datetime]] = None,
    base_events: Optional[pd.DataFrame] = None,
    baseline_record: Optional[Any] = None,
    baseline_live_time: float = 0.0,
    baseline_counts: Optional[Dict[str, float]] = None,
    isotopes_to_subtract: Optional[list] = None,
    baseline_info: Optional[Dict[str, Any]] = None,
    baseline_background_provenance: Optional[Dict[str, Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    """
    Perform time-series decay fits for Po-218 and Po-214.

    This function:
    1. Extracts events within energy windows for each isotope
    2. Applies baseline corrections when applicable
    3. Builds priors from configuration and baseline data
    4. Performs two-pass time-series fits
    5. Combines isotope results to estimate radon activity
    6. Prepares plot data for visualization

    Args:
        df_analysis: DataFrame with calibrated event data (must include energy_MeV,
                     denergy_MeV, timestamp columns)
        cfg: Configuration dictionary containing time_fit settings
        args: Command-line arguments with settle_s, eff_fixed, strict_covariance,
              baseline_mode flags
        t0_global: Global analysis start time
        t_end_global: Global analysis end time
        t_end_global_ts: Global analysis end time as Unix timestamp
        baseline_range: Optional tuple of (start, end) datetimes for baseline period
        base_events: Optional DataFrame of baseline events
        baseline_record: Optional baseline record object for tracking baseline info
        baseline_live_time: Live time of baseline period in seconds (default: 0.0)
        baseline_counts: Optional dict mapping isotope names to baseline count values
        isotopes_to_subtract: Optional list of isotope names to apply baseline
                             subtraction (default: None)
        baseline_info: Optional dict to store baseline information, modified in place
        baseline_background_provenance: Optional dict to track background parameter
                                       provenance, modified in place

    Returns:
        Dictionary containing:
            - time_fit_results: Dict[str, FitResult] - Fit results per isotope
            - time_fit_background_meta: Dict[str, Dict] - Background metadata per isotope
            - priors_time_all: Dict[str, Dict] - Priors used for each isotope
            - time_plot_data: Dict[str, Dict] - Event data for plotting per isotope
            - iso_live_time: Dict[str, float] - Live time per isotope in seconds
            - t_start_map: Dict[str, datetime] - Fit start time per isotope
            - iso_counts: Dict[str, float] - Counts per isotope (currently unused)
            - iso_counts_raw: Dict[str, float] - Raw counts per isotope
            - radon_estimate_info: Optional[Dict] - Combined radon activity estimate
            - po214_estimate_info: Optional[Dict] - Po-214 only activity estimate
            - po218_estimate_info: Optional[Dict] - Po-218 only activity estimate

    Notes:
        - The baseline_info and baseline_background_provenance dicts are modified in place
        - If time_fit.do_time_fit is False in config, returns empty results
        - Energy windows are auto-expanded if min_counts threshold not met
        - Two-pass fitting: first pass with fixed background, second with floating
    """
    # Initialize output variables
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

    # Initialize baseline-related variables if not provided
    if baseline_counts is None:
        baseline_counts = {}
    if isotopes_to_subtract is None:
        isotopes_to_subtract = []
    if baseline_info is None:
        baseline_info = {}
    if baseline_background_provenance is None:
        baseline_background_provenance = {}
    if base_events is None:
        base_events = pd.DataFrame()

    allow_negative_baseline = bool(cfg.get("allow_negative_baseline"))

    if not cfg.get("time_fit", {}).get("do_time_fit", False):
        # Time fit disabled, return empty results
        return {
            "time_fit_results": time_fit_results,
            "time_fit_background_meta": time_fit_background_meta,
            "priors_time_all": priors_time_all,
            "time_plot_data": time_plot_data,
            "iso_live_time": iso_live_time,
            "t_start_map": t_start_map,
            "iso_counts": iso_counts,
            "iso_counts_raw": iso_counts_raw,
            "radon_estimate_info": radon_estimate_info,
            "po214_estimate_info": po214_estimate_info,
            "po218_estimate_info": po218_estimate_info,
        }

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
            if not allow_negative_baseline and c_rate < 0.0:
                c_rate = 0.0
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
            "fit_initial": not cfg["time_fit"]["flags"].get(
                f"fix_N0_{iso.lower()}", False
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

    return {
        "time_fit_results": time_fit_results,
        "time_fit_background_meta": time_fit_background_meta,
        "priors_time_all": priors_time_all,
        "time_plot_data": time_plot_data,
        "iso_live_time": iso_live_time,
        "t_start_map": t_start_map,
        "iso_counts": iso_counts,
        "iso_counts_raw": iso_counts_raw,
        "radon_estimate_info": radon_estimate_info,
        "po214_estimate_info": po214_estimate_info,
        "po218_estimate_info": po218_estimate_info,
    }
