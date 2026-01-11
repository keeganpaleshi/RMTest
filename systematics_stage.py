"""
systematics_stage.py

Systematics Analysis Stage
============================

This module handles the systematics scan, efficiency calculations, baseline
subtraction, and radon activity extrapolation for the analysis pipeline.
"""

import logging
import math
import sys
from typing import Any, Mapping

import numpy as np

import radon_activity
from analysis_helpers import (
    _config_efficiency,
    _cov_lookup,
    _fallback_uncertainty,
    _fit_params,
    _float_with_default,
    _hl_value,
    _resolved_efficiency,
    _safe_float,
    get_spike_efficiency,
    window_prob,
)
from baseline_utils import (
    subtract_baseline_counts,
    subtract_baseline_rate,
    compute_dilution_factor,
    summarize_baseline,
    BaselineError,
)
import baseline_handling
from constants import NEGATIVE_ACTIVITY_CLAMP_UNCERTAINTY_BQ
from efficiency import calc_assay_efficiency, calc_decay_efficiency, blue_combine
from fitting import fit_time_series
from radon_activity import compute_radon_activity, radon_delta
from systematics import scan_systematics
from utils.time_utils import to_epoch_seconds

logger = logging.getLogger(__name__)


def run_systematics_stage(
    cfg: dict[str, Any],
    args: Any,
    df_analysis: Any,
    events_after_noise: Any,
    cal_result: Any,
    priors_time_all: dict[str, Any],
    time_fit_results: dict[str, Any],
    t0_global: Any,
    t_end_global_ts: float,
    baseline_record: dict[str, Any] | None,
    baseline_info: dict[str, Any],
    baseline_counts: dict[str, float],
    baseline_live_time: float,
    isotopes_to_subtract: list[str],
    iso_live_time: dict[str, float],
    iso_counts_raw: dict[str, float],
    monitor_vol: float,
    sample_vol: float,
    dilution_factor: float | None,
    baseline_background_provenance: dict[str, Any],
    allow_negative_baseline: bool,
    analysis_start: Any,
    radon_interval: tuple | None,
) -> dict[str, Any]:
    """
    Run the systematics analysis stage.

    This stage performs:
    1. Systematics scan (optional) - evaluates systematic uncertainties
    2. Efficiency calculations - from spike, assay, and decay methods
    3. Baseline subtraction - corrects activities for background
    4. Radon activity extrapolation - computes final radon activity
    5. Burst sensitivity scan (optional) - evaluates burst filter parameters

    Parameters
    ----------
    cfg : dict
        Configuration dictionary with analysis settings
    args : Namespace
        Command-line arguments including:
        - strict_covariance: bool
        - baseline_mode: str
        - burst_sensitivity_scan: bool
        - debug: bool
        - allow_negative_activity: bool
    df_analysis : DataFrame
        Analysis DataFrame with filtered events
    events_after_noise : DataFrame
        Events after noise filtering (for burst scan)
    cal_result : CalibrationResult
        Energy calibration result
    priors_time_all : dict
        Priors for time-series fitting by isotope
    time_fit_results : dict
        Time-series fit results by isotope
    t0_global : datetime
        Global analysis start time
    t_end_global_ts : float
        Global analysis end timestamp (seconds)
    baseline_record : dict, optional
        External baseline record
    baseline_info : dict
        Baseline information dictionary (will be modified)
    baseline_counts : dict
        Baseline event counts by isotope
    baseline_live_time : float
        Live time for baseline period (seconds)
    isotopes_to_subtract : list
        Isotopes for baseline subtraction
    iso_live_time : dict
        Live time by isotope (seconds)
    iso_counts_raw : dict
        Raw event counts by isotope
    monitor_vol : float
        Monitor volume (liters)
    sample_vol : float
        Sample volume (liters)
    dilution_factor : float, optional
        Dilution factor for baseline scaling
    baseline_background_provenance : dict
        Provenance information for baseline background
    allow_negative_baseline : bool
        Whether to allow negative baseline-corrected rates
    analysis_start : datetime
        Analysis start datetime
    radon_interval : tuple, optional
        Radon measurement interval (start, end)

    Returns
    -------
    dict
        Dictionary containing:
        - systematics_results: Systematic uncertainty results
        - efficiency_results: Efficiency calculation results
        - weights: BLUE combination weights (or None)
        - baseline_info: Updated baseline information
        - baseline_rates: Baseline rates by isotope
        - baseline_unc: Baseline uncertainties by isotope
        - corrected_rates: Baseline-corrected rates by isotope
        - corrected_unc: Baseline-corrected uncertainties by isotope
        - radon_results: Radon activity results
        - radon_combined_info: Combined radon info (or None)
        - scan_results: Burst sensitivity scan results
        - best_params: Best burst filter parameters (or None)
    """
    from analysis_helpers import _burst_sensitivity_scan

    # ────────────────────────────────────────────────────────────
    # 1. Systematics scan (optional)
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
    # 2. Optional efficiency calculations
    # ────────────────────────────────────────────────────────────
    efficiency_results = {}
    weights = None
    eff_cfg = cfg.get("efficiency", {})
    if eff_cfg:
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
    # 3. Baseline subtraction
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

    # Helper function for counts-based fallback rate
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
    # 4. Radon activity extrapolation
    # ────────────────────────────────────────────────────────────
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

    # ────────────────────────────────────────────────────────────
    # 5. Burst sensitivity scan (optional)
    # ────────────────────────────────────────────────────────────
    scan_results = {}
    best_params = None
    if args.burst_sensitivity_scan:
        scan_results, best_params = _burst_sensitivity_scan(
            events_after_noise, cfg, cal_result
        )

    # ────────────────────────────────────────────────────────────
    # 6. Debug output (optional)
    # ────────────────────────────────────────────────────────────
    if args.debug:
        from radon_activity import print_activity_breakdown

        print_activity_breakdown(activity_rows)

    # ────────────────────────────────────────────────────────────
    # 7. Radon delta calculation (optional)
    # ────────────────────────────────────────────────────────────
    if radon_interval is not None:
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
    # Return all results
    # ────────────────────────────────────────────────────────────
    return {
        "systematics_results": systematics_results,
        "efficiency_results": efficiency_results,
        "weights": weights,
        "baseline_info": baseline_info,
        "baseline_rates": baseline_rates,
        "baseline_unc": baseline_unc,
        "corrected_rates": corrected_rates,
        "corrected_unc": corrected_unc,
        "radon_results": radon_results,
        "radon_combined_info": radon_combined_info,
        "scan_results": scan_results,
        "best_params": best_params,
    }
