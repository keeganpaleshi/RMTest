"""
summary_stage.py

Summary JSON Construction and Writing
======================================

This module handles the construction and writing of the summary JSON file,
which aggregates all analysis results including:
- Calibration parameters
- Spectral fit results
- Time-series fit results
- Systematics analysis
- Baseline information
- Radon activity calculations
- Efficiency measurements
- Diagnostic information

The summary is written to the output directory along with associated plots.
"""

import logging
import math
import shutil
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import pandas as pd

from io_utils import Summary, copy_config, write_summary
from reporting import build_diagnostics
from fitting import FitResult
import baseline_handling
from analysis_helpers import (
    _fit_params,
    _radon_time_window,
    _total_radon_series,
    _resolved_efficiency,
    _save_stub_spectrum_plot,
    _ts_bin_centers_widths,
    _model_uncertainty,
    _segments_to_isotope_series,
    dedupe_isotope_series,
)
from plot_utils import (
    plot_spectrum,
    plot_spectrum_comparison,
    plot_activity_grid,
    plot_time_series,
)
from plotting_wrappers import (
    plot_radon_activity,
    plot_total_radon,
    plot_radon_trend,
)
from radon.external_rn_loader import load_external_rn_series
from radon.radon_inference import run_radon_inference
from radon.radon_plots import (
    plot_ambient_rn_vs_time,
    plot_rn_inferred_vs_time,
    plot_volume_equiv_vs_time,
)
from utils.time_utils import to_utc_datetime, to_datetime_utc


logger = logging.getLogger(__name__)


def build_and_write_summary(
    *,
    args,
    cfg: dict,
    spectrum_results,
    peak_deviation,
    time_fit_results: dict,
    time_fit_background_meta: dict,
    baseline_background_provenance: dict,
    baseline_record: dict,
    cal_params,
    cal_result,
    now_str: str,
    cfg_sha256: str,
    calibration_valid: bool,
    systematics_results: dict,
    baseline_info: dict,
    radon_results: dict,
    n_removed_noise: int,
    n_removed_burst: int,
    burst_mode: str,
    roi_diff: dict,
    scan_results: dict,
    best_params: tuple,
    drift_rate,
    drift_mode,
    drift_params,
    efficiency_results: dict,
    seed_used: int,
    commit: str,
    requirements_sha256: str,
    cli_sha256: str,
    cli_args: list,
    t0_cfg,
    t_end_cfg,
    spike_start_cfg,
    spike_end_cfg,
    spike_periods_cfg,
    run_periods_cfg,
    radon_interval_cfg,
    radon_combined_info,
    corrected_rates: dict,
    corrected_unc: dict,
    fit214,
    fit218,
    iso_live_time: dict,
    iso_mode: str,
    monitor_vol,
    sample_vol,
    weights,
    spec_plot_data: dict,
    df_analysis: pd.DataFrame,
    pre_spec_energies: np.ndarray,
    post_spec_energies: np.ndarray,
    t0_global,
    t_end_global,
    t_end_global_ts: float,
    analysis_start,
    radon_interval,
    time_plot_data: dict,
) -> tuple[Summary, Path, dict, dict]:
    """
    Build and write the summary JSON file with all analysis results.

    This function assembles results from all analysis stages into a comprehensive
    summary dictionary, writes it to disk, and generates associated plots.

    Parameters
    ----------
    args : argparse.Namespace
        Command-line arguments including output directory and configuration paths.
    cfg : dict
        Full analysis configuration dictionary.
    spectrum_results : FitResult or dict
        Results from spectral fitting stage.
    peak_deviation : dict or None
        Peak deviation information from calibration.
    time_fit_results : dict
        Time-series fit results for each isotope (Po214, Po218, Po210).
    time_fit_background_meta : dict
        Background mode metadata for each isotope's time fit.
    baseline_background_provenance : dict
        Provenance information for baseline background handling.
    baseline_record : dict
        Record of baseline measurements and calculations.
    cal_params : CalibrationParams or dict
        Energy calibration parameters and uncertainties.
    cal_result
        Full calibration result object.
    now_str : str
        Timestamp string for this analysis run.
    cfg_sha256 : str
        SHA256 hash of the configuration file.
    calibration_valid : bool
        Whether the calibration passed validation checks.
    systematics_results : dict
        Results from systematics uncertainty analysis.
    baseline_info : dict
        Baseline summary information.
    radon_results : dict
        Radon activity calculation results.
    n_removed_noise : int
        Number of events removed by noise filter.
    n_removed_burst : int
        Number of events removed by burst filter.
    burst_mode : str
        Burst filtering mode used.
    roi_diff : dict
        Region-of-interest differences for burst filtering.
    scan_results : dict
        Burst sensitivity scan results.
    best_params : tuple
        Best burst filter parameters (multiplier, window_size).
    drift_rate : float or None
        ADC drift rate in channels/s.
    drift_mode : str or None
        ADC drift correction mode.
    drift_params : dict
        ADC drift parameters.
    efficiency_results : dict
        Detection efficiency measurements and uncertainties.
    seed_used : int
        Random seed used for this analysis.
    commit : str
        Git commit hash of the analysis code.
    requirements_sha256 : str
        SHA256 hash of requirements.txt.
    cli_sha256 : str
        SHA256 hash of the CLI script.
    cli_args : list
        Command-line arguments as list of strings.
    t0_cfg : datetime or str
        Analysis start time from configuration.
    t_end_cfg : datetime or str
        Analysis end time from configuration.
    spike_start_cfg : datetime or str or None
        Spike start time from configuration.
    spike_end_cfg : datetime or str or None
        Spike end time from configuration.
    spike_periods_cfg : list or None
        Spike periods from configuration.
    run_periods_cfg : list or None
        Run periods from configuration.
    radon_interval_cfg : tuple or None
        Radon measurement interval from configuration.
    radon_combined_info : dict or None
        Combined radon information from multiple isotopes.
    corrected_rates : dict
        Baseline-corrected count rates for each isotope.
    corrected_unc : dict
        Uncertainties on corrected count rates.
    fit214
        Po-214 fit result object.
    fit218
        Po-218 fit result object.
    iso_live_time : dict
        Live time for each isotope measurement.
    iso_mode : str
        Analysis isotope mode ('radon', 'po214', or 'po218').
    monitor_vol : float or None
        Monitor volume in liters.
    sample_vol : float or None
        Sample volume in liters.
    weights : list or None
        Blue LED efficiency weights.
    spec_plot_data : dict
        Spectrum plot data including energies, bins, and fit values.
    df_analysis : pd.DataFrame
        Analysis dataframe with all event data.
    pre_spec_energies : np.ndarray
        Energy values before time window cuts (for pre/post comparison).
    post_spec_energies : np.ndarray
        Energy values after time window cuts (for pre/post comparison).
    t0_global : datetime
        Global analysis start time as datetime object.
    t_end_global : datetime
        Global analysis end time as datetime object.
    t_end_global_ts : float
        Global analysis end time as Unix timestamp.
    analysis_start : datetime
        Analysis start reference time.
    radon_interval : tuple or None
        Radon measurement interval as (start, end) datetimes.
    time_plot_data : dict
        Time-series plot data for each isotope.

    Returns
    -------
    summary : Summary
        Complete summary object with all analysis results.
    out_dir : Path
        Output directory path where results were written.
    isotope_series_data : dict
        Time-series data for each isotope.
    radon_inference_results : dict or None
        Results from radon inference analysis, if performed.
    """
    # 8. Assemble and write out the summary JSON
    # ────────────────────────────────────────────────────────────
    spec_dict = {}
    if isinstance(spectrum_results, FitResult):
        spec_dict = dict(spectrum_results.params)
        spec_dict["cov"] = spectrum_results.cov.tolist()
        spec_dict["ndf"] = spectrum_results.ndf
        spec_dict["likelihood_path"] = spectrum_results.params.get("likelihood_path")
    elif isinstance(spectrum_results, dict):
        spec_dict = spectrum_results
        spec_dict["likelihood_path"] = spectrum_results.get("likelihood_path")
    if peak_deviation:
        spec_dict["peak_deviation"] = peak_deviation

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
        meta = time_fit_background_meta.get(iso)
        if meta:
            d = dict(d)
            d["background_mode"] = meta.get("mode")
            if meta.get("baseline_rate_Bq") is not None:
                d["baseline_rate_Bq"] = meta["baseline_rate_Bq"]
            if meta.get("baseline_unc_Bq") is not None:
                d["baseline_unc_Bq"] = meta["baseline_unc_Bq"]
        time_fit_serializable[iso] = d

    baseline_handling.apply_time_fit_provenance(
        time_fit_serializable,
        baseline_background_provenance,
        baseline_record,
    )

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
        config_sha256=cfg_sha256,
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
            "roi_diff": roi_diff,
            "sensitivity_scan": {
                "grid": {f"{m}_{w}": v for (m, w), v in scan_results.items()},
                "best": {
                    "burst_multiplier": best_params[0] if best_params else None,
                    "burst_window_size_s": best_params[1] if best_params else None,
                },
            },
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
            "background_model": cfg.get("analysis", {}).get("background_model"),
            "likelihood": cfg.get("analysis", {}).get("likelihood"),
            "ambient_concentration": cfg.get("analysis", {}).get(
                "ambient_concentration"
            ),
            "settle_s": cfg.get("analysis", {}).get("settle_s"),
        },
    )

    if radon_combined_info is not None:
        summary.radon_combined = radon_combined_info

    from radon_joint_estimator import estimate_radon_activity

    rate214_final = corrected_rates.get("Po214") if corrected_rates else None
    err214_final = corrected_unc.get("Po214") if corrected_unc else None
    rate218_final = corrected_rates.get("Po218") if corrected_rates else None
    err218_final = corrected_unc.get("Po218") if corrected_unc else None

    if rate214_final is None and fit214:
        rate214_final = fit214.rate
        err214_final = fit214.err
    if rate218_final is None and fit218:
        rate218_final = fit218.rate
        err218_final = fit218.err

    radon = estimate_radon_activity(
        N218=fit218.counts if fit218 else 0,
        epsilon218=(
            _resolved_efficiency(cfg, "Po218", fit218.params) if fit218 else 1.0
        ),
        N214=fit214.counts if fit214 else 0,
        epsilon214=(
            _resolved_efficiency(cfg, "Po214", fit214.params) if fit214 else 1.0
        ),
        f218=1.0,
        f214=1.0,
        live_time218_s=iso_live_time.get("Po218") if fit218 else None,
        live_time214_s=iso_live_time.get("Po214") if fit214 else None,
        rate214=rate214_final,
        err214=err214_final,
        rate218=rate218_final,
        err218=err218_final,
        analysis_isotope=iso_mode,
    )

    # ── Construct a minimal time-series aligned with the measurement window ──
    ts_start, ts_end = _radon_time_window(t0_cfg, t_end_cfg, radon_interval_cfg)
    ts_points: list[float]
    if not math.isfinite(ts_start):
        ts_start = to_utc_datetime(t0_cfg).timestamp()
    if math.isfinite(ts_end) and ts_end > ts_start:
        ts_points = [float(ts_start), float(ts_end)]
    else:
        ts_points = [float(ts_start)]

    gaussian_ok = bool(radon.get("gaussian_uncertainty_valid", True))
    stat_unc = radon.get("stat_unc_Bq") if gaussian_ok else float("nan")
    try:
        stat_unc_val = float(stat_unc) if stat_unc is not None else float("nan")
    except (TypeError, ValueError):
        stat_unc_val = float("nan")
    if not math.isfinite(stat_unc_val) or stat_unc_val < 0:
        stat_unc_val = float("nan")
    errors_ts = [stat_unc_val] * len(ts_points)

    radon["time_series"] = {
        "time": ts_points,
        "activity": [float(radon["Rn_activity_Bq"])] * len(ts_points),
    }
    radon["time_series"]["error"] = errors_ts

    total_vals, total_errs = _total_radon_series(
        radon["time_series"]["activity"],
        radon["time_series"].get("error"),
        monitor_vol,
        sample_vol,
    )
    total_ts = {
        "time": list(radon["time_series"]["time"]),
        "activity": total_vals.tolist(),
    }
    if total_errs is not None:
        total_ts["error"] = total_errs.tolist()
    radon["total_time_series"] = total_ts

    summary["radon"] = radon

    if weights is not None:
        summary.efficiency = summary.efficiency or {}
        summary.efficiency["blue_weights"] = list(weights)

    summary.diagnostics = build_diagnostics(
        summary, spectrum_results, time_fit_results, df_analysis, cfg
    )

    drift_flag, drift_msg = baseline_handling.assess_baseline_drift(
        baseline_record,
        cal_result,
        cfg.get("baseline", {}),
    )
    if drift_flag:
        summary.diagnostics = summary.diagnostics or {}
        summary.diagnostics["baseline_compat_warning"] = True
        if drift_msg:
            warn_list = summary.diagnostics.setdefault("warnings", [])
            if drift_msg not in warn_list:
                warn_list.append(drift_msg)

    results_dir = Path(args.output_dir) / (args.job_id or now_str)
    if results_dir.exists():
        if args.overwrite:
            shutil.rmtree(results_dir)
        else:
            raise FileExistsError(f"Results folder already exists: {results_dir}")

    copy_config(results_dir, cfg, exist_ok=args.overwrite)
    out_dir = Path(write_summary(results_dir, summary))
    out_dir.mkdir(parents=True, exist_ok=True)

    radon_background_mode = locals().get("background_mode")
    if (
        iso_mode == "radon"
        and radon_background_mode is None
        and hasattr(summary, "get")
    ):
        summary_radon = summary.get("radon")
        if isinstance(summary_radon, Mapping):
            plot_payload = summary_radon.get("plot_series")
            if isinstance(plot_payload, Mapping):
                radon_background_mode = plot_payload.get("background_mode")

    if iso_mode == "radon" and "radon" in summary:
        rad_ts = summary["radon"]["time_series"]

        plot_radon_activity(
            rad_ts["time"],
            rad_ts["activity"],
            Path(out_dir) / "radon_activity.png",
            rad_ts.get("error"),
            config=cfg.get("plotting", {}),
            sample_volume_l=sample_vol,
            background_mode=radon_background_mode,
        )
        total_vals, total_errs = _total_radon_series(
            rad_ts["activity"],
            rad_ts.get("error"),
            monitor_vol,
            sample_vol,
        )
        plot_total_radon(
            rad_ts["time"],
            total_vals,
            Path(out_dir) / "total_radon.png",
            total_errs,
            config=cfg.get("plotting", {}),
            background_mode=radon_background_mode,
        )
        plot_radon_trend(
            rad_ts["time"],
            rad_ts["activity"],
            Path(out_dir) / "radon_trend.png",
            config=cfg.get("plotting", {}),
        )

    # Generate plots now that the output directory exists
    spectrum_png = Path(out_dir) / "spectrum.png"
    spectrum_components_png = Path(out_dir) / "spectrum_components.png"
    if spec_plot_data:
        try:
            _ = plot_spectrum(
                energies=spec_plot_data["energies"],
                fit_vals=spec_plot_data["fit_vals"],
                out_png=spectrum_png,
                bins=spec_plot_data["bins"],
                bin_edges=spec_plot_data["bin_edges"],
                config=cfg.get("plotting", {}),
                fit_flags=spec_plot_data.get("flags"),
            )
        except Exception as e:
            logger.warning("Could not create spectrum plot: %s", e)

        try:
            _ = plot_spectrum(
                energies=spec_plot_data["energies"],
                fit_vals=spec_plot_data["fit_vals"],
                out_png=spectrum_components_png,
                bins=spec_plot_data["bins"],
                bin_edges=spec_plot_data["bin_edges"],
                config=cfg.get("plotting", {}),
                fit_flags=spec_plot_data.get("flags"),
                show_total_model=False,
            )
        except Exception as e:
            logger.warning("Could not create component spectrum plot: %s", e)

    if not spectrum_png.exists():
        try:
            stub_bins = spec_plot_data["bins"] if spec_plot_data else None
            stub_edges = spec_plot_data.get("bin_edges") if spec_plot_data else None
        except Exception:
            stub_bins = None
            stub_edges = None
        try:
            energies_stub = (
                spec_plot_data.get("energies")
                if spec_plot_data and isinstance(spec_plot_data, dict)
                else df_analysis.get("energy_MeV", pd.Series(dtype=float)).to_numpy()
            )
        except Exception:
            energies_stub = (
                df_analysis.get("energy_MeV", pd.Series(dtype=float)).to_numpy()
                if isinstance(df_analysis, pd.DataFrame)
                else np.asarray([], dtype=float)
            )
        try:
            _save_stub_spectrum_plot(
                energies_stub,
                spectrum_png,
                bins=stub_bins,
                bin_edges=stub_edges,
                config=cfg.get("plotting", {}),
            )
            logger.info(
                "Saved fallback spectrum plot after unavailable fit to %s", spectrum_png
            )
        except Exception as e:
            logger.warning("Could not create fallback spectrum plot: %s", e)

        try:
            _save_stub_spectrum_plot(
                energies_stub,
                spectrum_components_png,
                bins=stub_bins,
                bin_edges=stub_edges,
                config=cfg.get("plotting", {}),
            )
            logger.info(
                "Saved fallback component spectrum plot after unavailable fit to %s",
                spectrum_components_png,
            )
        except Exception as e:
            logger.warning(
                "Could not create fallback component spectrum plot: %s", e
            )

    try:
        _ = plot_spectrum_comparison(
            pre_spec_energies,
            post_spec_energies,
            bins=spec_plot_data.get("bins", 400) if spec_plot_data else 400,
            bin_edges=spec_plot_data.get("bin_edges") if spec_plot_data else None,
            out_png=Path(out_dir) / "spectrum_pre_post.png",
            config=cfg.get("time_fit", {}),
        )
    except Exception as e:
        logger.warning("Could not create pre/post spectrum plot -> %s", e)

    if args.burst_sensitivity_scan and scan_results:
        try:
            plot_activity_grid(
                scan_results,
                out_png=Path(out_dir) / "burst_scan.png",
                config=cfg.get("plotting", {}),
            )
        except Exception as e:
            logger.warning("Could not create burst scan plot -> %s", e)

    overlay = cfg.get("plotting", {}).get("overlay_isotopes", False)
    save_individual_ts = bool(
        cfg.get("plotting", {}).get("save_individual_time_series", False)
    )
    isotope_series_data: dict[str, list[dict[str, float]]] = {}

    if overlay:
        # Guarantee plotting payloads exist for both daughter isotopes even
        # when the fit skipped one of them so the overlay consistently
        # generates Po-214 and Po-218 images.
        tf_cfg = cfg.get("time_fit", {})
        ts_start = to_datetime_utc(t0_global)
        for iso in ("Po218", "Po214"):
            if iso in time_plot_data:
                continue

            win_key = f"window_{iso.lower()}"
            win_range = tf_cfg.get(win_key) if isinstance(tf_cfg, Mapping) else None
            if win_range is None:
                continue

            lo, hi = win_range
            try:
                mask = (
                    (df_analysis["energy_MeV"] >= lo)
                    & (df_analysis["energy_MeV"] <= hi)
                    & (df_analysis["timestamp"] >= ts_start)
                    & (df_analysis["timestamp"] <= t_end_global)
                )
                iso_events = df_analysis.loc[mask]
            except Exception:
                continue

            time_plot_data[iso] = {
                "events_times": iso_events["timestamp"].values,
                "events_energy": iso_events["energy_MeV"].values,
            }

    def _prepare_model_errors(
        ts_times: np.ndarray, plot_cfg: Mapping[str, Any], iso_names: list[str]
    ) -> dict[str, np.ndarray]:
        """Return model uncertainty arrays for the requested isotopes."""

        centers, widths = _ts_bin_centers_widths(
            ts_times, plot_cfg, t0_global.timestamp(), t_end_global_ts
        )
        normalise = bool(plot_cfg.get("plot_time_normalise_rate", False))
        model_errs: dict[str, np.ndarray] = {}
        for iso_key in iso_names:
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
        return model_errs

    for iso in ("Po218", "Po214", "Po210"):
        pdata = time_plot_data.get(iso)
        if pdata is None:
            continue
        try:
            plot_cfg = dict(cfg.get("time_fit", {}))
            plot_cfg.update(cfg.get("plotting", {}))
            if run_periods_cfg:
                plot_cfg["run_periods"] = run_periods_cfg
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

            iso_list_err = (
                [iso]
                if not overlay
                else [i for i in ("Po214", "Po218", "Po210") if time_fit_results.get(i)]
            )
            model_errs = _prepare_model_errors(ts_times, plot_cfg, iso_list_err)

            ts_info = plot_time_series(
                all_timestamps=ts_times,
                all_energies=ts_energy,
                fit_results=fit_dict,
                t_start=t0_global.timestamp(),
                t_end=t_end_global_ts,
                config=plot_cfg,
                out_png=Path(out_dir) / f"time_series_{iso}.png",
                model_errors=model_errs,
            )
            if ts_info:
                series_map = _segments_to_isotope_series(ts_info)
                for iso_key, entries in series_map.items():
                    if not entries:
                        continue
                    existing = isotope_series_data.setdefault(iso_key, [])
                    existing.extend(entries)

            if overlay and save_individual_ts:
                indiv_cfg = dict(plot_cfg)
                for other_iso in ("Po214", "Po218", "Po210"):
                    if other_iso != iso:
                        indiv_cfg[f"window_{other_iso.lower()}"] = None
                indiv_times = pdata["events_times"]
                indiv_energy = pdata["events_energy"]
                indiv_errs = _prepare_model_errors(indiv_times, indiv_cfg, [iso])
                indiv_fit_dict = _fit_params(time_fit_results.get(iso))
                _ = plot_time_series(
                    all_timestamps=indiv_times,
                    all_energies=indiv_energy,
                    fit_results=indiv_fit_dict,
                    t_start=t0_global.timestamp(),
                    t_end=t_end_global_ts,
                    config=indiv_cfg,
                    out_png=Path(out_dir) / f"time_series_{iso}_individual.png",
                    model_errors=indiv_errs,
                )
        except Exception as e:
            logger.warning("Could not create time-series plot for %s -> %s", iso, e)

    for iso_entries in isotope_series_data.values():
        iso_entries.sort(key=lambda row: row.get("t", 0.0))

    # Deduplicate isotope series data to prevent duplicate bins when overlay_isotopes is enabled
    isotope_series_data = dedupe_isotope_series(isotope_series_data)

    radon_inference_results = None
    radon_inference_cfg = cfg.get("radon_inference")
    if isotope_series_data and isinstance(radon_inference_cfg, Mapping):
        timestamps_for_external = sorted(
            {
                float(entry.get("t"))
                for entries in isotope_series_data.values()
                for entry in entries
                if entry.get("t") is not None
            }
        )
        external_series = None
        if timestamps_for_external and radon_inference_cfg.get("enabled", False):
            try:
                raw_external = load_external_rn_series(
                    radon_inference_cfg.get("external_rn"), timestamps_for_external
                )
            except Exception as exc:
                logger.warning("Failed to load external radon series: %s", exc)
                raw_external = []

            if raw_external:
                external_series = []
                for ts_obj, value in raw_external:
                    try:
                        if hasattr(ts_obj, "timestamp"):
                            t_val = float(ts_obj.timestamp())
                        else:
                            t_val = float(ts_obj)
                        val_float = float(value)
                    except (TypeError, ValueError):
                        continue
                    if not np.isfinite(t_val) or not np.isfinite(val_float):
                        continue
                    external_series.append({"t": t_val, "rn_bq_per_m3": val_float})

        radon_inference_results = run_radon_inference(
            isotope_series_data,
            cfg,
            external_series,
        )
        if radon_inference_results:
            summary["radon_inference"] = radon_inference_results
            try:
                plot_rn_inferred_vs_time(radon_inference_results, Path(out_dir))
                plot_ambient_rn_vs_time(radon_inference_results, Path(out_dir))
                plot_volume_equiv_vs_time(radon_inference_results, Path(out_dir))
            except Exception as exc:
                logger.warning("Failed to create radon inference plots: %s", exc)

    return summary, out_dir, isotope_series_data, radon_inference_results
