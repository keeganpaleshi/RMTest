"""
spectral_fit_stage.py

Spectral fit stage for the radon monitor pipeline.
Handles binning, peak finding, prior setup, and spectral fitting.
"""

import logging
from typing import Any, Optional, Dict, Tuple

import numpy as np
import pandas as pd

from calibration import apply_calibration
from constants import DEFAULT_ADC_CENTROIDS
from utils import find_adc_bin_peaks
from fitting import fit_spectrum, FitResult
from analysis_helpers import (
    _normalise_mu_bounds,
    _spectral_fit_with_check,
)

logger = logging.getLogger(__name__)


def run_spectral_fit_stage(
    df_analysis: pd.DataFrame,
    cfg: dict,
    args,
    cal_params: dict,
) -> dict[str, Any]:
    """
    Run the spectral fit stage.

    Parameters
    ----------
    df_analysis : pd.DataFrame
        Analysis dataframe with calibrated energies
    cfg : dict
        Configuration dictionary
    args : argparse.Namespace
        Command-line arguments
    cal_params : dict
        Calibration parameters containing:
        - a: linear calibration coefficient
        - c: offset calibration coefficient
        - a2: quadratic calibration coefficient (optional)
        - sigE_mean: mean energy resolution
        - sigE_sigma: uncertainty on energy resolution

    Returns
    -------
    dict
        Dictionary containing:
        - spectrum_results: spectral fit results (FitResult or dict)
        - spec_plot_data: dictionary with plotting data
        - peak_deviation: peak deviation metrics
    """
    spectrum_results = {}
    spec_plot_data = None
    peak_deviation = {}

    if not cfg.get("spectral_fit", {}).get("do_spectral_fit", False):
        return {
            "spectrum_results": spectrum_results,
            "spec_plot_data": spec_plot_data,
            "peak_deviation": peak_deviation,
        }

    # Extract calibration parameters
    a = cal_params["a"]
    c = cal_params["c"]
    a2 = cal_params.get("a2", 0.0)
    sigE_mean = cal_params["sigE_mean"]
    sigE_sigma = cal_params["sigE_sigma"]

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

    # Determine binning based on method
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
        for i, coeff in enumerate(coeffs):
            priors_spec[f"b{i}"] = (float(coeff), abs(float(coeff)) * 0.1 + 1e-3)
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

    return {
        "spectrum_results": spectrum_results,
        "spec_plot_data": spec_plot_data,
        "peak_deviation": peak_deviation,
    }
