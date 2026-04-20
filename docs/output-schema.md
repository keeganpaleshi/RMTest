# Output Schema and Artifacts

Each analysis run writes a timestamped result directory under `<output-dir>/`, or `<output-dir>/<job-id>/` when `--job-id` is provided.

This page documents the stable filenames, the top-level structure of `summary.json`, and a few important semantics that are easy to miss when consuming the outputs programmatically.

Canonical output keys are lowercase (for example `rn_activity_bq` and `sigma_e_error`). Legacy mixed-case aliases are still written for compatibility.

## Common Files

These outputs are part of the normal analysis flow:

- `summary.json`: structured machine-readable summary
- `config_used.json`: archived copy of the configuration used for the run
- `spectrum.png`: stacked spectrum view with linear and log spectrum panels plus raw, log-raw, normalized, and fractional residual panels
- `spectrum_components.png`: same stacked spectrum view without the summed total-model curve
- `spectrum_pre_post.png`: pre-cut versus post-cut spectrum comparison
- `radon_activity.png`: radon activity and concentration versus time
- `total_radon.png`: total radon in the combined monitor-plus-sample volume
- `radon_trend.png`: trend view of the radon concentration
- `isotope_time_series.png`: overlaid isotope time series when overlay plotting is enabled
- `time_series_Po214.png`, `time_series_Po218.png`: per-isotope time-series plots

`time_series_Po210.png` is written when `window_po210` is configured.

## Optional Files

Additional artifacts appear when the corresponding feature is enabled:

- `time_series_<iso>_ts.json`: exported time-series payloads when `dump_time_series_json` is enabled
- `template_per_bin_diagnostics.json`: per-bin template-fit diagnostics when `time_fit.extraction_method: template`
- `template_bin_fits/`: per-bin template spectrum plots when `plot_template_bin_fits` is enabled
- `template_bin_fits/template_fit_plot_index.json`: manifest for the saved per-bin template plots; entries point at the main PNG, which already contains both linear and log spectrum panels
- `efficiency.png`, `eff_cov.png`: efficiency summary plots
- `burst_scan.png`: burst-sensitivity scan output
- `radon_activity_combined.png`: combined radon-activity diagnostic plot
- `radon_activity_po214.png`: Po-214-only activity plot
- `equivalent_air.png`, `equivalent_air_po214.png`: equivalent-air plots when ambient concentration information is available

When `radon_inference.enabled` is true, the stage may also write:

- `radon_inferred.png`
- `ambient_radon.png`
- `equivalent_volume_liters.png`
- `equivalent_flow.png`
- `equivalent_volume_cumulative.png`
- `equivalent_volume_cumulative_liters.png`

## `summary.json` Top-Level Structure

The top-level summary payload is defined by the `Summary` dataclass in `io_utils.py` and contains these primary keys:

```json
{
  "timestamp": "...",
  "config_used": "...",
  "config_sha256": "...",
  "calibration": {},
  "calibration_valid": true,
  "spectral_fit": {},
  "time_fit": {},
  "systematics": {},
  "baseline": {},
  "radon_results": {},
  "radon_combined": {},
  "radon_inference": {},
  "radon": {},
  "po214": {},
  "po218": {},
  "noise_cut": {},
  "burst_filter": {},
  "adc_drift_rate": 0.0,
  "adc_drift_mode": "linear",
  "adc_drift_params": {},
  "efficiency": {},
  "random_seed": 67,
  "git_commit": "...",
  "requirements_sha256": "...",
  "cli_sha256": "...",
  "cli_args": [],
  "analysis": {},
  "diagnostics": {}
}
```

Some blocks are optional in practice, but the top-level layout is stable enough for downstream tooling to key off these names.

## Important Nested Blocks

### `calibration`

Typical fields include:

- `coeffs`
- `covariance`
- `sigma_e`
- `sigma_e_error`
- `peaks`

### `spectral_fit`

This block contains the fitted spectral parameters plus:

- `cov`
- `ndf`
- `likelihood_path`
- optional `peak_deviation`

`likelihood_path` records the fitted spectral route, such as the default `binned_poisson` path or staged template-fit variants.

### `time_fit`

`time_fit` is a mapping keyed by isotope, for example `Po214` and `Po218`. Each isotope block typically contains:

- fitted parameters such as `E_<iso>`, `dE_<iso>`, `N0_<iso>`, `B_<iso>`
- `fit_valid`
- `cov`
- `ndf`
- `background_mode`
- optional `baseline_rate_bq` and `baseline_unc_bq`

The time-series fitter still fits only Po-214 and Po-218. When `window_po210` is configured, Po-210 is histogrammed and plotted for diagnostics but does not get a decay fit.

### `baseline`

Typical baseline fields include:

- `analysis_counts`
- `rate_bq`
- `rate_unc_bq`
- `scales`
- `dilution_factor`
- `corrected_rate_bq`
- `corrected_sigma_bq`
- `corrected_activity`

See [baseline.md](baseline.md) for the subtraction model.

### `radon_results`

This block stores the scalar radon outputs used for reporting:

- `radon_activity_bq`
- `radon_concentration_bq_per_l`
- `total_radon_in_sample_bq`
- optionally `radon_delta_bq`

### `radon`

The `radon` block stores the combined activity estimate and the plotting payloads:

- `isotope_mode`
- `rn_activity_bq`
- `stat_unc_bq`
- `components`
- `time_series`
- `total_time_series`
- `plot_series`

### `radon_inference`

When enabled, this block typically includes:

- `rn_inferred`
- `ambient_rn`
- `volume_equiv`
- `volume_cumulative`
- `meta`

See [radon_inference.md](radon_inference.md) for the meaning of those arrays and the leak-rate model.

### `diagnostics`

This block summarizes technical fit validity and warnings:

- `spectral_fit_fit_valid`
- `time_fit_po214_fit_valid`
- `time_fit_po218_fit_valid`
- `n_events_loaded`
- `n_events_discarded`
- `selected_analysis_modes`
- `warnings`

## Time-Series JSON Exports

When `dump_time_series_json` or `--dump-ts-json` is enabled, each `time_series_<iso>.png` can be accompanied by `time_series_<iso>_ts.json`.

Those files include:

- UTC bin edges and centers
- Unix-second bin edges and centers
- elapsed-hour bin centers
- bin widths in seconds
- flattened per-isotope counts
- per-bin live times
- per-bin efficiencies

## Activity Semantics

The time-series model multiplies the decay rate by the detection efficiency internally. As a result:

- `E_Po214` and `E_Po218` correspond to physical decay rates in Bq before any volume correction
- you should not divide those fitted rates by efficiency again
- `compute_radon_activity` expects those fitted rates directly

If you want a volumetric concentration, apply the appropriate volume conversion at the reporting stage.

## Invalid Fits

If a fit completed numerically but failed the covariance validity check, the result still appears in `summary.json` with `fit_valid: false`. Plotting code omits fit overlays in that case, so consumers should treat `fit_valid` as the technical gate for whether the overlayed model was trusted.

## Template Fit Diagnostics

When `time_fit.extraction_method: template` is enabled, `summary.json["template_fitting"]` includes aggregate per-bin fit-quality metrics for the template time-series stage. In addition to chi2 and centroid-shift statistics, it now reports:

- `bins_with_any_bound_hits`: number of fitted time bins with at least one free parameter ending at an active optimizer bound
- `total_bound_hits`: total number of per-bin bound hits across all fitted bins
- `bound_hit_param_counts`: per-parameter count of how often a bound was hit
- `suspicious_bound_hit_param_counts`: bound-hit counts after excluding benign lower-floor hits on optional nuisance terms
- `expected_floor_bound_hit_param_counts`: lower-bound hits that simply indicate an optional nuisance component went to zero
- `bound_hit_side_counts`: aggregate counts of `lower`, `upper`, and `both` bound hits
- `upper_bound_hit_param_counts`, `lower_bound_hit_param_counts`: per-parameter counts split by hit side
- `bins_with_non_centroid_bound_hits`: number of bins where a non-`mu_*` parameter hit a bound
- `non_centroid_bound_hit_param_counts`: per-parameter counts restricted to non-centroid parameters
- `bins_with_suspicious_bound_hits`: bins that still have diagnostically useful bound hits after the benign floor-hit split
- `bins_with_lower_bound_hits`, `bins_with_upper_bound_hits`, `bins_with_dual_sided_bound_hits`: bin-level rollups for the side-specific bound hits

The companion `template_per_bin_diagnostics.json` file carries the per-bin details through `n_bound_hits`, `bound_hit_params`, and `bound_hits`.
