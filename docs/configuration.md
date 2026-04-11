# Configuration Reference

This page collects the configuration details that used to live in the top-level `README`. It focuses on the YAML schema, the most important keys, and where each section feeds the pipeline.

For CLI overrides, see [cli.md](cli.md). For baseline-specific behavior, see [baseline.md](baseline.md). For time parsing and unit conventions, see [time-and-units.md](time-and-units.md).

## Loading Rules

- Configuration keys are case sensitive; lowercase keys are the supported form.
- Mixed-case legacy keys are still accepted in some places for backward compatibility, but they are deprecated.
- If `--config` is not supplied, `analyze.py` looks for `config.yaml` next to the script.
- CLI arguments override the corresponding configuration values.
- Duplicate YAML keys are rejected during loading.
- Most sections are validated strictly by schema. The `time_fit` section intentionally allows extra keys so older standalone fitter configurations still load.

## Minimal Skeleton

The required top-level sections are `pipeline`, `spectral_fit`, `time_fit`, `systematics`, and `plotting`.

```yaml
pipeline:
  log_level: INFO
  random_seed: 67

analysis:
  analysis_start_time: null
  analysis_end_time: null

columns:
  timestamp: timestamp
  adc: adc

calibration:
  noise_cutoff: 400

spectral_fit:
  do_spectral_fit: true
  clip_floor: 1e-300

time_fit:
  do_time_fit: true

systematics:
  enable: false

plotting:
  plot_save_formats: [png]
```

## Top-Level Toggles

- `allow_fallback`: enable fallback behavior in parts of the pipeline that support it
- `allow_negative_baseline`: preserve negative baseline-corrected rates instead of clipping them to `0.0`
- `allow_negative_activity`: preserve negative total radon values instead of clamping them to `0.0`
- `analysis_isotope`: one of `radon`, `po214`, or `po218`

`analysis_isotope: radon` is the default and combines Po-214 and Po-218 with inverse-variance weighting. The single-isotope modes are mainly for diagnostics. See [analysis-modes.md](analysis-modes.md).

## `pipeline`

- `log_level`: standard Python logging level
- `random_seed`: random seed used by analysis algorithms

The chosen seed is written to `summary.json`, and `--reproduce` reuses the archived value.

## `analysis`

This section holds the run-level analysis window and global mode selection.

- `analysis_start_time`: global time origin for fits and plots
- `analysis_end_time`: stop processing after this time
- `spike_start_time`, `spike_end_time`: discard data after or before a single cutoff
- `spike_periods`: list of `[start, end]` intervals to exclude
- `run_periods`: list of `[start, end]` intervals to keep
- `radon_interval`: interval used when computing radon deltas
- `ambient_concentration`: constant ambient radon concentration in Bq/L for equivalent-air plots
- `background_model`: `linear` or `loglin_unit`
- `likelihood`: `current` or `extended`

All time values accept ISO-8601 strings or Unix epoch seconds and are normalized to UTC.

Example:

```yaml
analysis:
  analysis_start_time: "2023-07-31T00:00:00Z"
  analysis_end_time: "2024-02-01T06:00:00Z"
  spike_start_time: null
  spike_end_time: "2023-07-31T00:10:00Z"
  spike_periods:
    - ["2023-11-12T00:00:00Z", "2023-11-13T12:00:00Z"]
  run_periods:
    - ["2023-09-28T00:00:00Z", "2023-10-28T23:59:59Z"]
    - ["2024-01-05T00:00:00Z", "2024-01-10T23:59:59Z"]
  radon_interval: ["2024-01-05T06:00:00Z", "2024-01-06T18:00:00Z"]
  ambient_concentration: 0.02
```

## `columns`

Map the pipeline's canonical column names to the headers used in your input CSV.

```yaml
columns:
  timestamp: ftimestamps
  adc: fadc_channels
```

At minimum you need mappings for `timestamp` and `adc`.

## `baseline`

This section enables baseline subtraction and defines the monitor and sample volumes used in the dilution correction.

- `range`: baseline interval as `[start, end]`
- `monitor_volume_l`: monitor volume in liters
- `sample_volume_l`: sample volume in liters
- `isotopes_to_subtract`: background components to subtract

If `isotopes_to_subtract` is omitted, `analyze.py` defaults to `["Po214", "Po218"]`. Some workflows also include `noise`, which is derived from baseline pedestal statistics and handled separately from the isotope rates.

Baseline behavior, output fields, and the subtraction model are documented in [baseline.md](baseline.md).

## `burst_filter`

This section controls burst rejection.

- `burst_mode`: `none`, `micro`, `rate`, or `both`
- `micro_window_size_s`, `micro_count_threshold`: parameters for the short micro-burst veto
- `burst_window_size_s`, `rolling_median_window`, `burst_multiplier`: parameters for the rolling-median rate veto

Example:

```yaml
burst_filter:
  burst_mode: rate
  burst_window_size_s: 60
  rolling_median_window: 5
  burst_multiplier: 5
  micro_window_size_s: 1
  micro_count_threshold: 3
```

The optional `--burst-sensitivity-scan` CLI mode sweeps a small grid around the current settings and writes `burst_scan.png`.

## `calibration`

This section controls the ADC-to-energy conversion and the pre-fit event filtering.

- `method`: calibration method, typically `two-point`
- `noise_cutoff`: pedestal threshold in ADC; `null` disables the cut
- `hist_bins`, `peak_search_radius`, `peak_prominence`, `peak_width`
- `peak_widths`: per-isotope overrides for `peak_width`
- `slope_mev_per_ch`: fixed linear slope
- `float_slope`: when `true`, treat the supplied slope as an initial guess and refine it in the two-point fit
- `intercept_mev`: optional intercept that bypasses the Po-214 search when combined with a fixed slope
- `nominal_adc`: expected raw ADC centroids for Po-210, Po-218, and Po-214
- `known_energies`: expected energies in MeV
- `fit_window_adc`, `fit_retry_window`, `fit_maxfev`, `curve_fit_max_evaluations`
- `sigma_e_init`: optional initial guess for the common energy resolution in MeV, either a single value or a per-isotope map
- `use_emg`, `init_sigma_adc`, `init_tau_adc`, `tau_bounds_adc`
- `sanity_tolerance_mev`: fail calibration if a centroid drifts too far from the known energy

Important details:

- `nominal_adc` defaults to `{"Po210": 1250, "Po218": 1400, "Po214": 1800}` when omitted.
- `curve_fit_max_evaluations` defaults to `10000`.
- `noise_cutoff` is applied immediately after the CSV is loaded, and the removed-event count is recorded under `noise_cut.removed_events` in `summary.json`.
- `--calibration-slope` overrides `calibration.slope_mev_per_ch`.

Examples:

```yaml
calibration:
  sigma_e_init: 0.015
```

```yaml
calibration:
  peak_width: 5
  peak_widths:
    Po214: 6
```

```yaml
calibration:
  slope_mev_per_ch: 0.00430
  float_slope: true
  # intercept_mev: -0.12
```

## `spectral_fit`

This section configures the spectral model and its priors.

- `do_spectral_fit`: enable or disable the spectral fit
- `spectral_binning_mode`: `energy` or `adc`
- `energy_bin_width`: fixed energy-bin width in MeV when `spectral_binning_mode: energy`
- `fd_hist_bins`: fallback number of bins when the Freedman-Diaconis rule fails
- `fit_energy_range`: analysis window in MeV
- `mu_sigma`: prior width on peak centroids
- `amp_prior_scale`: scale factor for amplitude priors
- `bkg_mode`: `auto` estimates the linear continuum; `manual` uses explicit priors
- `b0_prior`, `b1_prior`: `[mean, sigma]` priors for the continuum terms
- `max_b1_sigma`: optional clamp for `b1_prior` sigma; defaults to `10.0`
- `clip_floor`: small positive floor applied to the likelihood PDF values; must lie in `(0, 1e-6]`
- `background_norm_points`: number of integration samples for the unit-area log-linear background; defaults to `512`
- `s_bkg_prior`: prior for the unit-area log-linear background normalization
- `sigma_e_prior_source`: either `calibration` or an explicit numeric prior tuple
- `float_sigma_e`: allow the common energy resolution to float against its prior
- `expected_peaks`: approximate peak centroids used for the initial search
- `peak_search_method`: `prominence` or `cwt`
- `peak_search_prominence`, `peak_search_width_adc`, `peak_search_cwt_widths`
- `mu_bounds`: per-isotope centroid bounds
- `mu_bounds_units`: `mev` or `adc`
- `unbinned_likelihood`: when `true`, use the unbinned path instead of the default binned Poisson path
- `use_emg`, `tau_<iso>_prior_mean`, `tau_<iso>_prior_sigma`: per-isotope EMG tail configuration
- `f_beta_<iso>_prior`, `lambda_beta_<iso>_prior`: per-isotope beta-coincidence tail priors
- `beta_high_side_only`: per-isotope toggle that clips the beta-coincidence contribution to `E >= mu`
- `share_beta`: replace per-isotope beta-coincidence parameters with `f_beta_shared` and `lambda_beta_shared`
- `f_beta_shared_prior`, `lambda_beta_shared_prior`: priors used only when `share_beta: true`

Important details:

- The default spectral path is now binned Poisson. `summary.json` records the chosen path in `spectral_fit.likelihood_path`.
- `clip_floor` is applied only to the per-energy density used in the likelihood. The extended integral uses the unclipped model.
- A `tau_<iso>` prior automatically enables the corresponding EMG tail even if `use_emg` omits it.
- `mu_bounds_units: adc` is converted to MeV before fitting so the fit itself remains on a MeV scale.
- `emg_left` evaluations are wrapped for numerical stability so NaN or infinite values do not reach `curve_fit`.
- `beta_high_side_only` is useful for coincidence isotopes such as Po-214 and Po-212 when the fitted beta tail should stay entirely on the high-energy side of the alpha peak.
- `share_beta: true` is available for simpler scans, but the default config keeps per-isotope beta tails because Po-212 often needs a visibly stronger high-energy tail than Po-214.

Example:

```yaml
spectral_fit:
  bkg_mode: manual
  b0_prior: [0.0, 1.0]
  b1_prior: [0.0, 1.0]
  mu_sigma: 0.05
  amp_prior_scale: 1.0
  use_emg:
    Po210: true
  tau_po210_prior_mean: 0.005
  tau_po210_prior_sigma: 0.002
  unbinned_likelihood: false
```

For the full EMG precedence rules and legacy compatibility notes, see [emg_config.md](emg_config.md).

## `fitting`

This optional section controls the EMG backend used by the fitter.

- `use_stable_emg`: enable the numerically stable EMG implementation
- `emg_stable_mode`: stable backend selection
- `emg_tau_min`: lower floor for the EMG tail constant

Example:

```yaml
fitting:
  use_stable_emg: true
  emg_stable_mode: scipy_safe
  emg_tau_min: 1.0e-5
```

## `time_fit`

This section controls the time-series decay fits.

- `do_time_fit`: enable or disable time fitting
- `window_po214`, `window_po218`, `window_po210`: energy windows in MeV
- `eff_po214`, `eff_po218`, `eff_po210`: efficiencies used by the time fits
- `hl_po214`, `hl_po218`, `hl_po210`: half-lives in seconds
- `bkg_po214`, `bkg_po218`: background priors
- `sig_n0_po214`, `sig_n0_po218`: priors on the initial activity `N0` when no baseline range is supplied
- `background_guess`, `n0_guess_fraction`
- `fix_background_b_first_pass`: enable the first fixed-background pass before the free-background refit
- `background_b_fixed_value`: explicit value for the first pass; if omitted, the baseline Po-214 rate is used
- `extraction_method`: `roi` or `template`
- `template_rebin`, `template_min_counts`: per-bin template-fit histogram controls
- `float_centroids`: allow per-bin spectral centroids to move around the aggregate fit
- `centroid_shift_bound_kev`: initial half-width for per-bin centroid movement
- `centroid_shift_prior_sigma_kev`: Gaussian penalty width for centroid motion; `null` defaults to half of `centroid_shift_bound_kev`
- `centroid_shift_scan_auto_expand`: rerun a bin with wider centroid bounds when it lands on the limit
- `centroid_shift_max_expand_kev`: maximum half-width used by the auto-expansion retry; keep this modest to avoid reintroducing large centroid tails
- `fix_weak_isotopes`: legacy compatibility switch; when enabled it only fixes very weak auxiliary isotopes unless `template_fixed_isotopes` is set
- `template_fixed_isotopes`: explicit isotope list to amplitude-fix in per-bin template fits

The pipeline also accepts the simplified standalone-fitter schema:

```yaml
time_fit:
  model: single_exp
  fix_lambda: true
  lambda: 2.1e-6
  activity_units: cpd
```

Important details:

- The time fit still models only Po-214 and Po-218. When `window_po210` is defined, Po-210 is histogrammed for diagnostics but does not receive a decay fit.
- Custom half-lives affect both the fit and the overlaid model curves.
- For multi-day or multi-month monitoring, you can set `hl_po214` and `hl_po218` to the radon half-life so the fit tracks the slowly varying parent activity.
- `settling_time_s` was removed from the configuration; use `--settle-s` instead.

Long-timescale example:

```yaml
time_fit:
  hl_po214: 328320
  hl_po218: 328320
  hl_po210: 11923200
  sig_n0_po214: 1.0
  sig_n0_po218: 1.0
```

## `systematics`

This section controls optional systematic scans and ADC drift corrections.

- `enable`: enable the systematic study
- `sigma_e_frac`, `tail_fraction`: fractional shifts
- `energy_shift_kev`: absolute energy shift in keV; converted to MeV internally
- `adc_drift_rate`: linear drift in ADC counts per second
- `adc_drift_mode`: `linear`, `quadratic`, or `piecewise`
- `adc_drift_params`: additional parameters for non-linear drift modes

The chosen drift settings are written to `summary.json` under `adc_drift_rate`, `adc_drift_mode`, and `adc_drift_params`.

## `efficiency`

Efficiency inputs can come from spike, assay, or decay data. Each source may be a single mapping or a list of mappings that will be combined with BLUE.

Example:

```yaml
efficiency:
  spike:
    - counts: 1000
      activity_bq: 50
      live_time_s: 3600
      enabled: true
  assay:
    - rate_cps: 0.8
      reference_bq: 2.0
```

Each spike entry may include `enabled: false` to keep the record in the file while excluding it from the BLUE combination.

## `radon_inference`

This optional stage converts isotope time-series into inferred Rn-222 activity and leak-related volume estimates.

- `enabled`: required switch
- `source_isotopes`: non-empty list drawn from `Po214` and `Po218`
- `source_weights`: optional per-isotope weights
- `detection_efficiency`: required per-isotope efficiencies
- `transport_efficiency`, `retention_efficiency`
- `chain_correction`
- `external_rn`: ambient radon source, either `constant` or `file`
- `output.write_per_interval`, `output.write_cumulative`

Example:

```yaml
radon_inference:
  enabled: true
  source_isotopes: ["Po214", "Po218"]
  source_weights:
    Po214: 0.7
    Po218: 0.3
  detection_efficiency:
    Po214: 0.19
    Po218: 0.094
  transport_efficiency: 1.0
  retention_efficiency: 1.0
  external_rn:
    mode: file
    file_path: "ambient_radon.csv"
    interpolation: ffill
    allowed_skew_seconds: 300
    constant_bq_per_m3: 120.0
```

For formulas, interpolation behavior, and the radon-inference plots, see [radon_inference.md](radon_inference.md).

## `plotting`

This section controls saved formats, time-series binning, and presentation.

- `plot_save_formats`: list of output file extensions
- `plot_spectrum_binsize_adc`
- `plot_time_binning_mode`: `auto`, `fd`, or `fixed`
- `plot_time_bin_width_s`: fixed bin width in seconds when using `fixed`
- `time_bins_fallback`: fallback number of bins when the Freedman-Diaconis rule fails
- `dump_time_series_json`: write `*_ts.json` payloads alongside the plots
- `plot_time_style`: `steps` or `lines`
- `plot_time_show_models`
- `plot_marker_size`
- `plot_time_normalise_rate`: show counts per second instead of raw counts per bin
- `overlay_isotopes`: keep both isotope windows for a combined time-series overlay
- `save_individual_time_series`: when overlaying, also write `time_series_<iso>_individual.png`
- `fractional_residual_min_counts`: hide fractional-residual bins when both data and model are below this count scale
- `plot_spectrum_write_log_copy`: write the `_log.png` companion for spectrum plots
- `plot_template_bin_fits`: write the per-bin template spectrum fits into `template_bin_fits/`
- `plot_template_bin_fits_bad_only`: restrict per-bin template plots to invalid fits, bins that hit any active parameter bound, bins that hit the centroid limit, or bins above the chi2 threshold
- `plot_template_bin_fits_bad_chi2_ndf_min`: chi2/ndf threshold used by `plot_template_bin_fits_bad_only`
- `plot_template_bin_fits_log_scale`: keep the `_log.png` companion for template bin-fit plots
- `palette`: `default`, `colorblind`, or `grayscale`

Important details:

- `--plot-time-binning-mode` and `--plot-time-bin-width` override the matching keys here.
- `dump_time_series_json: true` writes per-bin counts, live times, efficiencies, and UTC bin edges next to the matching time-series plot.
- `overlay_isotopes: true` preserves both daughter windows in a shared plot; the pipeline can additionally save per-isotope plots if `save_individual_time_series` is enabled.
- `spectrum.png` now stacks the linear spectrum, log spectrum, raw residuals, log raw residuals, normalized residuals, and fractional residuals.
- `fractional_residual_min_counts` prevents the fractional residual panel from being dominated by bins in the extreme low-count tail.
