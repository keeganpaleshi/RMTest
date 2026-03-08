# Validation and Fit Quality

RMTest validates configuration, time windows, numerical safeguards, and fit quality before it writes its outputs. This page summarizes those checks and the user-visible outcomes.

## Configuration Validation

Configuration loading performs several layers of validation:

- duplicate YAML keys are rejected
- required top-level sections are enforced
- most sections reject unknown keys via JSON schema
- `time_fit` keeps `additionalProperties: true` for backward compatibility

Additional explicit checks include:

- `spectral_fit.clip_floor` must lie in `(0, 1e-6]`
- `spectral_fit.background_norm_points` must be a positive integer
- `spectral_fit.max_b1_sigma` must be positive
- `spectral_fit.b1_prior` must be `[mean, sigma]` with positive sigma
- `spectral_fit.b1_prior[1]` is clamped to `max_b1_sigma` with a warning when it exceeds the configured cap

## Baseline Window Validation

`validate_baseline_window` checks the baseline interval against the analysis window:

- `baseline.range` must contain two values
- the end must be after the start
- `baseline.start` must be earlier than `analysis.analysis_end_time` when an analysis end is set
- the baseline interval must not lie entirely before `analysis.analysis_start_time`

Example of an invalid configuration:

```yaml
baseline:
  range: ["2024-01-02T00:00:00Z", "2024-01-03T00:00:00Z"]
analysis:
  analysis_end_time: "2024-01-01T23:00:00Z"
```

This raises a `ValueError` before the run continues.

## Radon-Inference Validation

`validate_radon_inference` enforces:

- `radon_inference` must be a mapping
- `source_isotopes` must be a non-empty list drawn from `Po214` and `Po218`
- `source_weights` may not contain isotopes that are absent from `source_isotopes`
- `detection_efficiency` must provide a value for every listed source isotope
- `transport_efficiency` and `retention_efficiency` must lie in `(0, 1.5]` if provided
- `external_rn.mode: constant` requires a positive `constant_bq_per_m3`
- `external_rn.mode: file` requires `file_path`

## Empty-Selection Fail Fast

`analyze.py` exits with an error if filtering removes all events at any stage, including:

- noise cut
- burst filter
- time-window selection
- baseline subtraction

This prevents downstream fits from running on an empty selection.

## Covariance Validity and `fit_valid`

Both spectral and time-series fits check whether the returned covariance matrix is positive definite.

- If the matrix is valid, `fit_valid` remains `true`.
- If it is not positive definite, the fitter adds a tiny diagonal jitter and checks again.
- If the matrix still fails, the fitted values are retained but `fit_valid` is set to `false`.
- If `strict=True`, or if you pass `--strict-covariance`, the code raises immediately instead of continuing.

When `fit_valid` is `false`, plotting routines omit fit overlays and the result is reported without extrapolating beyond the data.

## Two-Pass Time-Fit Validation

The time-series fitter uses a two-pass strategy when `fix_background_b_first_pass` is enabled:

1. Fit with the background term fixed.
2. Refit with the background term free.
3. Keep the second fit only if it is valid and improves the Akaike Information Criterion by at least `0.5`.

If the second pass is invalid, the first pass is kept.

## Negative-Value Policies

Two separate negative-value guards exist:

- Total radon activity is clamped to `0.0` by default. Use `--allow-negative-activity` to preserve the raw negative value and continue.
- Baseline-corrected rates are clipped to `0.0` by default. Use `allow_negative_baseline: true` or `--allow-negative-baseline` to preserve them.

## Diagnostics in `summary.json`

Each run writes a `diagnostics` block containing:

- `spectral_fit_fit_valid`
- `time_fit_po214_fit_valid`
- `time_fit_po218_fit_valid`
- `n_events_loaded`
- `n_events_discarded`
- `selected_analysis_modes`
- `warnings`

This is the first place to check when a run technically completed but behaved unexpectedly.

## Local Validation and Tests

Run the test suite with:

```bash
pip install -r requirements.txt
pytest -v
```

For local setup you can also run:

```bash
scripts/setup_tests.sh
pytest -v
```

To verify the old `_seconds` helper is gone, search for the exact helper name instead of the generic `total_seconds()` call:

```bash
rg 'baseline\._seconds\(' -n
rg -P '\b_seconds\(' -n
```
