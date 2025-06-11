# Radon Monitor Analysis Pipeline

This repository provides a complete pipeline to analyze electrostatic radon monitor data.

## Structure

- `analyze.py`: Main entry point to run the full analysis.
- `config.json`: JSON configuration file containing thresholds and options.
- `io_utils.py`: Functions to load raw data and write outputs.
- `calibration.py`: Peak-finding and energy calibration routines.
- `fitting.py`: Unbinned likelihood fit for Po-214 (and optional Po-218).
- `efficiency.py`: Efficiency calculations and BLUE combination helpers.
- `systematics.py`: Scan for systematic uncertainties (optional).
- `plot_utils.py`: Plotting routines for spectrum and time-series.
- `utils.py`: Miscellaneous utilities (time conversion, JSON validation,
  count-rate conversions).
- `tests/`: `pytest` unit tests for calibration, fitting, and I/O.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python analyze.py --config config.json --input merged_data.csv \
    [--output_dir results] [--job-id MYRUN] \
    [--efficiency-json eff.json] [--systematics-json syst.json] \
    [--spike-count N --spike-count-err S] [--slope RATE] \
    [--analysis-end-time ISO --spike-end-time ISO] \
    [--spike-period START END] [--run-period START END] \
    [--radon-interval START END] \
    [--hl-po214 SEC] [--hl-po218 SEC] \
    [--settle-s SEC] [--debug] [--seed SEED] \
    [--ambient-file amb.txt (time conc)] [--ambient-concentration 0.1] \
    [--burst-mode rate] \
    [--time-bin-mode fixed --time-bin-width 3600] [--dump-ts-json]
```

## Input CSV Format

The input file must be a comma-separated table with these columns:

- `fUniqueID` – unique event number
- `fBits` – status bits or flags
- `timestamp` – event timestamp in seconds
- `adc` – raw ADC value
- `fchannel` – acquisition channel

Columns beyond these are ignored. See `example_input.csv` for a
sample layout.

## Output

The analysis writes results to `<output_dir>/<timestamp>/` by default. When `--job-id` is given the folder `<output_dir>/<job-id>/` is used instead. The directory includes:

- `summary.json` – calibration and fit summary.
- `config_used.json` – copy of the configuration used.
- `spectrum.png` – spectrum plot with fitted peaks.
- `time_series_Po214.png` and `time_series_Po218.png` – decay time-series plots.
- Optional `*_ts.json` files containing binned time series when enabled.
- `efficiency.png` – bar chart of individual efficiencies and the BLUE result.
- `eff_cov.png` – heatmap of the efficiency covariance matrix.
- `radon_activity.png` – extrapolated radon activity over time.
 - `equivalent_air.png` – equivalent air volume plot when `--ambient-file` or
   `--ambient-concentration` is provided.

The `time_fit` routine currently fits only Po‑214 and Po‑218.  Supporting
Po‑210 would require adding its half‑life and detection efficiency to the
configuration along with a plotting color in the code.

The time‐series model multiplies the decay rate by the detection efficiency
internally.  Therefore the fitted `E_Po214` and `E_Po218` values correspond to
the physical decay rates in Bq (decays/s) before any detector volume correction.
Do **not** divide these results by the efficiency again.  To obtain the
concentration simply convert the fitted rate to Bq/m³ using `utils.cps_to_bq`.
For example:

```python
from utils import cps_to_bq
activity_bq_m3 = cps_to_bq(fit_result["E_Po214"], volume_liters=10.0)
```

When using ``compute_radon_activity`` you should pass the fitted rates
directly. They already represent activities in Bq and no additional
division by the detection efficiency is required.

The time-series fit checks whether the covariance matrix returned by
Minuit is positive definite.  If not, a tiny diagonal jitter is added
before repeating the check.  When even the jittered matrix fails this
test the result still contains the fitted values but ``fit_valid`` is set
to ``False``.

## Configuration

`nominal_adc` under the `calibration` section sets the expected raw ADC
centroids for Po‑210, Po‑218 and Po‑214 when using automatic calibration.
If omitted, defaults of `{"Po210": 1250, "Po218": 1400, "Po214": 1800}`
are used.

`sanity_tolerance_mev` in the same section specifies how close the fitted
peak energies must be to their known values.  The default of `0.5` MeV
causes calibration to fail when any Po‑210, Po‑218 or Po‑214 centroid
deviates by more than this amount.

`noise_cutoff` sets the minimum ADC value retained when loading the event
CSV.  Events below this threshold are removed before burst filtering.
The template configuration uses `400`.

`analysis_start_time` in the optional `analysis` section sets the global
time origin for decay fitting and time-series plots.  Provide an
ISO‑8601 string such as `"2020-01-01T00:00:00Z"`.  When omitted the first
event timestamp is used.

All other time-related fields (`analysis_end_time`, `spike_end_time`,
`spike_periods`, `run_periods`, `radon_interval` and
`baseline.range`) likewise accept absolute timestamps in ISO 8601
format.

`analysis_end_time` may be specified to stop processing after the given
timestamp while `spike_end_time` discards all events before its value.
`spike_periods` holds a list of `[start, end]` pairs where events are
excluded entirely.  All of these accept ISO‑8601 strings and can also be
set with the corresponding CLI options.
`run_periods` specifies the intervals of valid data to keep after spike
filtering.  Events falling outside all provided periods are discarded.
`radon_interval` sets two timestamps used to compute the change in radon
activity between them.

`ambient_concentration` may also be specified here to record the ambient
radon concentration in Bq/m³ used for the equivalent air plot.  The
command-line option `--ambient-concentration` overrides this value.  The
default configuration sets this key to `null`.  The template
`config.json` therefore includes
```json
"ambient_concentration": null
```
under the `analysis` section.

Example snippet:

```json
"analysis": {
    "analysis_start_time": "2020-01-01T00:00:00Z",
    "analysis_end_time": "2020-01-02T00:00:00Z",
    "spike_end_time": "2020-01-01T01:00:00Z",
    "spike_periods": [["2020-01-01T03:00:00Z", "2020-01-01T04:00:00Z"]],
    "run_periods": [["2020-01-01T02:00:00Z", "2020-01-01T06:00:00Z"]],
    "radon_interval": ["2020-01-01T02:00:00Z", "2020-01-01T06:00:00Z"],
    "ambient_concentration": 0.02
}
```

When present the value is also written to `summary.json` under the
`analysis` section:

```json
"analysis": {
    "analysis_start_time": "2020-01-01T00:00:00Z",
    "analysis_end_time": "2020-01-02T00:00:00Z",
    "spike_end_time": "2020-01-01T01:00:00Z",
    "spike_periods": [["2020-01-01T03:00:00Z", "2020-01-01T04:00:00Z"]],
    "run_periods": [["2020-01-01T02:00:00Z", "2020-01-01T06:00:00Z"]],
    "radon_interval": ["2020-01-01T02:00:00Z", "2020-01-01T06:00:00Z"],
    "ambient_concentration": 0.02
}
```

`--ambient-file` may be supplied instead to account for a
time-dependent background.  The option expects a two-column text file
containing absolute timestamps (in seconds) and the corresponding
ambient concentration in Bq/L.  These values are linearly interpolated
to the radon-activity timestamps and override any constant value when
calling `plot_equivalent_air`.

`burst_filter` controls removal of short high-rate clusters.  The
`burst_mode` key selects the default strategy which can be overridden by
the command-line option `--burst-mode`.  `none` disables the filter,
`micro` applies a short sliding-window veto defined by
`micro_window_size_s` and `micro_count_threshold`, `rate` uses the
rolling-median threshold (`burst_window_size_s`, `rolling_median_window`,
`burst_multiplier`) and `both` applies the micro filter followed by the
rate veto.

Example snippet:

```json
"burst_filter": {
    "burst_mode": "rate",
    "burst_window_size_s": 60,
    "rolling_median_window": 5,
    "burst_multiplier": 5,
    "micro_window_size_s": 1,
    "micro_count_threshold": 3
}
```

`time_bins_fallback` under the `plotting` section sets the number of
histogram bins to use when the automatic Freedman&ndash;Diaconis rule
fails, typically due to zero IQR.  The default is `1`.

The CLI options `--time-bin-mode` and `--time-bin-width` override
`plot_time_binning_mode` and `plot_time_bin_width_s` in the configuration
to control the time-series histogram. Passing `--dump-ts-json` writes the
histogram counts to a `*_ts.json` file alongside the plot.

Additional convenience flags include `--spike-count` (with optional
`--spike-count-err`) to override spike efficiency inputs, `--slope` to
apply a linear ADC drift correction, `--analysis-end-time` and
`--spike-end-time` to clip the dataset, one or more `--spike-period`
options to exclude specific time windows, `--settle-s` to skip the
initial settling period in the decay fit, `--seed` to set the random
seed used by the analysis and `--debug` to increase log verbosity.
The half-lives used in the decay fit can also be changed with
`--hl-po214` and `--hl-po218`.

When the spectrum is binned in raw ADC channels (`"spectral_binning_mode": "adc"`),
the bin edges are internally converted to energy using the calibration
slope and intercept before plotting.  This ensures `spectrum.png`
reflects the calibrated energy scale regardless of binning mode.

The `spectral_fit` section provides priors for the unbinned likelihood
fit.  Important keys include:

- `fd_hist_bins` – number of histogram bins to use when the automatic
  Freedman–Diaconis rule fails.
- `mu_sigma` – uncertainty applied to peak centroids.
- `amp_prior_scale` – scales the width of the peak amplitude priors.
- `bkg_mode` – `"auto"` estimates the linear continuum from the spectrum
  while `"manual"` uses the `b0_prior` / `b1_prior` values.
- `b0_prior` / `b1_prior` – `[mean, sigma]` priors for the linear
  continuum terms.
- `tau_{iso}_prior_mean` and `tau_{iso}_prior_sigma` – mean and
  uncertainty for the exponential tail constant of each isotope when
  `use_emg` enables that tail.  Use a strictly positive prior mean (e.g.
  ``0.005``) to prevent numerical overflow when the tail constant
  approaches zero.
- `mu_bounds` – optional lower/upper limits for each peak centroid.
  Set for example `{"Po218": [5.9, 6.2]}` to keep the Po‑218 fit from
  drifting into the Po‑210 region.  Centroid guesses found during peak
  search are clamped to this range before the fit starts.

- `expected_peaks` – approximate ADC centroids used to locate the
  Po‑210, Po‑218 and Po‑214 peaks before fitting. The default is
  `{"Po210": 1250, "Po218": 1400, "Po214": 1800}`.
- `emg_left` evaluations are wrapped in `np.errstate` and passed through
  `np.nan_to_num` for stability so that NaN or infinite values never
  reach `curve_fit`.

Example snippet:

```json
"spectral_fit": {
    "bkg_mode": "manual",
    "b0_prior": [0.0, 1.0],
    "b1_prior": [0.0, 1.0],
    "mu_sigma": 0.05,
    "amp_prior_scale": 1.0
}
```

`dump_time_series_json` under `plotting` saves a `*_ts.json` file
containing the binned time-series data when set to `true`.

`scan_keys` in the `systematics` section selects which fit parameters
are varied during the systematic uncertainty scan.  By default no
parameters are scanned.

`adc_drift_rate` under `systematics` applies a linear time-dependent
shift to the raw ADC values before calibration.  The value is in ADC
counts per second and defaults to `0.0` (no correction).  When the rate
is non-zero `analyze.py` applies the shift using
`apply_linear_adc_shift` and stores the value in `summary.json` under
`adc_drift_rate`.

`plot_time_style` chooses how the histogram is drawn in the time-series
plot.  Use `"steps"` (default) for a stepped histogram or `"lines"` to
connect bin centers with straight lines.  The line style is useful when
overlaying multiple isotopes so one does not obscure the other.

`plot_time_normalise_rate` divides the histogram by its bin width so the
vertical axis shows a rate in counts per second.  The model curve is
likewise drawn in rate units.  Set this to `true` when comparing runs with
different bin widths.

`overlay_isotopes` under `plotting` keeps both isotope windows intact
when invoking `plot_time_series`.  When set to `true` the analysis does
not clear the other window, allowing Po‑214 and Po‑218 to be plotted
together on a single overlay.

`plot_time_series` takes its half-life values from the `time_fit` section.
When these keys are omitted, `hl_Po214` and `hl_Po218` default to the radon half-life (~3.8 days or about `3.28e5` s). Specify them to use other values. These custom half-lives control the decay model drawn over the time-series histogram.
The same values are used in the `time_fit` routine itself, so changing
`hl_Po214` or `hl_Po218` affects both the unbinned fit and the overlay in
`plot_time_series`. For monitoring that spans multiple days you may set
them to the radon half-life (~3.8 days) to match the slowly varying
radon activity.

`sig_N0_Po214` and `sig_N0_Po218` set the uncertainty on the prior for the
initial activity `N0` when no baseline range is provided.  Without a baseline,
the fit applies a Gaussian prior `(0, sig_N0_{iso})` so `N0` may vary rather
than being fixed to zero.  The default width is `1.0` if not specified in the
configuration.


`settling_time_s` was removed from the `time_fit` section and is no
longer needed. The CLI option `--settle-s` may be used instead to
discard the first seconds of data before the decay fit.

### Fitting Long Time Scales

When the data covers months or more, the short half-lives of Po‑218 and
Po‑214 no longer matter.  The defaults therefore set `hl_Po214` and
`hl_Po218` to the radon half-life (≈3.8 days) so the fit tracks the slowly
varying radon concentration.  The configuration values are in seconds;
3.8 days corresponds to roughly `3.8 * 86400 ≈ 3.3e5` seconds.

Example snippet:

```json
"time_fit": {
    "hl_Po214": [328320, 0.0],
    "hl_Po218": [328320, 0.0],
    "sig_N0_Po214": 1.0,
    "sig_N0_Po218": 1.0
}
```

These half-life values may also be set on the command line with
`--hl-po214` and `--hl-po218`.

### Baseline Runs

A baseline run measures the radon background with an empty monitor before
an assay. Configuration must define three keys under `baseline`:

- `baseline.range` – list of two ISO‑8601 timestamps selecting the baseline interval.
- `monitor_volume_l` – internal volume of the radon monitor in liters.
- `sample_volume_l` – volume of the assay sample in liters.

Events collected during the baseline period are counted in the Po‑214 and
Po‑218 windows. The counts are converted directly into a decay rate in
Bq by dividing by the baseline live time and detection efficiency.  This
rate is scaled by the dilution factor
`monitor_volume_l / (monitor_volume_l + sample_volume_l)` before being
subtracted from the fitted radon decay rate of the assay.

Example snippet:

```json
"baseline": {
    "range": ["2023-07-01T00:00:00Z", "2023-07-03T00:00:00Z"],
    "monitor_volume_l": 10.0,
    "sample_volume_l": 5.0
}
```

Command line usage:

```bash
python analyze.py --config assay.json --input run.csv --output_dir results \
    --baseline_range 2023-07-01T00:00:00Z 2023-07-03T00:00:00Z
```

## Utility Conversions

`utils.py` provides simple helpers to convert count rates:

- `cps_to_cpd(rate_cps)` converts counts/s to counts/day.
- `cps_to_bq(rate_cps, volume_liters=None)` returns the activity in Bq, or
  Bq/m^3 when a detector volume is supplied.

You can invoke these from the command line:

```bash
python utils.py 0.5 --to cpd
python utils.py 0.5 --to bq --volume_liters 10
```

## Radon Activity Output

After the decay fits a weighted average of the Po‑218 and Po‑214 rates is
converted to an instantaneous radon activity.  The result is written to
`summary.json` under `radon_results` together with the corresponding
concentration (per liter) and the total amount of radon contained in the
sample volume.  The file `radon_activity.png` visualises this
activity versus time.  When either `--ambient-file` or
`--ambient-concentration` is supplied an additional plot
`equivalent_air.png` shows the volume of ambient air containing the same
activity.
The Po‑214 activity alone is plotted in `radon_activity_po214.png`. When
ambient concentration data are available, `equivalent_air_po214.png`
shows the equivalent air volume derived from this Po‑214 activity.

## Efficiency Calculations

`efficiency.py` implements helpers to derive efficiencies from spike,
assay or decay data and combines multiple estimates using the BLUE
method.  Each entry may be a single dictionary or a list of dictionaries
which will be combined.  When the configuration file provides an
`efficiency` section with entries such as:

```json
"efficiency": {
    "spike": [
        {"counts": 1000, "activity_bq": 50, "live_time_s": 3600}
    ],
    "assay": [
        {"rate_cps": 0.8, "reference_bq": 2.0}
    ]
}
```

`analyze.py` stores the calculated values and their BLUE combination in
`summary.json` under the `efficiency` key.

The option `--efficiency-json PATH` may be supplied on the command line to
load the efficiency section from a separate file instead of embedding it
directly in the main configuration.  Similarly `--systematics-json PATH`
overrides the `systematics` section.


## Running Tests

Use the provided setup script to install the required packages:
```bash
scripts/setup_tests.sh
pytest -v
```

## Local Test Setup

When developing locally, run the setup script and then execute `pytest`:

```bash
scripts/setup_tests.sh
pytest -v
```

