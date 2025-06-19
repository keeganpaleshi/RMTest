# Radon Monitor Analysis Pipeline

This repository provides a complete pipeline to analyze electrostatic radon monitor data.

**Note:** All time quantities are expressed in seconds and all energies are given in MeV throughout the documentation and code. Event timestamps are stored as UTC `datetime` objects throughout the pipeline. Command-line options that take timestamps (such as `--analysis-start-time`) accept either ISO‑8601 strings or Unix seconds and are parsed accordingly.

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

This project requires **Python 3.11** or newer.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python analyze.py --config config.json --input merged_data.csv \
    [--output_dir results] [--job-id MYRUN] [--overwrite] \
    [--efficiency-json eff.json] [--systematics-json syst.json] \
    [--spike-count N --spike-count-err S] [--slope RATE] \
    [--noise-cutoff N] \
    [--analysis-start-time ISO --analysis-end-time ISO --spike-end-time ISO] \
    [--spike-period START END] [--run-period START END] \
    [--radon-interval START END] \
    [--hl-po214 SEC] [--hl-po218 SEC] \
    [--settle-s SEC] [--debug] [--seed SEED] \
    [--ambient-file amb.txt (time conc)] [--ambient-concentration 0.1] \
    [--burst-mode rate] \
    [--plot-time-binning-mode fixed --plot-time-bin-width 3600] [--dump-time-series-json] \
    [--hierarchical-summary OUT.json]
```

The script exits with an error message if filtering removes all events at any stage
(noise cut, burst filter, time-window selection or baseline subtraction).

## Input CSV Format

The input file must be a comma-separated table with these columns:

- `fUniqueID` – unique event number
- `fBits` – status bits or flags
- `timestamp` – event timestamp in seconds
- `adc` – raw ADC value
- `fchannel` – acquisition channel

Columns beyond these are ignored. See `example_input.csv` for a
sample layout which also includes typical auxiliary fields such as
`baseline_adc`, `spike_flag`, `valid`, `temperature`, `run_id`,
`pressure` and `humidity`.

When your CSV uses different header names you can specify them under
the `columns` section of the configuration.  Provide a mapping from the
canonical names (`timestamp`, `adc`, etc.) to the actual column names in
the file:

```json
"columns": {
    "timestamp": "ftimestamps",
    "adc": "fadc_channels"
}
```

## Output

The analysis writes results to `<output_dir>/<timestamp>/` by default. When `--job-id` is given the folder `<output_dir>/<job-id>/` is used instead. If the folder already exists run with `--overwrite` to replace it. The directory includes:

- `summary.json` – calibration and fit summary.
- `config_used.json` – copy of the configuration used.
  Any timestamps overridden on the command line are written
  back to this file in Unix seconds.
- `spectrum.png` – spectrum plot with fitted peaks.
- `time_series_Po214.png` and `time_series_Po218.png` – decay time-series plots.
- `time_series_Po210.png` when `window_po210` is set.
- Optional `*_ts.json` files containing binned time series when enabled.
- `efficiency.png` – bar chart of individual efficiencies and the BLUE result.
- `eff_cov.png` – heatmap of the efficiency covariance matrix.
- `radon_activity.png` – extrapolated radon activity over time.
 - `equivalent_air.png` – equivalent air volume plot when `--ambient-file` or
   `--ambient-concentration` is provided.

The `time_fit` routine still fits only Po‑214 and Po‑218.
When `window_po210` is provided the Po‑210 events are extracted and a
time‑series histogram is produced without a decay fit. The `hl_po210`
value controls only the model curve drawn in this plot.

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
to ``False``.  Passing ``strict=True`` (or ``--strict-covariance`` on the
command line) instead raises a ``RuntimeError`` as soon as the matrix is
found to be non-positive definite.

## Configuration

The parser is case sensitive, so all keys in `config.json` should be lowercase. Mixed-case names from older files remain supported for backward compatibility but are deprecated.

`nominal_adc` under the `calibration` section sets the expected raw ADC
centroids for Po‑210, Po‑218 and Po‑214 when using automatic calibration.
If omitted, defaults of `{"Po210": 1250, "Po218": 1400, "Po214": 1800}`
are used.

`sanity_tolerance_mev` in the same section specifies how close the fitted
peak energies must be to their known values.  The default of `0.5` MeV
causes calibration to fail when any Po‑210, Po‑218 or Po‑214 centroid
deviates by more than this amount.

`noise_cutoff` defines a pedestal noise threshold in ADC.  Events with raw
ADC values at or below this threshold are removed before any fits.  The
default is `400`.  Set it to `null` to skip the cut entirely.  The
`analyze.py` pipeline applies this filter right after loading the event
CSV.

The command-line option `--noise-cutoff` overrides this value when
provided. When you supply the option, its argument entirely replaces the
value from the configuration file.

Example snippet:

```json
"calibration": {
    "noise_cutoff": 400
}
```

To disable the cut:

```json
"calibration": {
    "noise_cutoff": null
}
```

When the cut is applied the analysis logs how many events were removed. This
count also appears in `summary.json` under `noise_cut.removed_events`.

`analysis_start_time` in the optional `analysis` section sets the global
time origin for decay fitting and time-series plots.  Provide an
ISO‑8601 string such as `"2023-07-31T00:00:00Z"`.  When omitted the first
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
radon concentration in Bq/L used for the equivalent air plot.  The
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
    "analysis_start_time": "2023-07-31T00:00:00Z",
    "analysis_end_time": "2024-02-01T06:00:00Z",
    "spike_end_time": "2023-07-31T00:10:00Z",
    "spike_periods": [["2023-11-12T00:00:00Z", "2023-11-13T12:00:00Z"]],
    "run_periods": [["2023-09-28T00:00:00Z", "2023-10-28T23:59:59Z"], ["2024-01-05T00:00:00Z", "2024-01-10T23:59:59Z"]],
    "radon_interval": ["2024-01-05T06:00:00Z", "2024-01-06T18:00:00Z"],
    "ambient_concentration": 0.02
}
```

When present the value is also written to `summary.json` under the
`analysis` section:

```json
"analysis": {
    "analysis_start_time": "2023-07-31T00:00:00Z",
    "analysis_end_time": "2024-02-01T06:00:00Z",
    "spike_end_time": "2023-07-31T00:10:00Z",
    "spike_periods": [["2023-11-12T00:00:00Z", "2023-11-13T12:00:00Z"]],
    "run_periods": [["2023-09-28T00:00:00Z", "2023-10-28T23:59:59Z"], ["2024-01-05T00:00:00Z", "2024-01-10T23:59:59Z"]],
    "radon_interval": ["2024-01-05T06:00:00Z", "2024-01-06T18:00:00Z"],
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

`time_bins_fallback` under the `plotting` section sets the number of histogram
bins to use when the automatic [Freedman–Diaconis rule](https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule) fails,
typically due to zero IQR.  The default is `1`.

The CLI options `--plot-time-binning-mode` (deprecated alias
`--time-bin-mode`) and `--plot-time-bin-width` override
`plot_time_binning_mode` and `plot_time_bin_width_s` in the configuration
to control the time-series histogram. Passing `--dump-time-series-json`
(alias `--dump-ts-json`) writes the histogram counts to a `*_ts.json`
file alongside the plot.

Additional convenience flags include `--spike-count` (with optional
`--spike-count-err`) to override spike efficiency inputs, `--slope` to
apply a linear ADC drift correction, `--analysis-start-time`, `--analysis-end-time` and
`--spike-end-time` to clip the dataset, one or more `--spike-period`
options to exclude specific time windows, `--settle-s` to skip the
initial settling period in the decay fit, `--seed` to set the random
seed used by the analysis, `--hierarchical-summary PATH` to produce a
Bayesian combination across runs and `--debug` to increase log verbosity.
The half-lives used in the decay fit can also be changed with
`--hl-po214` and `--hl-po218`.

When the spectrum is binned in raw ADC channels (`"spectral_binning_mode": "adc"`),
the bin edges are internally converted to energy using the calibration
`slope_MeV_per_ch` (MeV per channel) and intercept before plotting.  This ensures `spectrum.png`
reflects the calibrated energy scale regardless of binning mode.

Custom `bin_edges` arrays may be supplied when calling the spectral fitting or
plotting routines. The edges can have variable widths but must be strictly
increasing.

The `spectral_fit` section provides priors for the unbinned likelihood
fit.  Important keys include:

- `fd_hist_bins` – number of histogram bins to use when the automatic [Freedman–Diaconis rule](https://en.wikipedia.org/wiki/Freedman%E2%80%93Diaconis_rule) fails.
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
- `unbinned_likelihood` – when `true` use an extended unbinned likelihood
  instead of the default χ² fit to histogrammed data.
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
    "amp_prior_scale": 1.0,
    "unbinned_likelihood": false
}
```

`dump_time_series_json` under `plotting` saves a `*_ts.json` file
containing the binned time-series data when set to `true`.


`adc_drift_rate` under `systematics` applies a linear time-dependent
shift to the raw ADC values before calibration.  The value is in ADC
counts per second and defaults to `0.0` (no correction).  When the rate
is non-zero `analyze.py` applies the shift using
`apply_linear_adc_shift` and stores the value in `summary.json` under
`adc_drift_rate`.  More complex drift corrections can be configured via
`adc_drift_mode` and `adc_drift_params`.  Supported modes are
`"linear"`, `"quadratic"` and `"piecewise"`; the last two require
additional parameters as documented in `systematics.apply_linear_adc_shift`.

`sigma_E_frac`, `tail_fraction` and `energy_shift_keV` provide the
magnitude of systematic shifts applied during the scan.  The first two
are interpreted fractionally relative to the current parameter values,
while `energy_shift_keV` is an absolute shift supplied in keV.
Values with this suffix are automatically converted to MeV.
Each entry is optional and only affects the scan when present.

`plot_time_style` chooses how the histogram is drawn in the time-series
plot.  Use `"steps"` (default) for a stepped histogram or `"lines"` to
connect bin centers with straight lines.  The line style is useful when
overlaying multiple isotopes so one does not obscure the other.

`overlay_isotopes` under `plotting` keeps both isotope windows intact
when invoking `plot_time_series`.  When set to `true` the analysis does
not clear the other window, allowing Po‑214 and Po‑218 to be plotted
together on a single overlay.
Specifying `window_po210` (and optional `eff_po210`) adds a Po‑210
histogram to the time-series plots. The model curve appears only when
fit results for Po‑210 are available.

`palette` under `plotting` selects the color scheme used for all plots.
Available options are `"default"`, `"colorblind"` and `"grayscale"`.
The command line option `--palette NAME` overrides the configuration.


`plot_time_normalise_rate` controls how the y-axis is scaled in the
time-series plot.  With the default `true` the histogram is normalised to
counts per second.  Set it to `false` to show the raw counts per bin.

Example snippet:

```json
"plotting": {
    "plot_time_normalise_rate": false
}
```

`plot_time_series` takes its half-life values from the `time_fit` section.
Specify custom values using the keys `hl_po214`, `hl_po218` and `hl_po210`.
When these keys are omitted, `hl_po214` and `hl_po218` fall back to their
physical half-lives (≈164 µs and ≈183 s). `hl_po210` defaults to its physical
half-life (≈138 days). These custom half-lives control the decay model drawn
over the time-series histogram.
The same values are used in the `time_fit` routine itself, so changing
`hl_po214` or `hl_po218` affects both the unbinned fit and the overlay in
`plot_time_series`. For monitoring that spans multiple days you may set
them to the radon half-life (~3.8 days) to match the slowly varying
radon activity.

`sig_n0_po214` and `sig_n0_po218` set the uncertainty on the prior for the
initial activity `N0` when no baseline range is provided.  Without a baseline,
the fit applies a Gaussian prior `(0, sig_n0_{iso.lower()})` so `N0` may vary
rather than being fixed to zero.  Use the lower-case `sig_n0_{iso}` keys; the
legacy `sig_N0_{iso}` form is still accepted for backward compatibility.  The default width is `1.0` if not
specified in the configuration.


`settling_time_s` was removed from the `time_fit` section and is no
longer needed. The CLI option `--settle-s` may be used instead to
discard the first seconds of data before the decay fit.

### Fitting Long Time Scales

When the data covers months or more, the short half-lives of Po‑218 and
Po‑214 no longer matter.  The defaults therefore set `hl_po214` and
`hl_po218` to the radon half-life (≈3.8 days) so the fit tracks the slowly
varying radon concentration.  The configuration values are in seconds;
3.8 days corresponds to roughly `3.8 * 86400 ≈ 3.3e5` seconds.

Example snippet:

```json
"time_fit": {
    "hl_po214": [328320, 0.0],
    "hl_po218": [328320, 0.0],
    "hl_po210": [11923200, 0.0],
    "sig_n0_po214": 1.0,
    "sig_n0_po218": 1.0
}
```

These half-life values may also be set on the command line with
`--hl-po214` and `--hl-po218`.

### Baseline Runs

A baseline run measures the radon background with an empty monitor before
an assay. Configuration must define these keys under `baseline`:

- `baseline.range` – list of two ISO‑8601 timestamps selecting the baseline interval.
- `monitor_volume_l` – internal volume of the radon monitor in liters.
- `sample_volume_l` – volume of the assay sample in liters.
- `isotopes_to_subtract` – list of isotopes whose baseline rates are
  subtracted from the fitted decay rates. The default is
  `["Po214", "Po218"]`.

Events collected during the baseline period are counted in the selected
isotope windows. The counts for each isotope are converted into a decay
rate in Bq by dividing by the baseline live time and the corresponding
detection efficiency.  Each rate is scaled by the dilution factor
`monitor_volume_l / (monitor_volume_l + sample_volume_l)` before being
subtracted from the fitted radon decay rate of the assay. The multiplicative
scale factors for Po-214, Po-218, Po-210 and electronic noise are stored in
`summary.json` under `baseline.scales`. The command-line
option `--baseline_range` overrides `baseline.range` from the
configuration when provided. When you specify this option the
configuration's interval is ignored in favour of the CLI value.
The `--baseline-mode` option selects the background removal strategy.
Valid modes are `none`, `electronics`, `radon` and `all` (default).


Example snippet:

```json
"baseline": {
    "range": ["2023-07-01T00:00:00Z", "2023-07-03T00:00:00Z"],
    "monitor_volume_l": 10.0,
    "sample_volume_l": 5.0,
    "isotopes_to_subtract": ["Po214", "Po218"]
}
```

Command line usage:

```bash
# subtract electronics + radon
python analyze.py --config assay.json --input run.csv --output_dir results \
    --baseline_range 2023-07-01T00:00:00Z 2023-07-03T00:00:00Z \
    --baseline-mode all

# keep self-emanation, remove only electronics
python analyze.py --config assay.json --input run.csv --output_dir results \
    --baseline_range 2023-07-01T00:00:00Z 2023-07-03T00:00:00Z \
    --baseline-mode electronics

# debugging: no subtraction
python analyze.py --config assay.json --input run.csv --output_dir results \
    --baseline_range 2023-07-01T00:00:00Z 2023-07-03T00:00:00Z \
    --baseline-mode none
```

### Baseline Noise Cut

The helper `baseline_noise.estimate_baseline_noise` extracts the electronic
noise level from baseline events. Its optional `pedestal_cut` parameter
omits ADC values at or below the supplied threshold. When a baseline range
is specified, `analyze.py` forwards `calibration.noise_cutoff` as this value.

Example configuration to tighten the cut (set it to `null` to disable):

```json
"calibration": {
    "noise_cutoff": 300
}
```

## Utility Conversions

`utils.py` provides simple helpers to convert count rates and to search for
peak centroids:

- `cps_to_cpd(rate_cps)` converts counts/s to counts/day.
- `cps_to_bq(rate_cps, volume_liters=None)` returns the activity in Bq, or
  Bq/m^3 when a detector volume is supplied.
- `find_adc_bin_peaks(adc_values, expected, window=50, prominence=0.0, width=None)`
  histogramises the raw ADC spectrum, searches for maxima near each expected
  centroid and returns a `{peak: adc_centroid}` mapping in ADC units.

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
Here `activity_bq` is the spike activity expressed in decays per second (Bq).

`analyze.py` stores the calculated values and their BLUE combination in
`summary.json` under the `efficiency` key.

The helper `blue_combine.py` exposes a small wrapper so the combination
can be used independently via ``from blue_combine import BLUE``.

The option `--efficiency-json PATH` may be supplied on the command line to
load the efficiency section from a separate file instead of embedding it
directly in the main configuration.  Similarly `--systematics-json PATH`
overrides the `systematics` section.


## Running Tests

Install the required packages from `requirements.txt` before running the tests.
You can do this directly or via the provided helper script:
```bash
pip install -r requirements.txt   # or: bash scripts/setup_tests.sh
pytest -v
```

## Local Test Setup

When developing locally, run the setup script and then execute `pytest`:

```bash
scripts/setup_tests.sh
pytest -v
```


## Hierarchical Analysis

Use `hierarchical.py` to perform Bayesian hierarchical inference across multiple runs. The function `fit_hierarchical_runs` pools measurements of the half-life and calibration constants. It returns posterior means, standard deviations and 95% credible intervals for the global parameters. Running `analyze.py` with `--hierarchical-summary result.json` collects the half-life and calibration outputs from all `summary.json` files under the chosen output directory and writes the combined fit to `result.json`.

`fit_hierarchical_runs` requires the `pymc` package for sampling. Install it
along with the other dependencies:

```bash
pip install -r requirements.txt
```

Example usage:

```python
from hierarchical import fit_hierarchical_runs

run_results = [
    {"half_life": 160.5, "dhalf_life": 1.2, "slope_MeV_per_ch": 0.001, "dslope": 0.0005},
    {"half_life": 162.1, "dhalf_life": 1.0, "slope_MeV_per_ch": 0.0011, "dslope": 0.0004},
]

summary = fit_hierarchical_runs(run_results)
print(summary)
```

