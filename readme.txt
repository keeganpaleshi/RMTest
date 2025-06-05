# Radon Monitor Analysis Pipeline

This repository provides a complete pipeline to analyze electrostatic radon monitor data.

## Structure

- `analyze.py`: Main entry point to run the full analysis.
- `config.json`: JSON configuration file containing thresholds and options.
- `io_utils.py`: Functions to load raw data and write outputs.
- `calibration.py`: Peak-finding and energy calibration routines.
- `fitting.py`: Unbinned likelihood fit for Po-214 (and optional Po-218).
- `systematics.py`: Scan for systematic uncertainties (optional).
- `plot_utils.py`: Plotting routines for spectrum and time-series.
- `utils.py`: Miscellaneous utilities (time conversion, JSON validation).
- `tests/`: `pytest` unit tests for calibration, fitting, and I/O.

## Installation

```bash
pip install -r requirements.txt
```

## Usage

```bash
python analyze.py --config config.json --input merged_data.csv [--output_dir results]
```

## Output

The analysis writes results to `<output_dir>/<timestamp>/` including:

- `summary.json` – calibration and fit summary.
- `config_used.json` – copy of the configuration used.
- `spectrum.png` – spectrum plot with fitted peaks.
- `time_series_Po214.png` and `time_series_Po218.png` – decay time-series plots.
- Optional `*_ts.json` files containing binned time series when enabled.

The `time_fit` routine currently fits only Po‑214 and Po‑218.  Supporting
Po‑210 would require adding its half‑life and detection efficiency to the
configuration along with a plotting color in the code.

## Configuration

`nominal_adc` under the `calibration` section sets the expected raw ADC
centroids for Po‑210, Po‑218 and Po‑214 when using automatic calibration.
If omitted, defaults of `{"Po210": 1250, "Po218": 1400, "Po214": 1800}`
are used.

`time_bins_fallback` under the `plotting` section sets the number of
histogram bins to use when the automatic Freedman&ndash;Diaconis rule
fails, typically due to zero IQR.  The default is `1`.

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
- `b0_prior` / `b1_prior` – `[mean, sigma]` priors for the linear
  continuum terms.
- `tau_{iso}_prior_mean` and `tau_{iso}_prior_sigma` – mean and
  uncertainty for the exponential tail constant of each isotope when
  `use_emg` enables that tail.
- `mu_bounds` – optional lower/upper limits for each peak centroid.
  Set for example `{"Po218": [5.9, 6.2]}` to keep the Po‑218 fit from
  drifting into the Po‑210 region.  Centroid guesses found during peak
  search are clamped to this range before the fit starts.

`dump_time_series_json` under `plotting` saves a `*_ts.json` file
containing the binned time-series data when set to `true`.

`scan_keys` in the `systematics` section selects which fit parameters
are varied during the systematic uncertainty scan.  By default no
parameters are scanned.

`plot_time_style` chooses how the histogram is drawn in the time-series
plot.  Use `"steps"` (default) for a stepped histogram or `"lines"` to
connect bin centers with straight lines.

`overlay_isotopes` under `plotting` keeps both energy windows when
calling `plot_time_series`.  When `true`, Po‑214 and Po‑218 are overlaid
in the same plot instead of appearing separately.

`sig_N0_Po214` and `sig_N0_Po218` set the uncertainty on the prior for the
initial activity `N0` when no baseline range is provided.  Instead of fixing
`N0` strictly to zero, the time-series fit now uses a Gaussian prior centered at
zero with this width.


`settling_time_s` was removed from the `time_fit` section and is no
longer needed.

### Fitting Long Time Scales

When the data covers months or more, the short half-lives of Po‑218 and
Po‑214 no longer matter.  In that regime you may set `hl_Po214` and
`hl_Po218` to the radon half-life (≈3.8 days) so they track the slowly
varying radon concentration.  The configuration values are in seconds, so
3.8 days corresponds to roughly `3.8 * 86400 ≈ 3.3e5` seconds.

Example snippet:

```json
"time_fit": {
    "hl_Po214": [328320, 0.0],
    "hl_Po218": [328320, 0.0]
}
```


## Running Tests

Install the required Python packages and run the test suite with `pytest`.
The tests rely on `numpy`, `pandas`, `scipy`, `matplotlib`, `iminuit`, and
`pytest` which are all listed in `requirements.txt`.

```bash
pip install -r requirements.txt
pytest -v
```

## Local Test Setup

To run the tests locally, install the dependencies and execute `pytest`:

```bash
pip install -r requirements.txt
pytest -v
```

