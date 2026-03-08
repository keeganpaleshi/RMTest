# Time Handling and Units

Time parsing and units are shared across calibration, filtering, fits, plots, and the exported JSON files. This page consolidates those conventions.

## Core Conventions

- Time values are interpreted in seconds unless an output explicitly labels another unit such as elapsed time `(h)`.
- Energies are expressed in MeV throughout the code and documentation.
- Event timestamps are normalized to UTC as early as possible in the pipeline.
- Naive input timestamps are interpreted in the timezone selected by `--timezone`, which defaults to `UTC`.
- Internally, timestamps remain timezone-aware datetime objects. Unix epoch seconds are produced only where numeric fitting or histogramming requires them.

## Accepted Timestamp Inputs

The shared helpers in `utils.time_utils` accept:

- ISO-8601 strings
- Unix epoch seconds
- Python `datetime` objects
- `pandas.Timestamp`
- `numpy.datetime64`

These rules apply to:

- CSV `timestamp` values
- `analysis_start_time` and `analysis_end_time`
- `spike_start_time` and `spike_end_time`
- `spike_periods`
- `run_periods`
- `radon_interval`
- `baseline.range`
- CLI equivalents such as `--baseline-range`, `--spike-period`, and `--run-period`

## Shared Time Helpers

`utils.time_utils` exposes:

- `to_utc_datetime(value)`: convert a supported input to a timezone-aware UTC `datetime`
- `parse_timestamp(value)`: convert the same inputs to a timezone-aware UTC `pandas.Timestamp`
- `to_epoch_seconds(value)`: convert the same inputs to Unix epoch seconds

Backward-compatible wrappers remain available:

- `utils.to_utc_datetime(...)`
- `utils.parse_time_arg(...)`
- `utils.parse_datetime(...)` as a deprecated alias for `time_utils.parse_timestamp(...)`

There is also a small UTC-aware helper in `baseline_utils`:

- `baseline_period_before_data(end, start)`: compare a baseline interval and a data window in UTC, regardless of whether the inputs were naive or timezone-aware

## Plot Time Axes

The time-series plots mix two related time axes:

- absolute UTC timestamps for the real run timeline
- elapsed hours for the relative-time view used in monitoring plots

When `--dump-time-series-json` is enabled, the exported `*_ts.json` files include:

- UTC bin edges and centers
- Unix-second bin edges and centers
- elapsed-hour bin centers
- per-bin widths in seconds

## Energy Units

- Raw event input is in ADC channels.
- Calibration converts ADC to `energy_MeV`.
- Spectrum fitting and plotting operate on MeV, even when the histogram was initially formed in ADC bins.
- `mu_bounds_units: adc` is converted to MeV before the fit starts so the fit itself stays on one energy scale.

`CalibrationResult` stores the ADC-to-MeV conversion and its covariance:

```python
from calibration import CalibrationResult

cal = CalibrationResult(
    coeffs=[0.0, 0.001],
    cov=[[0.1**2, 0.0], [0.0, (5e-5)**2]],
)

energies = cal.predict([1500, 1700])
sigmas = cal.uncertainty([1500, 1700])
```

## Activity and Concentration Units

Several related units appear in the pipeline:

- fitted time-series activities are in Bq
- `radon_activity.png` and related concentration plots use Bq/L
- `total_radon.png` reports total activity in the combined monitor-plus-sample volume in Bq
- ambient concentration for `--ambient-file` and `--ambient-concentration` is expressed in Bq/L
- radon-inference ambient reference data is expressed in Bq/m^3

One utility helper uses a different concentration convention:

- `utils.cps_to_bq(rate_cps, volume_liters=...)` returns Bq/m^3 when a volume is supplied

That helper is useful for converting an activity rate to a volumetric concentration, but it is not the same unit convention used by the `radon_activity.png` plot.

## Utility Conversions

`utils.py` exposes several small helpers:

- `cps_to_cpd(rate_cps)`: counts per second to counts per day
- `cps_to_bq(rate_cps, volume_liters=None)`: activity in Bq, or Bq/m^3 when a detector volume is supplied
- `find_adc_bin_peaks(...)`: find raw ADC peak centroids near expected positions

Examples:

```bash
python utils.py 0.5 --to cpd
python utils.py 0.5 --to bq --volume-liters 10
```

`find_adc_bin_peaks(adc_values, expected, window=50, prominence=0.0, width=None, method="prominence")` returns a `{peak: adc_centroid}` mapping in ADC units. Set `method="cwt"` to use wavelet-based peak detection via `find_peaks_cwt`.
