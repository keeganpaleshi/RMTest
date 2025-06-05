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

## Configuration

See `config.json` for the default settings.  Plotting options are grouped under
the `"plotting"` key.  A new `"plot_time_style"` option controls how the
time-series plots are drawn.  Use `"lines"` to draw simple line graphs so that
Po‑214 and Po‑218 do not obscure each other.  The default `"hist"` style shows
the data as step histograms.

## Running Tests

Install the required Python packages and run the test suite with `pytest`.
The tests rely on `numpy`, `pandas`, `scipy`, `matplotlib`, `iminuit`, and
`pytest` which are all listed in `requirements.txt`.

```bash
pip install -r requirements.txt
pytest -v
```

