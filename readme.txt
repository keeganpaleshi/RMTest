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

## Usage

python analyze.py --config config.json --input merged_data.csv [--output_dir results]
