# Radon Monitor Analysis Pipeline

This repository analyzes electrostatic radon monitor event data from raw CSV input through calibration, spectral and time-series fits, baseline subtraction, radon activity estimation, optional radon inference, and report generation.

Time values are interpreted in seconds unless an output explicitly labels another unit such as elapsed hours. Energies are expressed in MeV throughout the code and documentation. Input timestamps are normalized to UTC, and `--timezone` controls how naive timestamps are interpreted.

## Installation

Python 3.11 or newer is required.

```bash
pip install -r requirements.txt
```

## Quick Start

```bash
python analyze.py --config config.yaml --input merged_data.csv --output-dir results
```

The input CSV must contain at least `timestamp` and `adc` columns. Alternate column names can be mapped in the `columns` section of the configuration.

Results are written to `results/<timestamp>/` by default, or `results/<job-id>/` when `--job-id` is supplied. Common artifacts include `summary.json`, `config_used.json`, spectrum plots, isotope time-series plots, and radon activity plots. Hyphenated CLI flags such as `--output-dir` and `--baseline-range` are canonical; the older underscore forms remain accepted.

## Documentation

- [CLI reference](docs/cli.md)
- [Configuration reference](docs/configuration.md)
- [Baseline runs and subtraction](docs/baseline.md)
- [Time handling and units](docs/time-and-units.md)
- [Analysis modes](docs/analysis-modes.md)
- [Validation and fit quality](docs/validation.md)
- [Output schema and artifacts](docs/output-schema.md)
- [EMG configuration guide](docs/emg_config.md)
- [Radon inference from daughters](docs/radon_inference.md)

## Testing

```bash
pip install -r requirements.txt
pytest -v
```

You can also use `scripts/setup_tests.sh` before running `pytest`.
