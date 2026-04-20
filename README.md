# Radon Monitor Analysis Pipeline

This repo turns raw event-level radon monitor data into calibrated spectra, isotope time series, and radon activity estimates.

It is built around one main workflow:

1. Run `analyze.py` on real data.
2. Inspect the generated `summary.json`, spectrum plots, and time-series plots.
3. If you are comparing model choices, run `run_scan.py` and compare outputs across configurations.

The project is intentionally biased toward graceful runtime behavior and inspectable output artifacts over a large unit-test surface.

## Quick Start

Python 3.11+ is recommended.

```bash
pip install -r requirements.txt
python analyze.py --config config.yaml --input merged_data.csv --output-dir results
```

The input CSV needs at least `timestamp` and `adc` columns. By default the run writes to `results/<timestamp>/`. If you want a stable folder name, pass `--job-id`.

## What The Pipeline Does

- Load raw event data and normalize timestamps, ADC, and channel fields.
- Calibrate ADC to energy.
- Fit the alpha spectrum with the configured binned Poisson spectral model.
- Optionally estimate and apply ADC DNL correction.
- Build daughter-isotope time series.
- Convert daughter activity to radon activity when enabled.
- Write plots, diagnostics, and machine-readable summaries.

## What To Check After A Run

The normal validation path is to inspect outputs, not to rely on a large automated test suite.

Start with:

- `summary.json`
- `spectrum.png`
- `spectrum_components.png`
- time-series plots in the run directory
- warnings and diagnostics in the summary

In practice the highest-signal checks are:

- Did the run complete without fatal warnings?
- Do the fitted peaks sit in the right places?
- Do residuals and pulls look sane?
- Do activity/time-series trends look physically plausible?
- If you changed fitting assumptions, did the summary metrics and plots improve on representative data?

## Core Entry Points

| File | Purpose |
|------|---------|
| `analyze.py` | Main end-to-end pipeline |
| `config.yaml` | Default configuration |
| `fitting.py` | Spectral fitting |
| `calibration.py` | ADC-to-energy calibration |
| `radon_activity.py` | Daughter-to-radon inference |
| `plot_utils/` | Spectrum and time-series plotting |
| `run_scan.py` | Batch configuration scans |
| `assay_bridge.py` | Comparison to external assay measurements |

## Running Scans

Use `run_scan.py` when you want to compare configurations instead of arguing from priors.

```bash
python run_scan.py --scans F2
python run_scan.py --scans D1,D2,D3,D4
python run_scan.py
```

That produces per-run outputs plus aggregated scan metrics in `scan_results.csv`.

## Configuration And Docs

- [CLI reference](docs/cli.md)
- [Configuration reference](docs/configuration.md)
- [Validation and fit quality](docs/validation.md)
- [Output schema](docs/output-schema.md)
- [Analysis modes](docs/analysis-modes.md)
- [Time handling and units](docs/time-and-units.md)
- [Radon inference](docs/radon_inference.md)

## Minimal Tests

The repo keeps only a very small automated guardrail layer.

These tests are for silent normalization mistakes that can skew results without obviously crashing or making a plot look broken, mainly:

- timestamp normalization to UTC
- ADC/energy unit normalization for spectral bounds

Run them with:

```bash
pytest -q
```

For most changes, the real validation path should still be a representative pipeline run and direct inspection of the outputs.
