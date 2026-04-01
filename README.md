# Radon Monitor Analysis Pipeline

Spectral fitting and time-series analysis for electrostatic radon monitor data. Takes raw event-level CSV (timestamps + ADC channels) through energy calibration, alpha spectrum fitting, DNL correction, time-series extraction, and radon activity estimation.

## Pipeline overview

1. **Load and calibrate** - Map ADC channels to energy (MeV) via two-point or auto calibration with polynomial (linear/quadratic/cubic) models.
2. **Spectral fit** - Binned NLL fit of the alpha spectrum. Peak shapes are Gaussian + EMG tail + shelf + halo for Po-210, Po-218, Po-214, Po-216, Po-212. Shape parameters can be per-isotope or shared across isotopes.
3. **DNL correction** - Compensates for ADC differential non-linearity. The primary method is a Fourier model at SAR bit-cycling periods (4, 8, ..., 512 codes), with per-period crossvalidation to reject harmonics that overfit. A two-stage variant estimates DNL at single-channel resolution, then rebins before the final fit.
4. **Time-series analysis** - Per-isotope count rates in configurable time bins, with Poisson + calibration-uncertainty error bars and optional decay-model overlays.
5. **Radon inference** - Converts daughter count rates to radon activity using detection/transport efficiencies. Optional comparison against Lucas cell assay data.
6. **Outputs** - JSON summary, spectrum plots, time-series plots, pull diagnostics, split-half validation results.

## Key files

| File | Role |
|------|------|
| `analyze.py` | Main pipeline (~5500 lines) |
| `calibration.py` | Energy calibration |
| `fitting.py` | Spectral NLL fitting, prior/bound construction |
| `config.yaml` | Default configuration |
| `radon_activity.py` | Radon activity calculation |
| `plot_utils/` | Time-series and diagnostic plotting |
| `run_scan.py` | Automated scan infrastructure |
| `assay_bridge.py` | Lucas cell assay comparison |

## Installation

Python 3.11+.

```bash
pip install -r requirements.txt
```

## Usage

```bash
python analyze.py --config config.yaml --input merged_data.csv --output-dir results
```

The input CSV needs at least `timestamp` and `adc` columns (alternate names configurable in the YAML). Results go to `results/<timestamp>/` by default, or `results/<job-id>/` with `--job-id`.

## Scan infrastructure

`run_scan.py` automates systematic exploration of the fitting configuration space. It takes a base config, applies overrides for each scan point, runs the full pipeline, and collects fit-quality metrics (chi2/ndf, AIC, pull diagnostics, covariance status) into `scan_results.csv`.

The scan grid covers three axes:
- **Shape parameter sharing** - shared vs. per-isotope freedom for tail, shelf, and halo parameters
- **DNL correction mode** - none, Fourier, bandpass, hybrid, and two-stage full-resolution Fourier
- **Binning and Fourier period selection** - ADC bin width, rebin factor, smooth window, iteration count, and which Fourier periods to include

```bash
# Run a single scan
python run_scan.py --scans F2

# Run a batch
python run_scan.py --scans D1,D2,D3,D4

# Run everything (A1 through G6, takes many hours)
python run_scan.py
```

The current best-performing configuration uses the two-stage pipeline: Fourier DNL estimated at full ADC resolution (bin_width=1) with crossvalidation-selected periods, then rebinned before the final fit (F-series scans). The G-series further explored minimal Fourier basis selection.

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
pytest -v
```
