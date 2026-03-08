# CLI Reference

`analyze.py` is the main entry point for the full pipeline. Command-line values override the corresponding entries in `config.yaml`.

## Basic Usage

```bash
python analyze.py --config config.yaml --input merged_data.csv \
    --output_dir results [--job-id RUN_ID] [--overwrite]
```

If `--output_dir` is omitted it defaults to `results`. If `--job-id` is omitted, the run directory is timestamped automatically.

## Input Requirements

The input file must be a CSV containing at least:

- `timestamp`: event time, provided as ISO-8601, Unix epoch seconds, or values parsable by the shared timestamp helpers
- `adc`: raw ADC value

Additional columns such as `fUniqueID`, `fBits`, `fchannel`, `baseline_adc`, `spike_flag`, `valid`, `temperature`, `run_id`, `pressure`, and `humidity` are ignored unless a downstream step uses them. If your file uses different header names, remap them in the `columns` section of the configuration.

## Option Groups

### Inputs and outputs

- `--config`, `-c`: configuration file
- `--input`, `-i`: merged event CSV
- `--output_dir`, `-o`: parent output directory
- `--job-id`: explicit output folder name
- `--overwrite`: replace an existing result directory
- `--reproduce SUMMARY`: reuse the archived `config_used.json` and stored seed from a previous run's `summary.json`

### Time handling and run selection

- `--timezone`: timezone used for naive timestamps; default `UTC`
- `--analysis-start-time`, `--analysis-end-time`: clip the global analysis window
- `--spike-start-time`, `--spike-end-time`: discard data after or before a single cutoff
- `--spike-period START END`: exclude one or more explicit windows
- `--run-period START END`: keep one or more explicit windows
- `--radon-interval START END`: interval used for radon delta calculations
- `--settle-s`: discard the first N seconds before the decay fit

See [time-and-units.md](time-and-units.md) for accepted timestamp formats.

### Baseline and background handling

- `--baseline_range TSTART TEND`: override `baseline.range`
- `--baseline-mode {none,electronics,radon,all}`: choose what is removed from the assay result
- `--allow-negative-baseline`: preserve negative baseline-corrected rates instead of clipping them to zero
- `--allow-negative-activity`: preserve negative total radon values instead of clamping to zero
- `--check-baseline-only`: print baseline diagnostics and exit early

See [baseline.md](baseline.md) for the subtraction model and summary fields.

### Calibration, filtering, and fit preparation

- `--noise-cutoff`: override `calibration.noise_cutoff`
- `--calibration-slope`: override `calibration.slope_MeV_per_ch`
- `--float-slope`: treat a supplied slope as an initial guess instead of fixing it
- `--calibration-method {two-point,auto}`: override `calibration.method`
- `--slope`: apply a linear ADC drift correction
- `--burst-mode {none,micro,rate,both}`: override `burst_filter.burst_mode`
- `--burst-sensitivity-scan`: sweep burst parameters and write `burst_scan.png`

### Analysis modes

- `--iso {radon,po218,po214}`: choose whether the final radon estimate uses both daughters or a single progeny
- `--background-model {linear,loglin_unit}`: opt into the alternate background parameterization
- `--likelihood {current,extended}`: opt into the alternate spectral likelihood
- `--hl-po214`, `--hl-po218`: override the time-fit half-lives in seconds
- `--eff-fixed`: force efficiencies to 1.0
- `--hierarchical-summary OUTFILE`: combine prior run summaries with the hierarchical fitter

See [analysis-modes.md](analysis-modes.md) for the mode definitions and use cases.

### Efficiency and systematics inputs

- `--efficiency-json`: load the `efficiency` block from a separate JSON file
- `--systematics-json`: load the `systematics` block from a separate JSON file
- `--spike-count`, `--spike-count-err`, `--spike-activity`, `--spike-duration`: override spike-efficiency inputs
- `--no-spike`: disable spike efficiency entirely

### Plotting and reporting

- `--plot-time-binning-mode {auto,fd,fixed}`: override `plotting.plot_time_binning_mode`
- `--time-bin-mode {auto,fd,fixed}`: deprecated alias for `--plot-time-binning-mode`
- `--plot-time-bin-width`: override `plotting.plot_time_bin_width_s`
- `--dump-ts-json`, `--dump-time-series-json`: write `*_ts.json` time-series payloads
- `--ambient-file`: time-dependent ambient radon file in Bq/L for equivalent-air plots
- `--ambient-concentration`: constant ambient radon concentration in Bq/L
- `--palette`: override the plotting palette
- `--strict-covariance`: raise instead of continuing when a fit covariance is not positive definite
- `--debug`: increase log verbosity
- `--seed`: override the random seed

See [output-schema.md](output-schema.md) for the produced artifacts.

## Examples

### Standard run

```bash
python analyze.py --config config.yaml --input merged_data.csv --output_dir results
```

### Baseline subtraction

```bash
python analyze.py --config assay.yaml --input assay.csv --output_dir results \
    --baseline_range 2023-07-01T00:00:00Z 2023-07-03T00:00:00Z \
    --baseline-mode all
```

### Experimental background and likelihood

```bash
python analyze.py --input merged_data.csv \
    --background-model loglin_unit \
    --likelihood extended
```

### Reproduce a previous run

```bash
python analyze.py --reproduce results/20260307T101710Z/summary.json
```

### Build a hierarchical summary from prior outputs

```bash
python analyze.py --output_dir results --hierarchical-summary result.json
```
