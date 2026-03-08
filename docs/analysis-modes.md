# Analysis Modes

This page collects the higher-level modes that change how RMTest interprets or combines the data. Most of them are opt-in and leave the historical defaults unchanged unless you enable them explicitly.

## Background and Likelihood Modes

Two experimental switches are exposed through the `analysis` section or the CLI:

- `background_model: linear` or `loglin_unit`
- `likelihood: current` or `extended`

The defaults preserve the historical behavior. `loglin_unit` uses a unit-area log-linear background shape scaled by a positive `S_bkg` parameter, and `extended` includes the Poisson event-count term in the likelihood.

CLI example:

```bash
python analyze.py --input merged_data.csv \
    --background-model loglin_unit \
    --likelihood extended
```

Configuration example:

```yaml
analysis:
  background_model: loglin_unit
  likelihood: extended
```

The spectral fitter now uses a binned Poisson likelihood by default for numerical stability. The legacy unbinned path remains available through `spectral_fit.unbinned_likelihood: true`. The chosen spectral path is written to `summary.json` as `spectral_fit.likelihood_path`.

## Radon vs. Single-Progeny Mode

The `analysis_isotope` configuration key, or the `--iso` CLI flag, determines which daughter chain drives the final radon estimate:

- `radon`: combine Po-218 and Po-214 with inverse-variance weighting
- `po214`: use only Po-214
- `po218`: use only Po-218

`radon` is the default and still records the result under the `radon` entry of `summary.json`, even when a single progeny was selected.

## Long-Timescale Time Fits

When a run spans many days or months, the short Po-214 and Po-218 half-lives are often not the physically useful timescale. In that case you can set `hl_po214` and `hl_po218` to the radon half-life so the fit tracks the slowly varying parent activity instead of the fast daughter decay.

Example:

```yaml
time_fit:
  hl_po214: 328320
  hl_po218: 328320
  hl_po210: 11923200
  sig_n0_po214: 1.0
  sig_n0_po218: 1.0
```

The same values can be supplied from the CLI with `--hl-po214` and `--hl-po218`.

## Two-Pass Time-Fit Background Mode

The time fitter supports a two-pass strategy:

1. First pass with the background term `B` fixed.
2. Second pass with `B` free.
3. Keep the second pass only if it is valid and improves the Akaike Information Criterion by at least `0.5`.

Configuration:

```yaml
time_fit:
  fix_background_b_first_pass: true
  background_b_fixed_value: null
```

When `background_b_fixed_value` is `null`, the first pass falls back to the baseline Po-214 rate if one is available.

## Hierarchical Analysis

`hierarchical.py` performs Bayesian hierarchical inference across multiple runs. `analyze.py --hierarchical-summary OUTFILE` scans prior `summary.json` files under the selected output directory, collects the half-life and calibration outputs, and writes the combined fit to `OUTFILE`.

Example:

```python
from hierarchical import fit_hierarchical_runs

run_results = [
    {"half_life": 160.5, "dhalf_life": 1.2, "slope_mev_per_ch": 0.001, "dslope": 0.0005},
    {"half_life": 162.1, "dhalf_life": 1.0, "slope_mev_per_ch": 0.0011, "dslope": 0.0004},
]

summary = fit_hierarchical_runs(run_results)
print(summary)
```

`fit_hierarchical_runs` requires `pymc`.

## Radon Inference Stage

The optional `radon_inference` stage converts daughter time-series into inferred Rn-222 activity and leak-related volume estimates. It is configured separately from the main radon estimate and writes its own summary block and plot set.

See [radon_inference.md](radon_inference.md) for the equations, interpolation rules, and outputs.
