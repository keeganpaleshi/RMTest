# Baseline Runs and Subtraction

A baseline run measures monitor background with an empty detector or otherwise sample-free setup before an assay. The analysis pipeline can use that interval to estimate baseline rates, electronic noise, and the background term passed into the time-series fit.

## Required Configuration

The `baseline` section should define:

- `range`: two timestamps selecting the baseline interval
- `monitor_volume_l`: internal monitor volume in liters
- `sample_volume_l`: sample volume in liters
- `isotopes_to_subtract`: background components to subtract

Example:

```yaml
baseline:
  range: ["2023-07-01T00:00:00Z", "2023-07-03T00:00:00Z"]
  monitor_volume_l: 10.0
  sample_volume_l: 5.0
  isotopes_to_subtract: ["Po214", "Po218"]
```

If `isotopes_to_subtract` is omitted, `analyze.py` defaults to `["Po214", "Po218"]`. Some workflows also include `noise`, which is treated separately from isotope rates.

## Baseline Modes

The CLI flag `--baseline-mode` chooses the subtraction strategy:

- `none`: do not subtract baseline components
- `electronics`: remove only the electronics contribution
- `radon`: remove only the radon contribution
- `all`: remove both electronics and radon contributions

Examples:

```bash
# subtract electronics + radon
python analyze.py --config assay.yaml --input run.csv --output_dir results \
    --baseline_range 2023-07-01T00:00:00Z 2023-07-03T00:00:00Z \
    --baseline-mode all
```

```bash
# keep self-emanation, remove only electronics
python analyze.py --config assay.yaml --input run.csv --output_dir results \
    --baseline_range 2023-07-01T00:00:00Z 2023-07-03T00:00:00Z \
    --baseline-mode electronics
```

```bash
# debugging: no subtraction
python analyze.py --config assay.yaml --input run.csv --output_dir results \
    --baseline_range 2023-07-01T00:00:00Z 2023-07-03T00:00:00Z \
    --baseline-mode none
```

## How the Subtraction Works

Events collected during the baseline period are counted in the selected isotope windows. For each isotope:

1. The baseline counts are converted into a decay rate in Bq by dividing by the baseline live time and the relevant efficiency.
2. The baseline rate is scaled by the dilution factor

   ```text
   monitor_volume_l / (monitor_volume_l + sample_volume_l)
   ```

   before it is subtracted from the fitted assay rate.
3. The propagated uncertainty includes both the fit uncertainty and the counting uncertainty from the baseline interval.

The multiplicative scale factors for Po-214, Po-218, Po-210, and electronic noise are written under `baseline.scales` in `summary.json`.

Baseline subtraction for time-fit outputs is handled by `radon.baseline.subtract_baseline_rate`, which in turn uses `radon.baseline.subtract_baseline_counts` so the propagated uncertainty reflects the unweighted event statistics from the analysis window. `baseline_utils` re-exports these helpers for backward compatibility.

## Dedicated Long Baselines

A long monitor-only baseline can be re-used across multiple assays. The baseline interval does not need to overlap the assay or the configured `run_periods`.

First analyze the baseline period on its own:

```bash
python analyze.py --config examples/long_baseline.yaml --input baseline.csv \
    --output_dir baseline_results --job-id baseline
```

Later assay runs can reference that same interval:

```bash
python analyze.py --config assay.yaml --input assay.csv --output_dir results \
    --baseline_range 2023-07-01T00:00:00Z 2023-07-31T23:59:59Z
```

When results are written, the metadata may record `background_mode: fixed_from_baseline`, meaning the monitor background was measured in a dedicated baseline run and then held fixed during the assay analysis.

## Summary Fields

When baseline subtraction is active, `summary.json` records these values under `baseline`:

- `analysis_counts`: unweighted counts in the assay window for each isotope
- `rate_Bq` and `rate_unc_Bq`: baseline rates and their uncertainties
- `dilution_factor`: volume scale factor applied before subtraction
- `scales`: per-component subtraction multipliers
- `corrected_rate_Bq` and `corrected_sigma_Bq`: baseline-subtracted rates and uncertainties
- `corrected_activity`: baseline-corrected activity payload used for reporting
- `background_mode`: whether the time-fit background floated or was fixed from the baseline

Time-fit blocks may also include:

- `background_mode`
- `baseline_rate_Bq`
- `baseline_unc_Bq`
- provenance fields added by `baseline_handling.apply_time_fit_provenance`

## Negative-Value Policy

Set the top-level `allow_negative_baseline: true` or pass `--allow-negative-baseline` to preserve negative baseline-corrected activities exactly as fitted.

With the default `false` setting, negative baseline-corrected values are clipped to `0.0` so downstream consumers do not see negative activities unless you opt in explicitly.

## Baseline Noise Cut

The helper `baseline_noise.estimate_baseline_noise` estimates the electronic noise level from baseline events. Its optional `pedestal_cut` parameter omits ADC values at or below the supplied threshold. When a baseline range is present, `analyze.py` forwards `calibration.noise_cutoff` as that threshold.

Example:

```yaml
calibration:
  noise_cutoff: 300
```

Set `noise_cutoff: null` to disable the cut.

## Validation Rules

Baseline windows are validated before the run proceeds:

- `baseline.range` must contain exactly two timestamps
- the end must be after the start
- the interval must overlap the analysis window if `analysis_start_time` or `analysis_end_time` is set

See [validation.md](validation.md) for the exact error conditions.
