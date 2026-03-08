# Radon inference from daughters

The radon inference stage converts fitted isotope time-series data into an
estimate of the Rn-222 activity inside the detector cell. It consumes the
Po-214 and Po-218 count histograms generated during the decay-rate fitting and
translates them into inferred radon levels, aligned with the original binning.

## What it does

* Converts per-bin counts into isotope activities using the configured
  detection efficiencies.
* Applies optional transport and retention efficiency corrections to obtain the
  inferred Rn-222 activity inside the cell.
* Combines multiple isotopes via configurable weights and records the derived
  series in the analysis summary for downstream reporting.
* Uses the ambient radon concentration together with the Rn-222 decay
  constant to estimate a leak rate, a per-interval leaked volume, and a
  cumulative leaked-volume sum that is intended to be stable under rebinning.

## Why it matters

The derived series bridges the gap between Po-214 and Po-218 counts and the
ambient radon concentration. It turns the inferred radon inventory into a leak
rate and cumulative leaked-air estimate, which is closer to how leak checking
is normally interpreted in operational units. Without this conversion the
daughter counts are difficult to compare against ambient radon levels.

## Configuration

Configure the stage in `config.yaml` under `radon_inference`:

```yaml
radon_inference:
  enabled: true
  source_isotopes: ["Po214", "Po218"]
  source_weights:
    Po214: 0.7
    Po218: 0.3
  detection_efficiency:
    Po214: 0.19
    Po218: 0.094
  transport_efficiency: 1.0
  retention_efficiency: 1.0
  chain_correction: none
  external_rn:
    mode: constant
    constant_bq_per_m3: 120.0
```

Set `mode: file` inside `external_rn` to point at a CSV with `timestamp` and
`rn_bq_per_m3` columns. Missing samples fall back to the optional
`constant_bq_per_m3` value after interpolation.

## Mathematical Formulas

The calculation follows these steps:

1. Convert counts to isotope activity for each contributing isotope:

   ```text
   A_iso = counts / (eff * dt)
   ```

2. Combine the isotope activities with normalized source weights:

   ```text
   A_rn = sum(w_i * A_iso_i)
   ```

3. Apply transport and retention corrections:

   ```text
   A_rn_corrected = A_rn / (transport * retention)
   ```

4. Estimate the leak rate using the interval-average radon inventory balance:

   ```text
   dA_rn/dt = q * C_mine - lambda_Rn * A_rn
   ```

   In RMTest this is evaluated per interval using the interval-start activity
   `A_start,i`, the interval-average activity `A_mean,i`, the interval width
   `Delta t`, and the ambient concentration `C_i`:

   ```text
   survival = exp(-lambda_Rn * Delta t)
   response = (1 - survival) / lambda_Rn
   mean_weight = response / Delta t
   leak_weight = 1 - mean_weight
   q_i = lambda_Rn * (A_mean,i - A_start,i * mean_weight) / (C_i * leak_weight)
   ```

   The first interval uses the steady-state estimate:

   ```text
   q_0 = lambda_Rn * A_mean,0 / C_0
   ```

5. Convert the leak rate into per-interval and cumulative leaked volume:

   ```text
   Delta V_i = max(q_i, 0) * Delta t
   V_cum = sum_i Delta V_i
   ```

Negative leak rates caused by statistical fluctuations are clipped to zero so
the cumulative leaked-volume estimate remains monotonic.

## Overlay and Deduplication

When `overlay_isotopes` is enabled, RMTest can receive the same isotope time
bin from more than one plot source. Radon inference expects unique bins, so
bins are deduplicated by time within microsecond precision and identical
`counts` and `dt`. When exact duplicates are encountered for the same isotope
and timestamp, the last entry is retained.

## Sparse Ambient Inputs

When the ambient radon monitor reports more slowly than RMTest, the external
values are interpolated to each RMTest timestamp using the `interpolation`
field in `external_rn`:

* `nearest`
* `ffill`

`allowed_skew_seconds` defines the maximum permitted time gap. If the gap
exceeds that threshold, RMTest falls back to `constant_bq_per_m3` or
`default_bq_per_m3` when provided.

## Outputs

When enabled, radon inference records a `radon_inference` block in
`summary.json` and can write a dedicated plot set including:

* `radon_inferred.png`
* `ambient_radon.png`
* `equivalent_volume_liters.png`
* `equivalent_flow.png`
* `equivalent_volume_cumulative.png`
* `equivalent_volume_cumulative_liters.png`

## Limitations

* When the chosen isotope is not in secular equilibrium with Rn-222 the
  inferred activity will lag the true radon level.
* Extremely small external radon concentrations produce unrealistically large
  leak-rate estimates; double check the reference monitor calibration in that
  case.
* Transport and retention efficiencies default to `1.0`. If they are unknown
  the reported activity represents a lower bound.
* The leak-rate model currently assumes radioactive decay is the only sink.
  If the monitored volume is actively purged or vented, an additional removal
  term is required for a fully physical leak-rate estimate.
