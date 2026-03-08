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

The derived series bridges the gap between Po-214/Po-218 counts and the
ambient radon concentration. It turns the inferred radon inventory into a leak
rate and cumulative leaked-air estimate, which is closer to how leak checking
is normally interpreted in operational units. Without this conversion the
Po-214 counts are difficult to compare against ambient radon levels.

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

## Limitations

* When the chosen isotope is not in secular equilibrium with Rn-222 the
  inferred activity will lag the true radon level.
* Extremely small external radon concentrations produce unrealistically large
  leak-rate estimates - double check the reference monitor calibration in that
  case.
* Transport and retention efficiencies default to 1.0. If they are unknown the
  reported activity represents a lower bound.
* The leak-rate model currently assumes radioactive decay is the only sink.
  If the monitored volume is actively purged or vented, an additional removal
  term is required for a fully physical leak-rate estimate.
