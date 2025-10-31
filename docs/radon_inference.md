# Radon inference from daughters

The radon inference stage converts the fitted Po-214 and Po-218 time-series into
an inferred Rn-222 activity inside the monitor.  The stage consumes the
per-bin isotope counts produced for plotting, rescales them by the configured
detector efficiencies, and optionally folds in transport and retention
corrections.  When an ambient mine-radon time-series is provided the inferred
activity is divided by that concentration to yield an equivalent sampled volume
for each time bin.

## What it does

1. Selects the configured source isotopes (Po-214, Po-218) and reads their
   per-bin counts and live times.
2. Converts the counts to activities using the provided detection efficiencies
   and optional transport/retention efficiencies.
3. Combines multiple isotopes using weighted averages to produce a single
   Rn-222 activity estimate for every time bin.
4. Divides the activity by the external mine radon concentration (when
   available) to report an equivalent sampled volume and cumulative volume.

These outputs are stored under `summary.radon_inference` so that downstream
reporting and plotting stages can use a consistent data source.

## Why

The inferred Rn-222 series allows direct comparison between the monitor and an
independent radon sensor.  By expressing the daughter counts as an equivalent
air intake the plots become immediately interpretable: a mismatch indicates
either disequilibrium in the chain or transport/retention issues in the
monitor.

## Configuration

Enable the stage with the `radon_inference` block.  The snippet below matches
the defaults used in the integration tests:

```yaml
radon_inference:
  enabled: true
  source_isotopes: [Po214, Po218]
  source_weights:
    Po214: 0.7
    Po218: 0.3
  detection_efficiency:
    Po214: 0.12
    Po218: 0.10
  transport_efficiency: 1.0
  retention_efficiency: 1.0
  chain_correction: none
  external_rn:
    mode: file          # or "constant"
    file_path: mine_rn_timeseries.csv
    interpolation: nearest
    allowed_skew_seconds: 600
    default_bq_per_m3: 80.0
```

## Limitations

* If the daughter isotopes are not in equilibrium with Rn-222 (for example
  immediately after a step change in mine concentration) the inferred radon
  activity will lag the true value.
* Equivalent volume calculations can become unrealistically large when the
  external radon concentration is set too low.  Ensure that the ambient monitor
  calibration is sound before drawing conclusions from the flow rate plot.
* Transport and retention efficiencies are applied as simple scale factors.  If
  they are unknown leave them at 1.0, but note that the inferred volumes then
  represent lower bounds.

