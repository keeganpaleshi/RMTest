# Radon inference from daughters

The radon inference stage converts the fitted Po-214 and Po-218 daughter
counts into an estimated Rn-222 activity inside the collection cell. The
conversion divides the observed counts by the configured detection
efficiencies and live-time for each time bin:

\[A_{\text{iso}} = \frac{N_{\text{counts}}}{\epsilon_{\text{det}} \cdot \Delta t}\]

Multiple isotopes can contribute to the estimate. The configured source
weights are renormalised to the isotopes that are actually present in the
time series so that missing data never causes the stage to fail. The weighted
activity is adjusted by the transport and retention efficiencies to recover
the implied Rn-222 activity in the monitor volume. A chain-correction mode of
`none` leaves the activity unchanged; other modes currently fall back to this
behaviour and emit a warning so the output remains well defined.

When `radon_inference.external_rn` is configured, the pipeline aligns the
ambient radon concentration with the isotope time bins. The instantaneous
equivalent sample volume is then computed by comparing the inferred activity
with the ambient concentration, and the cumulative volume is integrated across
all bins. This makes the derived plots directly interpretable against an
independent mine radon monitor.

## Why this matters

The daughter activities can be misleading on their own. Expressing them as an
Rn-222 time series allows direct comparison with independent sensors and
highlights periods where the detector intake differs from the mine
concentration. The equivalent volume curve is especially helpful when
investigating transport issues or retention inefficiencies because it shows
how much air the detector appears to have sampled.

## Configuration

The stage is controlled by the `radon_inference` block. A representative
configuration is shown below:

```yaml
radon_inference:
  enabled: true
  source_isotopes:
    - Po214
    - Po218
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
    mode: constant
    constant_bq_per_m3: 80.0
```

When `mode` is `file`, the loader reindexes the external series against the
time bins and falls back to the configured constant when data are missing. The
plot footer prints the efficiency assumptions so that reviewers can quickly
see the parameters used for the conversion.

## Limitations

* Po-214 alone will lag the true radon activity during transient periods.
* Unrealistically low ambient concentrations inflate the equivalent volume;
  check the external series when the volume plot looks suspicious.
* If transport or retention efficiencies are unknown, leave them at `1.0` and
  treat the inferred activity as a lower bound.

