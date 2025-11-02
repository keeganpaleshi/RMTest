# EMG Configuration Guide

This document describes the EMG (Exponentially Modified Gaussian) configuration options available in RMTest.

## Overview

The EMG configuration controls how exponential tails are applied to spectral peaks during fitting. This is crucial for accurately modeling peak shapes in alpha spectroscopy where detector response and physical processes create asymmetric tails.

## Configuration Structure

All EMG configuration lives under the `fitting.emg` section of your configuration file:

```yaml
fitting:
  emg:
    stable_mode: true      # Use numerically stable EMG implementation
    min_tau: 5.0e-4       # Minimum tau value (hard floor)
    method: "erfcx"        # Numerical method: "erfcx" or "direct"
    use_emg: true         # Global EMG enable/disable
```

## Configuration Options

### `fitting.emg.stable_mode`

**Type**: Boolean
**Default**: `true`
**Description**: Controls whether to use the numerically stable EMG implementation.

- `true`: Use stable implementation (recommended)
- `false`: Use legacy scipy implementation

### `fitting.emg.min_tau`

**Type**: Float
**Default**: `5.0e-4`
**Description**: Hard floor for tau values. All tau values will be clamped to be at least this value. This prevents numerical instability from very small tau values.

**Units**: Same as your ADC or time units (typically seconds or ADC equivalent)

### `fitting.emg.method`

**Type**: String
**Default**: `"erfcx"`
**Description**: Selects the numerical method for EMG evaluation.

- `"erfcx"`: Stable method using the scaled complementary error function (recommended)
- `"direct"`: Direct calculation method (legacy)

### `fitting.emg.use_emg`

**Type**: Boolean or Mapping
**Default**: `false`
**Description**: Controls which isotopes use EMG tails.

**As a boolean**:
```yaml
fitting:
  emg:
    use_emg: true  # Enable EMG for all isotopes
```

**As a mapping** (per-isotope control):
```yaml
fitting:
  emg:
    use_emg:
      Po210: true
      Po218: false
      Po214: false
```

## Per-Isotope Tau Priors

You can specify tau priors for individual isotopes in the `spectral_fit` section:

```yaml
spectral_fit:
  tau_Po210_prior_mean: 0.004    # Mean tau value for Po210
  tau_Po210_prior_sigma: 0.001   # Uncertainty in tau
  tau_Po218_prior_mean: 0.003
  tau_Po218_prior_sigma: 0.0005
```

**Important**: If `tau_{isotope}_prior_mean` is specified for an isotope, EMG is automatically enabled for that isotope, regardless of the `use_emg` setting.

## Resolution Order

The system determines whether to use EMG for each isotope using the following precedence:

1. **Highest Priority**: If `tau_{isotope}_prior_mean` exists, EMG is **ON** for that isotope
2. **Medium Priority**: If `fitting.emg.use_emg` is a mapping, use the value for that isotope
3. **Low Priority**: If `fitting.emg.use_emg` is a boolean, use that value for all isotopes
4. **Default**: EMG is **OFF** if none of the above are specified

## Tau Value Resolution

When fitting, tau values are resolved as follows:

1. If a prior is specified (`tau_{isotope}_prior_mean`), start from that value
2. Fit will optimize tau within bounds
3. Final tau is clamped: `tau_final = max(tau_fitted, min_tau)`

This ensures that tau never goes below the configured `min_tau` floor.

## Complete Example

```yaml
fitting:
  emg:
    stable_mode: true
    min_tau: 5.0e-4
    method: "erfcx"
    use_emg:
      Po210: true
      Po218: false
      Po214: false

spectral_fit:
  expected_peaks:
    Po210: 1250
    Po218: 1400
    Po214: 1800
  # Po210 has explicit tau prior (EMG automatically enabled)
  tau_Po210_prior_mean: 0.005
  tau_Po210_prior_sigma: 0.002
  # Po218 EMG is OFF (use_emg: false, no prior)
  # Po214 EMG is OFF (use_emg: false, no prior)
```

## Migration from Legacy Configuration

If you're migrating from older configuration formats:

- `fitting.use_stable_emg` → `fitting.emg.stable_mode`
- `fitting.emg_tau_min` → `fitting.emg.min_tau`
- `fitting.emg_stable_mode` → `fitting.emg.stable_mode`

The old configuration keys are still supported for backward compatibility but will be deprecated in future versions.

## Best Practices

1. **Start with defaults**: The default `min_tau = 5e-4` works well for most cases
2. **Use priors when available**: If you have calibration data, specify tau priors for better fits
3. **Enable EMG selectively**: Only enable EMG for peaks that genuinely show asymmetry
4. **Use erfcx method**: The `"erfcx"` method is more stable for small tau and narrow sigma
5. **Monitor tau values**: Check fitted tau values in your results; if they're hitting `min_tau`, you may need to adjust

## Troubleshooting

**Problem**: Fits are unstable or produce NaN values
**Solution**: Ensure `stable_mode: true` and consider increasing `min_tau`

**Problem**: Tau values are always at the minimum
**Solution**: Your data may not support EMG; consider disabling EMG for that isotope

**Problem**: Fit is too slow
**Solution**: Try `method: "direct"` for faster (but less stable) calculation

## Technical Details

The EMG distribution is a convolution of a Gaussian and an exponential distribution:

- **mu** (μ): Gaussian mean (peak center)
- **sigma** (σ): Gaussian standard deviation (resolution)
- **tau** (τ): Exponential decay constant (tail parameter)

The `min_tau` parameter prevents numerical issues when τ/σ becomes very small, where the EMG approaches a pure Gaussian but the calculation becomes unstable.
