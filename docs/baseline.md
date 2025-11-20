# Baseline Configuration

The `baseline` section in the configuration controls how background activity is measured and subtracted from the analysis results.

## Configuration Options

### `mode`
Type: `string` (enum: `auto`, `linear`, `manual`)

Specifies the baseline estimation mode:
- `auto`: Automatically determine the best baseline model
- `linear`: Use a linear baseline model with b0 (intercept) and b1 (slope) parameters
- `manual`: Use manually specified baseline values

### `allow_negative_baseline`
Type: `boolean`

Controls whether negative baseline-corrected rates are preserved or clipped:
- `false` (default): Negative corrected rates are clipped to 0.0
- `true`: Raw negative values are preserved in the output

**Note**: When `allow_negative_baseline` is `false`, any negative activity values resulting from baseline subtraction are automatically set to zero. This ensures downstream consumers never observe negative activities without explicitly opting in.

### `b0_prior`
Type: `array` of 2 numbers `[mean, sigma]`

Prior distribution parameters for the baseline intercept (b0) when using a linear baseline model.
- First element: Prior mean
- Second element: Prior standard deviation

### `b1_prior`
Type: `array` of 2 numbers `[mean, sigma]`

Prior distribution parameters for the baseline slope (b1) when using a linear baseline model.
- First element: Prior mean
- Second element: Prior standard deviation

### `range`
Type: `array` of 2 timestamps or numbers

Time interval for baseline measurement, specified as `[start, end]`.

### `monitor_volume_l`
Type: `number` (minimum: 0)

Detector chamber volume in litres.

### `sample_volume_l`
Type: `number` (minimum: 0)

Volume of sample air introduced to the chamber in litres.

### `isotopes_to_subtract`
Type: `array` of strings

List of isotopes or background components to subtract from the analysis results (e.g., `["noise"]`).

## Example Configuration

```yaml
baseline:
  mode: linear
  allow_negative_baseline: false
  b0_prior: [0.0, 0.1]
  b1_prior: [0.0, 0.1]
  range: ['2024-01-05T00:00:00Z', '2024-01-05T12:00:00Z']
  monitor_volume_l: 605.0
  sample_volume_l: 4196.0
  isotopes_to_subtract:
    - noise
```

## Backward Compatibility

All new baseline configuration options are optional and maintain backward compatibility with existing configurations. If not specified:
- `mode` defaults to the existing baseline behavior
- `allow_negative_baseline` defaults to `false` (clip negative values)
- `b0_prior` and `b1_prior` use default priors if not specified
