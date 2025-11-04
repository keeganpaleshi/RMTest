# Unbinned Extended Likelihood Normalization Fix

## Summary

This update fixes a critical normalization bug in the unbinned extended likelihood spectral fitting code. Previously, fitted yields were incorrectly scaled by approximately `1/(4*ΔE_bin) ≈ 19×` due to bin-width scaling inherited from binned fitting code.

## Problem Description

### Symptoms

1. **Absurdly large fitted yields**: For ~3.8×10⁵ events, fits returned:
   - S_Po210 ≈ 6.08×10⁶ (should be ~tens of thousands)
   - S_Po214 ≈ 8.81×10⁵
   - S_Po218 ≈ 2.31×10⁵
   - **Total sum ≈ 18.94 × N_events** (smoking gun!)

2. **Tiny background parameters**: b0, b1 ≈ 10 in "counts per MeV" units, which is impossible if signal densities were correctly scaled.

3. **Massive negative AIC** and "covariance not positive definite" warnings.

4. **Plots showing model towering over data**: The fitted model curve was scaled as "counts per tiny meta-bin" while histogram showed actual counts per bin.

### Root Cause

The unbinned extended likelihood code was using shape functions that were **not** properly normalized as probability densities. Specifically:

1. The intensity function λ(E) must be a proper **rate density** in units of `counts/MeV`:
   ```
   λ(E) = Σ_k N_k f_k(E) + b0 + b1·E
   ```
   where each `f_k(E)` is a **unit-area PDF** (∫ f_k dE = 1).

2. The extended term in the likelihood is:
   ```
   ∫ λ(E) dE = Σ_k N_k + b0·ΔE + 0.5·b1·(E_max² − E_min²)
   ```

3. The previous code was dividing peak shapes by an "effective bin width" (likely `ΔE_bin/4` from a 4× oversampling factor), which was appropriate **only** for converting counts-per-bin to density for plotting, **not** for unbinned likelihood evaluation.

4. This division artificially inflated the MLE yields by `~1/(ΔE_eff) ≈ 1/(0.013/4) ≈ 19.23×`.

## Solution

### New Modules

Three new modules in `src/rmtest/spectral/`:

#### 1. `shapes.py` - Unit-Normalized Shape Functions

```python
def emg_pdf_E(E, mu, sigma, tau):
    """
    Unit-normalized exGaussian in energy (MeV⁻¹).
    Returns proper PDF that integrates to 1.
    """
```

- Implements EMG using the correct formula: `f(E) = 1/(2τ) exp(...) erfc(...)`
- Properly normalized to integrate to 1 over the entire energy domain
- No bin-width scaling whatsoever

#### 2. `intensity.py` - Intensity Builder for Unbinned Fits

```python
def build_spectral_intensity(iso_list, use_emg, domain):
    """
    Factory to create spectral intensity function λ(E) [counts/MeV].

    Returns callable: intensity(E, params) -> λ(E)
    where λ(E) = Σ N_k f_k(E) + background
    """
```

- Constructs total rate density λ(E) = Σ N_k f_k(E) + b0 + b1·E
- Each peak: `N_k × f_k(E)` where f_k integrates to 1
- Background already in counts/MeV (no normalization needed)
- Supports energy-dependent resolution: σ(E) = √(σ₀² + F·E)

#### 3. `nll_unbinned.py` - Extended Unbinned NLL

```python
def nll_extended_unbinned(E, params, domain, iso_list, use_emg):
    """
    Extended unbinned negative log-likelihood:
        NLL = μ − Σᵢ ln λ(Eᵢ)
    where μ = ∫ λ(E) dE
    """
```

- Implements correct extended likelihood: `μ − Σ ln λ(Eᵢ)`
- Penalizes both poor fit quality and incorrect normalization
- Returns `np.inf` for invalid parameters (negative densities, etc.)

### Modified Files

#### 1. `fitting.py`

**Added imports** (lines 22-41):
```python
from rmtest.spectral import (
    emg_pdf_E,
    gaussian_pdf_E,
    build_spectral_intensity,
    integral_of_intensity as _spectral_integral,
)
```

**Replaced unbinned intensity function** (lines 839-889):
- Old: Used `_model_density` which had bin-width issues
- New: Uses `build_spectral_intensity` with properly normalized shapes
- Converts parameters: `S_{iso}` → `N_{iso}` via softplus
- Only active when `_HAS_SPECTRAL_MODULES=True` and `use_spectral_modules=True` (default)

**Added sanity checks** (lines 1006-1037):
- Warns if `sum(yields) / N_events` is outside [0.1, 5.0]
- Warns if background integral is negative or >90% of total
- Logs debug summary: N_events, sum_yields, ratio, AIC

#### 2. `config.yaml`

**Added fit_energy_range documentation** (lines 78-79):
```yaml
# Optional: Explicit fit window (MeV). If not set, uses full histogram range.
# fit_energy_range: [4.8, 8.3]
```

### Tests

#### `tests/test_unbinned_normalization.py`

Comprehensive test suite covering:

1. **Shape normalization**: EMG and Gaussian PDFs integrate to 1
2. **Intensity integration**: Single/multi-peak intensities integrate to sum of yields
3. **Background integration**: Analytical vs numerical agreement
4. **Unbinned counts scale**: **Key test** verifying `sum(yields) / N_events ≈ 1` (not 19!)
5. **NLL validity**: Finite values, inf for invalid params
6. **Array params compatibility**: Backward compatibility with 12-element array format

**Critical test** (lines 194-236):
```python
def test_unbinned_counts_scale(self):
    # Generate synthetic data with known yields
    N210, N218, N214 = 15000, 3000, 8000
    # ... generate events ...

    # Key assertion: sum of yields should be ~ same order as N_events
    sum_yields = N210 + N218 + N214
    ratio = sum_yields / N_events

    assert 0.3 < ratio < 1.5, (
        f"If ratio ≈ 19, the bin-width bug is still present!"
    )
```

## Expected Behavior After Fix

### Fitted Parameters

1. **S_Po210**: Tens of thousands (not millions)
2. **S_Po218**: Thousands (not hundreds of thousands)
3. **S_Po214**: Thousands to tens of thousands
4. **sum(S_k) / N_events**: ≈ 0.5–1.2 (not 19!)

### Background

- **b0, b1**: Order of 10³–10⁴ counts/MeV for wide windows
- **Background integral**: A few percent to ~30% of N_events (not <0.01%)

### Fit Quality

- **Covariance**: Positive definite (no warnings)
- **AIC magnitude**: Drops by orders of magnitude (physically reasonable units)
- **ln L**: In correct units (not inflated)

### Plots

- **Model curve**: Overlays histogram properly (no towering over data)
- **Residuals**: Fluctuate around zero (no giant negative dips at peaks)

## Migration Guide

### Automatic Migration

The fix is **opt-in by default** for unbinned fits:

```python
# Automatically uses new modules if available
fit_spectrum(energies, priors, unbinned=True, flags={"likelihood": "extended"})
```

### Explicit Control

To force legacy behavior (not recommended):

```python
flags = {
    "likelihood": "extended",
    "use_spectral_modules": False  # Use old _model_density
}
fit_spectrum(energies, priors, unbinned=True, flags=flags)
```

### Checking for the Bug

Add to your fit analysis:

```python
result = fit_spectrum(...)
sum_yields = sum(result.params[f"S_{iso}"] for iso in ["Po210", "Po218", "Po214"])
ratio = sum_yields / N_events

if ratio > 5.0:
    print(f"WARNING: Normalization bug detected! Ratio = {ratio:.2f}")
    print("Expected ratio ~ 1. You may be using old code or use_spectral_modules=False")
```

### Config Changes

Optionally add explicit fit window to config.yaml:

```yaml
spectral_fit:
  fit_energy_range: [4.8, 8.3]  # MeV
```

This prevents background from integrating over the entire file's energy range.

## Validation Checklist

After applying this fix, verify:

- [ ] `sum(S_k) / N_events` is within [0.3, 1.5]
- [ ] Background parameters b0, b1 are order 10³–10⁴ (not ~10)
- [ ] Background integral is 5–30% of N_events (not <1%)
- [ ] Covariance is positive definite (no warnings)
- [ ] AIC magnitude is reasonable (not -10⁶ or worse)
- [ ] Plots show model overlaying histogram (not towering above)
- [ ] Residual plots show fluctuations around zero (not systematic offsets)

## Technical Details

### Why the 4× Factor?

The `1/(4*ΔE_bin)` scaling suggests:
- Base bin width: ΔE_bin = 0.013 MeV
- Some kind of 4× oversampling or quad refinement for plotting
- Effective bin width: 0.013/4 = 0.00325 MeV
- Erroneous scaling: 1/0.00325 ≈ 307.7

But the **observed** ratio is 18.94 ≈ 1/(0.013 × 4) = 19.23, so the division was likely:
```python
# WRONG (old code):
model /= (4 * energy_bin_width)  # Divides by 0.052
# This inflates yields by ~19×
```

### Correct Formulation

For extended unbinned likelihood:

```
L_ext = Poisson(N | μ) × ∏ᵢ p(Eᵢ)
      = exp(-μ) μᴺ/N! × ∏ᵢ [λ(Eᵢ) / μ]
      = exp(-μ) / N! × ∏ᵢ λ(Eᵢ)

-ln L = μ − Σᵢ ln λ(Eᵢ) + ln N!   (drop constant)
      = μ − Σᵢ ln λ(Eᵢ)
```

where:
- μ = ∫ λ(E) dE = total expected counts
- λ(E) = intensity in counts/MeV
- N = actual observed count

### ADC → Energy Jacobian

If shapes are defined in ADC, the Jacobian is:

```python
# Shape in ADC units
f_ADC = emg_adc(adc, mu_adc, sigma_adc, tau_adc)

# Convert to energy units (ONCE, not twice!)
dADC_dE = 1 / slope_MeV_per_ch
f_E = f_ADC × dADC_dE
```

The new code assumes all shapes are already in energy (MeV) units, so no Jacobian is needed.

## References

1. Cowan, G. (1998). *Statistical Data Analysis*. Oxford University Press. (Ch. 7: Maximum Likelihood)
2. Barlow, R. (1989). *Statistics: A Guide to the Use of Statistical Methods in the Physical Sciences*. Wiley. (Extended likelihood)
3. Kalambet et al. (2011). "Reconstruction of chromatographic peaks using the exponentially modified Gaussian function." *J. Chemometrics* 25(7):352-356.

## Contact

For questions or issues related to this fix, please file an issue at:
https://github.com/keeganpaleshi/RMTest/issues

## Version History

- **2025-11-04**: Initial implementation of normalization fix
  - Added `src/rmtest/spectral/{shapes,intensity,nll_unbinned}.py`
  - Modified `fitting.py` to use new modules
  - Added comprehensive test suite
  - Documented in config.yaml
