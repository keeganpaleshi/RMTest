# RMTest Enhancement Framework

This document describes the enhancement framework for the RMTest analysis pipeline. The framework allows you to add optional performance and diagnostic improvements without modifying core analysis code.

## Overview

The enhancement framework consists of:

1. **`enhancements.py`** - Core enhancement pipeline that wraps RMTest functionality
2. **Enhancement modules** - Optional modules that provide specific improvements
3. **Configuration** - YAML configuration to enable/disable enhancements

## Architecture

The enhancement system uses a **wrapper pattern** that:
- Works with existing RMTest code without modifications
- Gracefully falls back to standard functionality if enhancement modules are unavailable
- Can be enabled/disabled via configuration
- Tracks performance metrics and diagnostics

## Available Enhancements

### 1. Parallel Data Loading

**Module:** `parallel_loader.py` (optional)
**Configuration:** `enhancements.parallel_loading`

Speeds up loading of large CSV files (>100MB) by reading and processing chunks in parallel.

**Benefits:**
- Faster load times for large datasets (600MB+ files)
- Configurable memory limits
- Auto-detects CPU count for optimal parallelism

**Configuration:**
```yaml
enhancements:
  parallel_loading:
    enabled: true
    chunk_size: 100000      # Rows per chunk
    n_workers: null          # null = auto-detect, or specify number
    memory_limit_gb: 4.0     # Memory limit
```

### 2. Baseline Diagnostics

**Module:** `baseline_diagnostics.py` (optional)
**Configuration:** `enhancements.baseline_diagnostics`

Provides comprehensive baseline analysis including stability checks, rate analysis, and diagnostic plots.

**Benefits:**
- Baseline stability scoring
- Visual diagnostics for baseline quality
- Automated report generation
- Isotope-specific rate analysis

**Configuration:**
```yaml
enhancements:
  baseline_diagnostics:
    enabled: true
    generate_report: true    # Generate text report
```

### 3. Enhanced Radon Inference

**Module:** `radon_inference_enhanced.py` (optional)
**Configuration:** `enhancements.radon_inference`

Improves radon activity inference with deduplication, better uncertainty propagation, and multiple inference methods.

**Benefits:**
- Automatic detection and removal of duplicate measurements
- Multiple inference methods (BLUE, weighted average, single-source)
- Improved uncertainty calculations
- Better handling of ambient radon

**Configuration:**
```yaml
enhancements:
  radon_inference:
    enhanced: true
    dedupe_enabled: true     # Enable deduplication
```

### 4. Stable EMG Fitting

**Module:** `emg_stable.py` (optional)
**Configuration:** `enhancements.stable_emg`

Uses scipy special functions for more stable and accurate EMG (Exponentially Modified Gaussian) fitting.

**Benefits:**
- More numerically stable fitting
- Better handling of extreme parameters
- Log-scale computations to avoid overflow
- Improved convergence

**Configuration:**
```yaml
enhancements:
  stable_emg:
    enabled: true
    use_log_scale: true      # Use log-scale computations
```

## Usage

### Basic Usage

```python
from enhancements import EnhancementPipeline
from io_utils import load_config

# Load configuration
config = load_config('config.yaml')

# Create enhancement pipeline
pipeline = EnhancementPipeline(config)

# Use enhanced loading
events = pipeline.enhanced_load_events('data.csv')

# Use enhanced baseline diagnostics
baseline_metrics = pipeline.enhanced_baseline_diagnostics(
    events,
    output_dir='results/'
)

# Use enhanced radon inference
radon_results = pipeline.enhanced_radon_inference(
    isotope_series,
    output_dir='results/'
)

# Save metrics
pipeline.save_metrics('results/')
```

### Integration with analyze.py

You can integrate enhancements into the existing `analyze.py` without modifying it:

```python
import analyze
from enhancements import integrate_enhancements

# Integrate enhancements
integrate_enhancements(analyze)

# Now run analysis normally
# analyze.main() will use enhanced functionality if enabled in config
```

## Enhancement Module Development

To create a new enhancement module:

1. **Create the module file** (e.g., `my_enhancement.py`)
2. **Add configuration** to `config.yaml` under `enhancements:`
3. **Add wrapper method** to `EnhancementPipeline` in `enhancements.py`
4. **Implement fallback** to standard RMTest functionality

### Example Template

```python
# my_enhancement.py
"""My custom enhancement for RMTest."""

def my_enhanced_function(data, config):
    """Enhanced version of some RMTest function."""
    # Your implementation
    return result
```

```python
# In enhancements.py, add:
def enhanced_my_function(self, data, **kwargs):
    """Use enhanced version if available."""
    if self.enh_config.my_enhancement_enabled:
        try:
            from my_enhancement import my_enhanced_function
            return my_enhanced_function(data, self.config)
        except ImportError:
            logger.info("my_enhancement module not available")

    # Fallback to standard implementation
    from standard_module import my_function
    return my_function(data, **kwargs)
```

## Metrics and Diagnostics

The enhancement framework tracks:

- **Timings**: Execution time for each enhancement
- **Speedup**: Performance improvements over standard methods
- **Quality metrics**: Stability scores, deduplication counts, etc.
- **Warnings**: Issues encountered during enhancement execution

Metrics are saved to `enhancement_metrics.json` in the output directory:

```json
{
  "timings": {
    "data_loading_parallel": 2.34,
    "baseline_diagnostics": 1.23,
    "radon_inference_enhanced": 0.56
  },
  "radon_duplicates_removed": 5,
  "warnings": [],
  "enhancements_enabled": {
    "parallel_loading": true,
    "baseline_diagnostics": true,
    "radon_inference_enhanced": true,
    "stable_emg": false
  }
}
```

## Performance Guidelines

### When to Enable Parallel Loading

- File size > 100 MB: Likely beneficial
- File size > 500 MB: Highly recommended
- File size < 100 MB: Standard loading is sufficient

### Memory Considerations

- Each worker needs memory to process chunks
- Set `memory_limit_gb` to avoid system memory exhaustion
- Monitor memory usage with first run and adjust if needed

### CPU Considerations

- Set `n_workers: null` to auto-detect CPU count
- On shared systems, set explicit worker count to avoid oversubscription
- More workers ≠ always faster; optimal is typically CPU count - 1

## Troubleshooting

### Enhancement module not found

**Symptom:** Warning: "module not found"

**Solution:**
- Check that the enhancement module file exists in RMTest directory
- Verify the module can be imported: `python -c "import module_name"`
- Check for syntax errors in the module

### Enhanced functionality not being used

**Symptom:** Standard timing in metrics, no speedup

**Solution:**
- Verify enhancement is enabled in `config.yaml`
- Check log for "Enabled enhancements:" message
- Ensure conditions are met (e.g., file size for parallel loading)

### Performance worse with enhancements

**Symptom:** Slower execution with enhancement enabled

**Solution:**
- Reduce `n_workers` for parallel loading
- Increase `chunk_size` to reduce overhead
- Check system resource availability (CPU, memory)
- Some enhancements add diagnostic overhead

## Testing

To test an enhancement:

```python
import pytest
from enhancements import EnhancementPipeline

def test_enhancement():
    config = {
        'enhancements': {
            'my_enhancement': {'enabled': True}
        }
    }

    pipeline = EnhancementPipeline(config)
    # Test your enhancement
    assert pipeline.enh_config.my_enhancement_enabled
```

See `tests/test_enhancements.py` for examples.

## Future Enhancements

Potential future enhancements:

1. **GPU-accelerated fitting** - Use GPU for likelihood maximization
2. **Incremental processing** - Process data in streaming fashion
3. **Advanced deconvolution** - Improved peak deconvolution
4. **Machine learning QA** - ML-based quality assessment
5. **Distributed processing** - Process multiple files in parallel

## Contributing

To contribute a new enhancement:

1. Create the enhancement module
2. Add configuration schema
3. Add wrapper to `EnhancementPipeline`
4. Write tests
5. Update this documentation
6. Submit pull request

## References

- Main RMTest documentation: `README.md`
- Configuration guide: `config.yaml` comments
- API documentation: `docs/`

## Version History

- **v1.0** (2025-10-31): Initial enhancement framework
  - Enhancement pipeline infrastructure
  - Configuration system
  - Metrics tracking
  - Documentation

---

For questions or issues with enhancements, please open an issue on GitHub or contact the RMTest development team.
