---
name: Follow-up improvements for clip_floor implementation
about: Optional code quality improvements after PR #1516
title: 'Consider clip_floor module constant and import simplification'
labels: enhancement, refactoring, low-priority
assignees: ''
---

## Summary

After PR #1516 successfully implemented configurable `clip_floor`, consider these optional code quality improvements:

## 1. Promote clip_floor to a module constant

**Current state:**
- Default value `1e-300` is duplicated in multiple locations:
  - `io_utils.py`: default in config loading
  - `src/rmtest/spectral/intensity.py`: function parameter default
  - Example YAML configs
  - Test files
  - Documentation

**Proposed:**
Create a single source of truth:

```python
# src/rmtest/spectral/constants.py (or similar)
DEFAULT_CLIP_FLOOR = 1e-300
CLIP_FLOOR_MIN = 0.0  # exclusive
CLIP_FLOOR_MAX = 1e-6
```

**Benefits:**
- Single reference point for default value
- Easier to maintain and update if needed
- Self-documenting constant names
- Reduces magic numbers in code

**Trade-offs:**
- Adds one more import for users who want to reference it
- Current approach works fine; this is purely for code hygiene

## 2. Simplify import fallback in fitting.py

**Current state:**
```python
try:
    from rmtest.spectral.shapes import emg_pdf_E as emg_left, gaussian_pdf_E as gaussian
except ImportError:
    # Fallback to legacy calibration module
    try:
        from calibration import emg_left, gaussian
    except ImportError:
        # Further fallback with sys.path manipulation
        ...
```

**Proposed:**
Simplify to single try-except if multi-level fallback is no longer needed:

```python
try:
    from rmtest.spectral.shapes import emg_pdf_E as emg_left, gaussian_pdf_E as gaussian
except ImportError:  # pragma: no cover
    from calibration import emg_left, gaussian
```

**Consider:**
- Is the sys.path fallback still necessary?
- Are there environments where the package isn't properly installed?
- Can we assume users have a proper installation?

**Benefits:**
- Cleaner, more readable code
- Fewer nested try-except blocks
- Easier to maintain

**Trade-offs:**
- May break some local development workflows
- Need to verify all supported deployment scenarios

## Priority

**Low** - Current implementation is correct and works well. These are quality-of-life improvements that can be addressed when convenient.

## Related

- PR #1516: Make density clipping floor configurable
- Original implementation discussion
