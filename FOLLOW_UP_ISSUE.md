# Follow-up Issue: Consider clip_floor Improvements

## Title
Consider promoting `clip_floor` to a module constant for better maintainability

## Description

Following the recent centralization of density clipping (#1516), there are a couple of small improvements to consider:

### 1. Promote `clip_floor` to a module constant

Currently, the default value of `1e-300` appears in multiple locations:
- `io_utils.py` (schema default)
- `fitting.py` (fallback default)
- `src/rmtest/spectral/intensity.py` (function signature default)
- Test files
- Documentation

**Suggestion**: Consider creating a single module-level constant (e.g., `DEFAULT_CLIP_FLOOR = 1e-300`) in an appropriate location (perhaps `constants.py` or in the `intensity.py` module) that all code references. This would:
- Provide a single source of truth for docs/tests
- Make it easier to adjust the default if needed
- Improve code maintainability

### 2. Simplify fitting.py imports (optional)

The current one-try fallback pattern in `fitting.py` for importing `build_spectral_intensity` could be simplified if desired. The fallback logic is fine but could be streamlined as an optional cleanup.

### Priority

Low priority - these are minor code hygiene improvements. The current implementation works correctly.

### Related

- PR #1516: Make density clipping floor configurable and improve code hygiene
