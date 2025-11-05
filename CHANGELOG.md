# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [Unreleased]

### Added
- Configurable clipping floor via `spectral_fit.clip_floor` parameter
  - Default: `1e-300` for numerical safety
  - Range: `(0, 1e-6]` to prevent biasing fits
  - Applied to per-energy PDF values to avoid `log(0)` in likelihood calculations
- Three new test modules for clip_floor validation and numerical safety
- Comprehensive documentation for clip_floor in README.md
- Declared SciPy and other core dependencies in pyproject.toml

### Changed
- **Breaking/Behavioral**: Density clipping is now centralized in `rmtest.spectral.intensity`
  - Clipping applies only to per-energy densities used in likelihood calculations
  - Extended likelihood integrals remain unclipped for mathematical correctness
  - This ensures the total expected count (mu_total) matches the true integral of the model
  - Previous behavior: clipping occurred in multiple locations with hardcoded floor values
  - New behavior: single source of truth with configurable floor value

### Improved
- Import hygiene in `fitting.py`: `rmtest.spectral.shapes` is now the single source of truth for shape PDFs
  - Clean fallback to legacy `calibration` module for backwards compatibility
  - Simplified import structure with clear pragma comments

### Fixed
- Config validation order ensures custom ValueError messages are visible to tests
- Test files now use YAML format (matching loader requirements)

## Notes

For upgrade path: Existing configurations without `clip_floor` will automatically use the default `1e-300` value, maintaining backwards compatibility.
