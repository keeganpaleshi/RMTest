# Changelog

All notable changes to this project will be documented in this file.

## [Unreleased]

### Breaking/Behavioral Changes

- **Clipping is centralized**: Density clipping now occurs in a single location (`build_spectral_intensity`) using the configurable `clip_floor` parameter (default: `1e-300`). This ensures consistent numerical safety across the codebase.
- **Extended integrals remain unclipped**: The extended Poisson term integrals are computed on the unclipped model to preserve statistical correctness. Only per-energy PDF evaluations used in the likelihood are clipped to avoid `log(0)`.

### Added

- New `spectral_fit.clip_floor` configuration parameter to control the minimum density floor for numerical stability (default: `1e-300`, valid range: `(0, 1e-6]`)
- Added `clip_floor` to example YAML configurations for better discoverability

### Changed

- Improved code hygiene around density clipping implementation
