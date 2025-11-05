"""Tests for numerical safety of density clipping."""
import numpy as np
import pytest


def test_density_never_zero_after_clip():
    """Verify spectral_intensity never returns zero after clipping, even in extreme tails."""
    # Import here to avoid issues if module structure changes
    try:
        from rmtest.spectral.intensity import build_spectral_intensity
    except ImportError:
        pytest.skip("Cannot import build_spectral_intensity")

    # Create extreme energy range far from the peak
    x = np.linspace(-1e6, 1e6, 2048)

    iso_list = ["Po210"]
    use_emg = {"Po210": False}
    domain = (-10.0, 10.0)
    clip_floor = 1e-50

    spectral_intensity = build_spectral_intensity(
        iso_list, use_emg, domain, clip_floor=clip_floor
    )

    params = {
        "mu_Po210": 0.0,
        "sigma0": 0.1,
        "S_Po210": 10.0,
        "b0": 0.0,
        "b1": 0.0,
    }

    y = spectral_intensity(x, params, domain)

    # All values should be >= clip_floor
    assert np.all(y >= clip_floor), "Some density values are below clip_floor"
    # No zeros
    assert np.all(y > 0), "Some density values are zero"


def test_density_clip_prevents_log_zero():
    """Verify clipping prevents log(0) in likelihood calculations."""
    try:
        from rmtest.spectral.intensity import build_spectral_intensity
    except ImportError:
        pytest.skip("Cannot import build_spectral_intensity")

    # Energy far from peak should still give positive density
    x = np.array([100.0, 200.0, 300.0])

    iso_list = ["Po214"]
    use_emg = {"Po214": False}
    domain = (4.8, 8.3)
    clip_floor = 1e-100

    spectral_intensity = build_spectral_intensity(
        iso_list, use_emg, domain, clip_floor=clip_floor
    )

    params = {
        "mu_Po214": 7.7,
        "sigma0": 0.05,
        "S_Po214": 300.0,
        "b0": 0.0,
        "b1": 0.0,
    }

    y = spectral_intensity(x, params, domain)

    # log should not fail
    log_y = np.log(y)
    assert np.all(np.isfinite(log_y)), "log(y) contains non-finite values"
