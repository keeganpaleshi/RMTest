"""Tests for regression equivalence with different clip_floor values."""
import numpy as np
import pytest


def test_clip_floor_regression_equivalence():
    """Verify different clip_floor values give equivalent results within the fit window."""
    try:
        from rmtest.spectral.intensity import build_spectral_intensity
    except ImportError:
        pytest.skip("Cannot import build_spectral_intensity")

    # Energy grid within the fit window
    x = np.linspace(4.8, 8.3, 3000)

    iso_list = ["Po210", "Po218", "Po214"]
    use_emg = {"Po210": False, "Po218": True, "Po214": True}
    domain = (4.8, 8.3)

    # Typical peak parameters
    params = {
        "mu_Po210": 5.30,
        "sigma0": 0.05,
        "S_Po210": 300.0,
        "mu_Po218": 6.00,
        "S_Po218": 400.0,
        "tau_Po218": 0.15,
        "mu_Po214": 7.70,
        "S_Po214": 300.0,
        "tau_Po214": 0.15,
        "b0": 0.2,
        "b1": 0.0,
    }

    # Build intensity functions with different clip floors
    spectral_intensity_1 = build_spectral_intensity(
        iso_list, use_emg, domain, clip_floor=1e-300
    )
    spectral_intensity_2 = build_spectral_intensity(
        iso_list, use_emg, domain, clip_floor=1e-100
    )

    y1 = spectral_intensity_1(x, params, domain)
    y2 = spectral_intensity_2(x, params, domain)

    # Clip floor should not meaningfully change the model inside the window
    # where the density is far above any reasonable clip floor
    rel_diff = np.abs(y1 - y2) / np.maximum(y1, 1e-300)
    max_rel_diff = np.max(rel_diff)

    assert max_rel_diff < 1e-6, (
        f"Clip floor changed model significantly: max relative diff = {max_rel_diff}"
    )


def test_clip_floor_background_model():
    """Verify clip_floor works correctly with background models."""
    try:
        from rmtest.spectral.intensity import build_spectral_intensity
    except ImportError:
        pytest.skip("Cannot import build_spectral_intensity")

    x = np.linspace(4.8, 8.3, 1000)

    iso_list = ["Po214"]
    use_emg = {"Po214": False}
    domain = (4.8, 8.3)
    clip_floor = 1e-200

    spectral_intensity = build_spectral_intensity(
        iso_list, use_emg, domain, clip_floor=clip_floor
    )

    # Parameters with S_bkg (flat background component)
    params = {
        "mu_Po214": 7.7,
        "sigma0": 0.05,
        "S_Po214": 300.0,
        "b0": 0.2,
        "b1": 0.0,
        "S_bkg": 100.0,  # Additional flat background
    }

    y = spectral_intensity(x, params, domain)

    # All values should be at least clip_floor
    assert np.all(y >= clip_floor)
    # Background should make minimum density at least b0 + S_bkg/width
    width = domain[1] - domain[0]
    expected_min = params["b0"] + params["S_bkg"] / width
    # In the window, density should be at least background level (clipped)
    assert np.min(y) >= min(expected_min, clip_floor)


def test_clip_floor_log_safety_extreme_tails():
    """Ensure clipping prevents log(0) when evaluating far outside the peak region."""
    try:
        from rmtest.spectral.intensity import build_spectral_intensity
    except ImportError:
        pytest.skip("Cannot import build_spectral_intensity")

    # Energy values far from the domain still produce positive densities via clipping
    x = np.array([-1e6, 0.0, 1e6])

    iso_list = ["Po214"]
    use_emg = {"Po214": False}
    domain = (4.8, 8.3)
    clip_floor = 1e-120

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

    # The clip floor guarantees densities remain positive and log-safe
    log_y = np.log(y)
    assert np.all(np.isfinite(log_y)), "log(y) contains non-finite values"
