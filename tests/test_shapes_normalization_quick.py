"""Quick sanity check for shape function normalization.

Run this before the full test suite to verify EMG/Gaussian integrals are ~1.
"""

import numpy as np
import pytest


def test_gaussian_scalar_sigma_normalization():
    """Gaussian with scalar sigma should integrate to 1."""
    from rmtest.spectral.shapes import gaussian_pdf_E

    E = np.linspace(0, 15, 20001)
    f = gaussian_pdf_E(E, mu=5.3, sigma=0.134)

    integral = np.trapz(f, E)
    assert np.isclose(integral, 1.0, rtol=1e-4, atol=1e-5), \
        f"Gaussian integral should be ~1, got {integral}"


def test_gaussian_vector_sigma_normalization():
    """Gaussian with vector sigma (energy-dependent) should integrate properly."""
    from rmtest.spectral.shapes import gaussian_pdf_E

    E = np.linspace(4.0, 10.0, 10001)
    # Energy-dependent resolution
    sigma0 = 0.134
    F = 0.01
    sigma_E = np.sqrt(sigma0**2 + F * E)

    f = gaussian_pdf_E(E, mu=5.3, sigma=sigma_E)

    # For vector sigma, the PDF is not exactly normalized to 1 over all E
    # because sigma varies with E. But it should still be finite and reasonable.
    assert np.all(np.isfinite(f)), "All PDF values must be finite"
    assert np.all(f >= 0), "PDF must be non-negative"
    assert f.max() > 0, "PDF should have non-zero values"


def test_emg_scalar_sigma_normalization():
    """EMG with scalar sigma should integrate to 1."""
    from rmtest.spectral.shapes import emg_pdf_E

    E = np.linspace(0, 15, 20001)
    f = emg_pdf_E(E, mu=5.3, sigma=0.134, tau=0.02)

    integral = np.trapz(f, E)
    print(f"EMG integral: {integral:.6e} (should be ~1.0)")

    assert np.isclose(integral, 1.0, rtol=1e-3, atol=1e-4), \
        f"EMG integral should be ~1, got {integral:.6e}. If ~9e10, formula is wrong!"


def test_emg_various_parameters():
    """Test EMG normalization across different parameter combinations."""
    from rmtest.spectral.shapes import emg_pdf_E

    E = np.linspace(0, 20, 30001)

    test_cases = [
        {"mu": 5.3, "sigma": 0.134, "tau": 0.02, "name": "Po210-like"},
        {"mu": 7.687, "sigma": 0.15, "tau": 0.05, "name": "Po214-like"},
        {"mu": 6.0, "sigma": 0.1, "tau": 0.01, "name": "Small tau"},
    ]

    for case in test_cases:
        f = emg_pdf_E(E, **{k: v for k, v in case.items() if k != "name"})
        integral = np.trapz(f, E)
        print(f"EMG {case['name']}: integral = {integral:.6e}")

        assert np.isclose(integral, 1.0, rtol=1e-3, atol=1e-4), \
            f"EMG {case['name']} integral should be ~1, got {integral:.6e}"


def test_emg_vector_sigma():
    """EMG with energy-dependent sigma should be finite and positive."""
    from rmtest.spectral.shapes import emg_pdf_E

    E = np.linspace(4.0, 10.0, 5001)
    sigma0 = 0.134
    F = 0.01
    sigma_E = np.sqrt(sigma0**2 + F * E)

    f = emg_pdf_E(E, mu=5.3, sigma=sigma_E, tau=0.02)

    assert np.all(np.isfinite(f)), "All EMG PDF values must be finite"
    assert np.all(f >= 0), "EMG PDF must be non-negative"
    assert f.max() > 0, "EMG PDF should have non-zero values"


def test_shapes_reject_invalid_parameters():
    """Shape functions should return zeros for invalid parameters."""
    from rmtest.spectral.shapes import gaussian_pdf_E, emg_pdf_E

    E = np.linspace(4, 8, 1000)

    # Zero sigma
    f_gauss_zero = gaussian_pdf_E(E, mu=5.0, sigma=0.0)
    assert np.allclose(f_gauss_zero, 0.0), "Gaussian with sigma=0 should return zeros"

    # Negative sigma
    f_gauss_neg = gaussian_pdf_E(E, mu=5.0, sigma=-0.1)
    assert np.allclose(f_gauss_neg, 0.0), "Gaussian with sigma<0 should return zeros"

    # Vector sigma with some zeros
    sigma_vec = np.ones_like(E) * 0.1
    sigma_vec[500] = 0.0  # One bad value
    f_gauss_vec = gaussian_pdf_E(E, mu=5.0, sigma=sigma_vec)
    assert np.allclose(f_gauss_vec, 0.0), "Gaussian with any sigma<=0 should return zeros"

    # EMG with zero tau
    f_emg_zero_tau = emg_pdf_E(E, mu=5.0, sigma=0.1, tau=0.0)
    assert np.allclose(f_emg_zero_tau, 0.0), "EMG with tau=0 should return zeros"

    # EMG with negative tau
    f_emg_neg_tau = emg_pdf_E(E, mu=5.0, sigma=0.1, tau=-0.01)
    assert np.allclose(f_emg_neg_tau, 0.0), "EMG with tau<0 should return zeros"


if __name__ == "__main__":
    print("Running quick shape normalization tests...")
    print()

    test_gaussian_scalar_sigma_normalization()
    print("✓ Gaussian (scalar sigma) normalization")

    test_gaussian_vector_sigma_normalization()
    print("✓ Gaussian (vector sigma) finite/positive")

    test_emg_scalar_sigma_normalization()
    print("✓ EMG (scalar sigma) normalization")

    test_emg_various_parameters()
    print("✓ EMG various parameters")

    test_emg_vector_sigma()
    print("✓ EMG (vector sigma) finite/positive")

    test_shapes_reject_invalid_parameters()
    print("✓ Invalid parameters handled")

    print()
    print("All quick tests passed!")
