#!/usr/bin/env python
"""Quick test runner for window normalization regression tests."""
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

import numpy as np
from rmtest.spectral.window_norm import normalize_pdf_to_window
from rmtest.spectral.shapes import emg_cdf_E
from rmtest.spectral.intensity import build_spectral_intensity, integral_of_intensity

def test_emg_window_mass_is_near_one_for_this_case():
    print("Test 1: EMG window mass is near one...")
    mu, sigma, tau = 6.0, 0.05, 0.10
    E_lo, E_hi = 4.8, 8.3
    pwin = float(emg_cdf_E(E_hi, mu, sigma, tau) - emg_cdf_E(E_lo, mu, sigma, tau))
    print(f"  pwin = {pwin:.6f}")
    assert 0.99999 < pwin < 1.0, f"Expected 0.99999 < {pwin} < 1.0"
    print("  ✓ PASSED")

def test_extended_mu_equals_sum_of_yields_after_window_renorm():
    print("\nTest 2: Extended μ equals sum of yields...")
    domain = (4.8, 8.3)
    iso_list = ("Po210", "Po218", "Po214")
    spectral = build_spectral_intensity(iso_list, use_emg=True, domain=domain)

    params = dict(
        N_Po210=300.0, mu_Po210=5.3,
        N_Po218=400.0, mu_Po218=6.0, tau_Po218=0.10,
        N_Po214=300.0, mu_Po214=7.7,
        sigma0=0.05, b0=0.0, b1=0.0
    )
    E = np.linspace(domain[0], domain[1], 6001)
    lam = spectral(E, params, domain)
    mu = integral_of_intensity(params, domain, iso_list=iso_list, use_emg=True)
    num_mu = np.trapz(lam, E)

    print(f"  Expected μ: {300+400+300}")
    print(f"  Analytical μ: {mu:.6f}")
    print(f"  Numerical μ: {num_mu:.6f}")
    print(f"  Relative error: {abs(num_mu - mu) / mu * 100:.4f}%")

    assert abs(mu - (300+400+300)) < 1e-6, f"Expected μ = 1000, got {mu}"
    assert abs(num_mu - mu) / mu < 1e-3, f"Relative error too large: {abs(num_mu - mu) / mu}"
    print("  ✓ PASSED")

if __name__ == "__main__":
    try:
        test_emg_window_mass_is_near_one_for_this_case()
        test_extended_mu_equals_sum_of_yields_after_window_renorm()
        print("\n✓ ALL TESTS PASSED")
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ TEST FAILED: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
