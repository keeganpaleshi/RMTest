#!/usr/bin/env python
"""Quick test runner for test_spectrum_tail_amplitude_stability without full conftest."""
import sys
from pathlib import Path
# Adjust path for scripts/ subdirectory
repo_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "src"))

import numpy as np
from fitting import fit_spectrum

def test_spectrum_tail_amplitude_stability():
    rng = np.random.default_rng(50)
    base = np.concatenate([
        rng.normal(5.3, 0.05, 300),
        rng.normal(6.0, 0.05, 300),
        rng.normal(7.7, 0.05, 300),
    ])
    tail = 6.0 + rng.exponential(0.15, 100)
    energies = np.concatenate([base, tail])

    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (300, 30),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (400, 40),
        "tau_Po218": (0.1, 0.05),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (300, 30),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
        "S_bkg": (0.0, 100.0),
    }

    print("Running fit_spectrum with unbinned likelihood...")
    res = fit_spectrum(energies, priors, unbinned=True)

    print(f"\nFit results:")
    print(f"  S_Po210 = {res.params.get('S_Po210', 'N/A'):.2f}")
    print(f"  S_Po218 = {res.params.get('S_Po218', 'N/A'):.2f}")
    print(f"  S_Po214 = {res.params.get('S_Po214', 'N/A'):.2f}")

    expected = 395
    actual = res.params["S_Po218"]
    error = abs(actual - expected) / expected
    tol = 0.03

    print(f"\nTest assertion:")
    print(f"  Expected: {expected}")
    print(f"  Actual:   {actual:.2f}")
    print(f"  Error:    {error*100:.2f}%")
    print(f"  Tolerance: {tol*100:.2f}%")

    if error < tol:
        print("\n✓ TEST PASSED")
        return True
    else:
        print(f"\n✗ TEST FAILED - Error {error*100:.2f}% exceeds tolerance {tol*100:.2f}%")
        return False

if __name__ == "__main__":
    try:
        passed = test_spectrum_tail_amplitude_stability()
        sys.exit(0 if passed else 1)
    except Exception as e:
        print(f"\n✗ TEST ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
