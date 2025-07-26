import numpy as np

import analyze
from fitting import FitResult, FitParams

def test_spectral_refit_narrows_bounds(monkeypatch):
    calls = []
    def fake_fit_spectrum(*args, **kwargs):
        calls.append(kwargs.get("bounds"))
        if len(calls) == 1:
            return FitResult(FitParams({"mu_Po210": 5.55, "mu_Po218": 6.0, "mu_Po214": 7.7}), np.zeros((3,3)), 0)
        return FitResult(FitParams({"mu_Po210": 5.30, "mu_Po218": 6.0, "mu_Po214": 7.7}), np.zeros((3,3)), 0)

    monkeypatch.setattr(analyze, "fit_spectrum", fake_fit_spectrum)

    energies = np.array([5.55, 6.0, 7.7])
    priors = {
        "sigma0": (0.05, 0.01),
        "F": (0.0, 0.01),
        "mu_Po210": (5.3, 0.1),
        "S_Po210": (1.0, 0.1),
        "mu_Po218": (6.0, 0.1),
        "S_Po218": (1.0, 0.1),
        "mu_Po214": (7.7, 0.1),
        "S_Po214": (1.0, 0.1),
        "b0": (0.0, 1.0),
        "b1": (0.0, 1.0),
    }
    flags = {}
    cfg = {"calibration": {"known_energies": {"Po210": 5.304, "Po218": 6.002, "Po214": 7.687}},
           "spectral_fit": {"spectral_peak_tolerance_mev": 0.2}}

    res, dev = analyze._spectral_fit_with_check(energies, priors, flags, cfg)

    assert len(calls) == 2
    assert dev["Po210"] < 0.2
    assert calls[1]["mu_Po210"] == (5.304 - 0.1, 5.304 + 0.1)
