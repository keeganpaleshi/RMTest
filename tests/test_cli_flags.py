import analyze
import numpy as np
from fitting import FitResult, FitParams


def test_analysis_flags_threaded(monkeypatch):
    captured = {}

    def fake_fit_spectrum(*args, **kwargs):
        captured.update(kwargs.get("flags", {}))
        params = FitParams({
            "mu_Po210": 5.3,
            "mu_Po218": 6.0,
            "mu_Po214": 7.7,
        })
        return FitResult(params, None, 0)

    monkeypatch.setattr(analyze, "fit_spectrum", fake_fit_spectrum)

    energies = np.array([5.3, 6.0, 7.7])
    priors = {}
    cfg = {
        "analysis": {"background_model": "loglin_unit", "likelihood": "extended"},
        "spectral_fit": {"flags": {}},
        "calibration": {"known_energies": {"Po210": 5.3, "Po218": 6.0, "Po214": 7.7}},
    }

    spec_flags = cfg["spectral_fit"]["flags"].copy()
    spec_flags["background_model"] = cfg["analysis"]["background_model"]
    spec_flags["likelihood"] = cfg["analysis"]["likelihood"]

    analyze._spectral_fit_with_check(energies, priors, spec_flags, cfg)

    assert captured["background_model"] == "loglin_unit"
    assert captured["likelihood"] == "extended"


def test_unbinned_likelihood_switch(monkeypatch):
    calls = []

    def fake_fit_spectrum(*args, **kwargs):
        calls.append(kwargs.get("unbinned", False))
        method = "unbinned" if kwargs.get("unbinned", False) else "binned"
        return FitResult(FitParams({}), None, 0, method=method)

    monkeypatch.setattr(analyze, "fit_spectrum", fake_fit_spectrum)

    energies = np.array([5.3, 6.0, 7.7])
    priors = {}
    cfg = {"calibration": {"known_energies": {"Po210": 5.3, "Po218": 6.0, "Po214": 7.7}}}

    res, _ = analyze._spectral_fit_with_check(energies, priors, {}, cfg, unbinned=True)
    assert calls[-1] is True
    assert res.method == "unbinned"

    res2, _ = analyze._spectral_fit_with_check(energies, priors, {}, cfg)
    assert calls[-1] is False
    assert res2.method == "binned"
