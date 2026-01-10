import numpy as np
import analyze
import analysis_helpers
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

    analysis_helpers._spectral_fit_with_check(energies, priors, spec_flags, cfg)

    assert captured["background_model"] == "loglin_unit"
    assert captured["likelihood"] == "extended"
