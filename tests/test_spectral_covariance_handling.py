import math

import numpy as np

import analyze
from fitting import FitResult


def test_sigma_e_uncertainty_nan_when_covariance_missing(monkeypatch):
    """Missing covariance entries should not abort spectral post-processing."""

    params = {
        "sigma0": 1.0,
        "F": 0.5,
        "fit_valid": True,
        "mu_Po214": 5.305,
    }
    cov = np.eye(2)
    fake_result = FitResult(params, cov, 10, param_index={"sigma0": 0, "F": 1})

    original_get_cov = fake_result.get_cov

    def selective_cov(name1: str, name2: str) -> float:
        if "F" in (name1, name2):
            raise KeyError("missing covariance")
        return original_get_cov(name1, name2)

    monkeypatch.setattr(fake_result, "get_cov", selective_cov)
    monkeypatch.setattr(analyze, "fit_spectrum", lambda **_: fake_result)

    cfg = {"calibration": {"known_energies": {"Po214": params["mu_Po214"]}}}

    result, deviation = analyze._spectral_fit_with_check(
        energies=np.array([1.0, 1.2, 0.8]),
        priors={},
        flags={},
        cfg=cfg,
    )

    assert deviation["Po214"] == 0.0
    assert isinstance(result, FitResult)
    assert math.isnan(result.params["dsigma_E"])
    assert result.params["fit_valid"] is True
