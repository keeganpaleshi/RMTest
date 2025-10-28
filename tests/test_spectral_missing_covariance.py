import math
import numpy as np

import analyze
from fitting import FitResult


def test_spectral_fit_missing_covariance_retains_result(monkeypatch):
    """Sigma uncertainty should fall back to NaN without failing the fit."""

    energies = np.linspace(5.0, 6.0, 32)
    priors = {}
    flags = {}
    cfg = {"spectral_fit": {}, "calibration": {}}

    params = {"sigma0": 0.2, "F": 0.05, "fit_valid": True}
    cov = np.array([[1.0e-4]])
    fit_result = FitResult(
        params=params,
        cov=cov,
        ndf=1,
        param_index={"sigma0": 0},
        counts=len(energies),
    )

    monkeypatch.setattr(analyze, "fit_spectrum", lambda *args, **kwargs: fit_result)

    result, deviation = analyze._spectral_fit_with_check(
        energies,
        priors,
        flags,
        cfg,
    )

    assert isinstance(result, FitResult)
    assert result.params.get("fit_valid") is True
    assert deviation == {}
    assert "sigma_E" in result.params
    assert math.isnan(result.params.get("dsigma_E"))
