from __future__ import annotations

from typing import Mapping, Any

__all__ = ["run_assertions"]

def run_assertions(summary: Mapping[str, Any], constants: Mapping[str, Any], config: Mapping[str, Any]) -> None:
    """Validate key runtime invariants.

    Parameters
    ----------
    summary : mapping
        Summary dictionary produced by ``analyze.py``.
    constants : mapping
        Dictionary of nuclide constants.
    config : mapping
        Loaded configuration dictionary.
    """
    if "radon" in summary:
        assert summary["radon"]["Rn_activity_Bq"] >= 0
        assert summary["radon"]["stat_unc_Bq"] >= 0
    # Spectral fit should be valid when requested
    spec = summary.get("spectral_fit", {})
    if spec:
        # Valid fit and finite AIC
        assert bool(spec.get("fit_valid", True)) is True
        aic = float(spec.get("aic", 0.0))
        assert aic == aic and aic < 1e12  # finite and not astronomical

        # Non-negative background amplitude
        B = float(spec.get("S_bkg", 0.0))
        assert B >= 0.0

    assert constants["Po214"]["half_life_s"] < 1e3
    assert config["baseline"]["sample_volume_l"] >= 0
