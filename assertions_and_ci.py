from __future__ import annotations

from dataclasses import is_dataclass
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

    def _const_field(obj: Any, field: str) -> float:
        if isinstance(obj, Mapping):
            return float(obj[field])
        if is_dataclass(obj) or hasattr(obj, field):
            return float(getattr(obj, field))
        raise TypeError(f"Unsupported constant container for field {field!r}")

    po214 = _const_field(constants["Po214"], "half_life_s")
    assert po214 < 1e3

    baseline_cfg = config.get("baseline") if isinstance(config, Mapping) else None
    if baseline_cfg:
        assert float(baseline_cfg.get("sample_volume_l", 0.0)) >= 0
