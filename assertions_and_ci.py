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
        assert summary["radon"]["stat_unc_Bq"] > 0
    assert constants["Po214"]["half_life_s"] < 1e3
    assert config["baseline"]["sample_volume_l"] > 0
