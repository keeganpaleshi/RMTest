# constants.py
"""Shared constants for analysis modules."""

# Minimum allowed value for the exponential tail constant used in EMG fits.
_TAU_MIN = 1e-6

from dataclasses import dataclass, replace
from typing import Mapping, Any


@dataclass(frozen=True)
class NuclideConst:
    """Physical constants for a nuclide."""

    half_life_s: float
    Q_value_MeV: float | None = None


# Default nuclide constants
PO214 = NuclideConst(half_life_s=1.64e-4)
PO218 = NuclideConst(half_life_s=183.0)


def load_half_life_overrides(cfg: Mapping[str, Any]) -> None:
    """Override nuclide half-lives using ``cfg['constants']`` if present."""

    global PO214, PO218
    consts = cfg.get("constants", {}) if isinstance(cfg, Mapping) else {}

    val = consts.get("hl_Po214")
    if val is not None:
        PO214 = replace(PO214, half_life_s=float(val))

    val = consts.get("hl_Po218")
    if val is not None:
        PO218 = replace(PO218, half_life_s=float(val))


__all__ = ["_TAU_MIN", "NuclideConst", "PO214", "PO218", "load_half_life_overrides"]
