# constants.py
"""Shared constants for analysis modules."""

# Minimum allowed value for the exponential tail constant used in EMG fits.
_TAU_MIN = 1e-6

# Thresholds shared across the analysis modules
# Maximum exponent before ``exp`` overflows a IEEE-754 double
EXP_OVERFLOW_DOUBLE = 700.0
# Default ADC threshold for the optional noise cut
DEFAULT_NOISE_CUTOFF = 400
# Iteration cap for ``scipy.optimize.curve_fit``
CURVE_FIT_MAX_EVALS = 10000

from dataclasses import dataclass


@dataclass(frozen=True)
class NuclideConst:
    """Basic constants for a radioactive nuclide."""

    half_life_s: float
    Q_value_MeV: float | None = None


PO214 = NuclideConst(half_life_s=1.64e-4)
PO218 = NuclideConst(half_life_s=183.0)
RN222 = NuclideConst(half_life_s=3.8 * 86400.0)


_NUCLIDE_DEFAULTS = {
    "Po214": PO214,
    "Po218": PO218,
    "Rn222": RN222,
}


def load_nuclide_overrides(cfg: dict | None) -> dict[str, NuclideConst]:
    """Return nuclide constants with optional overrides from ``cfg``.

    The configuration may define a ``"nuclides"`` section mapping isotope
    names to ``{"half_life_s": <float>}`` dictionaries. Missing values fall
    back to :mod:`constants` defaults.
    """

    if cfg is None:
        return _NUCLIDE_DEFAULTS.copy()

    section = cfg.get("nuclides", {}) if isinstance(cfg, dict) else {}

    result: dict[str, NuclideConst] = {}
    for name, const in _NUCLIDE_DEFAULTS.items():
        override = section.get(name, {}) if isinstance(section, dict) else {}
        hl = override.get("half_life_s", const.half_life_s)
        qv = override.get("Q_value_MeV", const.Q_value_MeV)
        result[name] = NuclideConst(half_life_s=float(hl), Q_value_MeV=qv)

    return result

__all__ = [
    "_TAU_MIN",
    "EXP_OVERFLOW_DOUBLE",
    "DEFAULT_NOISE_CUTOFF",
    "CURVE_FIT_MAX_EVALS",
    "NuclideConst",
    "PO214",
    "PO218",
    "RN222",
    "load_nuclide_overrides",
]
