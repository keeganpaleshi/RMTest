# constants.py
"""Shared constants for analysis modules."""

import numpy as np

# Minimum allowed value for the exponential tail constant used in EMG fits.
_TAU_MIN = 1e-6

# Thresholds shared across the analysis modules
# Maximum exponent before ``exp`` overflows a IEEE-754 double
EXP_OVERFLOW_DOUBLE = 700.0
# Default ADC threshold for the optional noise cut
DEFAULT_NOISE_CUTOFF = 400
# Iteration cap for ``scipy.optimize.curve_fit``
CURVE_FIT_MAX_EVALS = 10000

# Limit for stable exponentiation when evaluating EMG tails or similar
# likelihood terms. Values beyond roughly ``±700`` overflow in IEEE-754
# doubles, so clip the exponent to this range.
_EXP_LIMIT = EXP_OVERFLOW_DOUBLE


def _safe_exp(x: np.ndarray) -> np.ndarray:
    """Return ``exp(x)`` with the input clipped to ``[-_EXP_LIMIT, _EXP_LIMIT]``."""
    return np.exp(np.clip(x, -_EXP_LIMIT, _EXP_LIMIT))

# Nominal ADC centroids for the Po-210, Po-218 and Po-214 peaks used
# when calibration data does not specify otherwise.
DEFAULT_NOMINAL_ADC = {
    "Po210": 1250,
    "Po218": 1400,
    "Po214": 1800,
}

# Default ADC centroids used when the configuration does not
# specify ``spectral_fit.expected_peaks``.
DEFAULT_ADC_CENTROIDS = DEFAULT_NOMINAL_ADC.copy()

from dataclasses import dataclass


@dataclass(frozen=True)
class NuclideConst:
    """Basic constants for a radioactive nuclide."""

    half_life_s: float
    Q_value_MeV: float | None = None


PO214 = NuclideConst(half_life_s=1.64e-4)
PO218 = NuclideConst(half_life_s=183.0)
PO210 = NuclideConst(half_life_s=138.376 * 24 * 3600)
RN222 = NuclideConst(half_life_s=3.8 * 86400.0)


_NUCLIDE_DEFAULTS = {
    "Po214": PO214,
    "Po218": PO218,
    "Po210": PO210,
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
    tf = cfg.get("time_fit", {}) if isinstance(cfg, dict) else {}

    result: dict[str, NuclideConst] = {}
    for name, const in _NUCLIDE_DEFAULTS.items():
        override = section.get(name, {}) if isinstance(section, dict) else {}
        hl = override.get("half_life_s", const.half_life_s)
        tf_key = f"hl_{name.lower()}"
        if isinstance(tf, dict) and tf_key in tf:
            val = tf[tf_key]
            if isinstance(val, list):
                if val:
                    hl = val[0]
            else:
                hl = val
        qv = override.get("Q_value_MeV", const.Q_value_MeV)
        result[name] = NuclideConst(half_life_s=float(hl), Q_value_MeV=qv)

    return result


def load_half_life_overrides(cfg: dict | None) -> dict[str, float]:
    """Return half-life values with optional overrides from ``cfg``."""

    consts = load_nuclide_overrides(cfg)
    return {name: nc.half_life_s for name, nc in consts.items()}

__all__ = [
    "_TAU_MIN",
    "EXP_OVERFLOW_DOUBLE",
    "DEFAULT_NOISE_CUTOFF",
    "CURVE_FIT_MAX_EVALS",
    "DEFAULT_NOMINAL_ADC",
    "DEFAULT_ADC_CENTROIDS",
    "_safe_exp",
    "NuclideConst",
    "PO214",
    "PO218",
    "PO210",
    "RN222",
    "load_nuclide_overrides",
    "load_half_life_overrides",
]
