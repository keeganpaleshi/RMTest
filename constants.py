# constants.py
"""Shared constants for analysis modules."""

import numpy as np
from dataclasses import dataclass
from pathlib import Path
import yaml

# Minimum allowed value for the exponential tail constant used in EMG fits.
_TAU_MIN = 1e-8

# Thresholds shared across the analysis modules
# Maximum exponent before ``exp`` overflows a IEEE-754 double
EXP_OVERFLOW_DOUBLE = 700.0
# Default ADC threshold for the optional noise cut
DEFAULT_NOISE_CUTOFF = 400
# Iteration cap for ``scipy.optimize.curve_fit``
CURVE_FIT_MAX_EVALS = 10000

# Floor applied when negative baseline-corrected activities are permitted.
# Values more negative than this threshold are clipped to prevent
# unphysical numbers from propagating through the reports while still
# allowing small negative fluctuations.
NEGATIVE_ACTIVITY_FLOOR_BQ = -1.0
# Minimum uncertainty reported when the negative floor is applied.  This
# acts as a sentinel to highlight that the value was clamped rather than
# inferred directly from data.
NEGATIVE_ACTIVITY_CLAMP_UNCERTAINTY_BQ = 5e-6

# Clip exponents to ``+/-EXP_OVERFLOW_DOUBLE`` to avoid floating-point overflow
# when evaluating functions with large tails (e.g. EMG).
def safe_exp(x: np.ndarray) -> np.ndarray:
    """Return ``exp(x)`` with the input clipped to ``[-EXP_OVERFLOW_DOUBLE, EXP_OVERFLOW_DOUBLE]``."""
    return np.exp(np.clip(x, -EXP_OVERFLOW_DOUBLE, EXP_OVERFLOW_DOUBLE))

# Backwards compatibility for older modules/tests
_safe_exp = safe_exp

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

# Default alpha energies (MeV) used when calibration configuration does not
# specify ``known_energies``.  Values correspond to the Po-210, Po-218 and
# Po-214 peaks.
DEFAULT_KNOWN_ENERGIES = {
    "Po210": 5.304,  # MeV
    "Po218": 6.002,  # MeV
    "Po214": 7.687,  # MeV
}


@dataclass(frozen=True)
class NuclideConst:
    """Basic constants for a radioactive nuclide."""

    half_life_s: float
    Q_value_MeV: float | None = None


def _read_yaml_defaults() -> dict:
    path = Path(__file__).resolve().with_name("constants.yaml")
    if not path.is_file():
        return {}
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    return data.get("nuclide_constants", {})


_YAML_DEFAULTS = _read_yaml_defaults()

PO214 = NuclideConst(
    half_life_s=float(_YAML_DEFAULTS.get("Po214", {}).get("half_life_s", 1.64e-4))
)
PO218 = NuclideConst(
    half_life_s=float(_YAML_DEFAULTS.get("Po218", {}).get("half_life_s", 183.0))
)
PO210 = NuclideConst(
    half_life_s=float(
        _YAML_DEFAULTS.get("Po210", {}).get("half_life_s", 138.376 * 24 * 3600)
    )
)
RN222 = NuclideConst(
    half_life_s=float(_YAML_DEFAULTS.get("Rn222", {}).get("half_life_s", 3.8 * 86400.0))
)


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
            elif val is not None:
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
    "NEGATIVE_ACTIVITY_FLOOR_BQ",
    "NEGATIVE_ACTIVITY_CLAMP_UNCERTAINTY_BQ",
    "DEFAULT_NOMINAL_ADC",
    "DEFAULT_ADC_CENTROIDS",
    "DEFAULT_KNOWN_ENERGIES",
    "safe_exp",
    "_safe_exp",
    "NuclideConst",
    "PO214",
    "PO218",
    "PO210",
    "RN222",
    "load_nuclide_overrides",
    "load_half_life_overrides",
]
