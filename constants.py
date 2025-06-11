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

__all__ = [
    "_TAU_MIN",
    "EXP_OVERFLOW_DOUBLE",
    "DEFAULT_NOISE_CUTOFF",
    "CURVE_FIT_MAX_EVALS",
]
