import math

import numpy as np

from analyze import _resolve_mu_bounds


def test_resolve_mu_bounds_auto_detects_adc_and_converts():
    # Large magnitudes should be treated as ADC and converted to MeV using the calibration
    bounds_cfg = {"Po210": [8200.0, 8400.0]}
    a = 0.001
    c = 0.0
    a2 = 0.0

    resolved = _resolve_mu_bounds(bounds_cfg, unit_hint="auto", a=a, c=c, a2=a2)
    lo, hi = resolved["Po210"]

    assert math.isclose(lo, a * 8200.0 + c)
    assert math.isclose(hi, a * 8400.0 + c)


def test_resolve_mu_bounds_respects_mev_unit():
    bounds_cfg = {"Po210": [5.2, 5.4]}
    a = 0.001
    c = 0.0
    a2 = 0.0

    resolved = _resolve_mu_bounds(bounds_cfg, unit_hint="mev", a=a, c=c, a2=a2)
    assert np.allclose(resolved["Po210"], (5.2, 5.4))


def test_resolve_mu_bounds_explicit_adc_unit():
    bounds_cfg = {"Po210": [1000.0, 1100.0]}
    a = 0.002
    c = 0.5
    a2 = 0.0

    resolved = _resolve_mu_bounds(bounds_cfg, unit_hint="adc", a=a, c=c, a2=a2)
    lo, hi = resolved["Po210"]
    assert math.isclose(lo, a * 1000.0 + c)
    assert math.isclose(hi, a * 1100.0 + c)
