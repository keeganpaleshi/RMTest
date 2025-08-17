import numpy as np
import pytest


from feature_selectors import select_background_factory


class _Opts:
    def __init__(self, background_model: str | None = None):
        self.background_model = background_model


def test_select_background_factory_linear_does_not_use_S_bkg():
    bkg = select_background_factory(_Opts(), 0.0, 1.0)
    E = np.array([0.1, 0.9])
    params = {"b0": 1.0, "b1": 0.0}
    np.testing.assert_allclose(bkg(E, params), np.array([1.0, 1.0]))


def test_select_background_factory_loglin_requires_S_bkg():
    bkg = select_background_factory(_Opts("loglin_unit"), 0.0, 1.0)
    E = np.array([0.5])
    params = {"b0": 0.0, "b1": 0.0}
    with pytest.raises(ValueError):
        bkg(E, params)
    params["S_bkg"] = 0.0
    np.testing.assert_allclose(bkg(E, params), np.full_like(E, np.log1p(np.exp(0.0))))

