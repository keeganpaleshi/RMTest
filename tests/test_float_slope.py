import numpy as np
import calibration


def test_float_slope_prior(monkeypatch):
    recorded = {}

    def fake_run(adc_values, cfg, hist_bins=None):
        recorded.update(cfg["calibration"]["nominal_adc"])
        return "ok"

    monkeypatch.setattr(calibration, "calibrate_run", fake_run)

    cfg = {"calibration": {"slope_MeV_per_ch": 0.00427, "float_slope": True}}

    res = calibration.derive_calibration_constants(np.array([0.0]), cfg)

    assert res == "ok"
    assert recorded["Po210"] == int(round(5.304 / 0.00427))
    assert recorded["Po218"] == int(round(6.002 / 0.00427))
    assert recorded["Po214"] == int(round(7.687 / 0.00427))
