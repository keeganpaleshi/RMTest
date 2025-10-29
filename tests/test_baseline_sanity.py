import pytest
from baseline_utils import summarize_baseline


def test_summarize_baseline_negative_clipped_to_zero():
    cfg = {
        "baseline": {
            "rate_Bq": {"Po214": 0.2},
            "scales": {"Po214": 1.0},
            "corrected_rate_Bq": {"Po214": -5.0},
        },
        "time_fit": {"Po214": {"E_Po214": 0.1}},
    }
    out = summarize_baseline(cfg, ["Po214"])
    assert out["Po214"] == pytest.approx((0.1, 0.2, 0.0))


def test_summarize_baseline_allow_negative():
    cfg = {
        "allow_negative_baseline": True,
        "baseline": {
            "rate_Bq": {"Po214": 0.2},
            "scales": {"Po214": 1.0},
            "corrected_rate_Bq": {"Po214": -5.0},
        },
        "time_fit": {"Po214": {"E_Po214": 0.1, "E_corrected": -5.0}},
    }
    out = summarize_baseline(cfg, ["Po214"])
    assert out["Po214"] == pytest.approx((0.1, 0.2, -5.0))


def test_summarize_baseline_uses_corrected_rate_map_when_available():
    cfg = {
        "allow_negative_baseline": True,
        "baseline": {
            "rate_Bq": {"Po214": 0.2},
            "scales": {"Po214": 1.0},
            "corrected_rate_Bq": {"Po214": -5.0},
        },
        "time_fit": {
            "Po214": {"E_Po214": 0.1, "E_corrected": -1.0},
        },
    }
    out = summarize_baseline(cfg, ["Po214"])
    assert out["Po214"] == pytest.approx((0.1, 0.2, -5.0))
