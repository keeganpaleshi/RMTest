import pytest
from baseline_utils import summarize_baseline, BaselineError


def test_summarize_baseline_negative_raises():
    cfg = {
        "baseline": {
            "rate_Bq": {"Po214": 0.2},
            "scales": {"Po214": 1.0},
            "corrected_rate_Bq": {"Po214": -0.1},
        },
        "time_fit": {"Po214": {"E_Po214": 0.1}},
    }
    with pytest.raises(BaselineError):
        summarize_baseline(cfg, ["Po214"])


def test_summarize_baseline_allow_negative():
    cfg = {
        "allow_negative_baseline": True,
        "baseline": {
            "rate_Bq": {"Po214": 0.2},
            "scales": {"Po214": 1.0},
            "corrected_rate_Bq": {"Po214": -0.1},
        },
        "time_fit": {"Po214": {"E_Po214": 0.1}},
    }
    out = summarize_baseline(cfg, ["Po214"])
    assert out["Po214"] == pytest.approx((0.1, 0.2, -0.1))
