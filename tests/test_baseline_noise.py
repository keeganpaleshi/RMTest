import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from baseline_noise import estimate_baseline_noise


def test_constant_model():
    rng = np.random.default_rng(0)
    adc = rng.uniform(0, 200, 1000)
    level, params = estimate_baseline_noise(
        adc, peak_adc=250, nbins=20, model="constant"
    )
    assert level == pytest.approx(1000 / 20, rel=0.2)
    assert "A" in params


def test_exponential_model():
    rng = np.random.default_rng(1)
    k_true = 0.02
    adc = rng.exponential(scale=1 / k_true, size=5000)
    level, params = estimate_baseline_noise(
        adc, peak_adc=400, nbins=40, model="exponential"
    )
    assert params.get("k") == pytest.approx(k_true, rel=0.3)
    assert level > 0


def test_empty_adc_values():
    level, params = estimate_baseline_noise(
        [], peak_adc=200, nbins=10, model="constant"
    )
    assert level == 0.0
    assert params == {}


def test_unknown_model_raises():
    rng = np.random.default_rng(2)
    adc = rng.uniform(0, 200, 100)
    with pytest.raises(ValueError):
        estimate_baseline_noise(adc, peak_adc=250, nbins=20, model="unknown")
