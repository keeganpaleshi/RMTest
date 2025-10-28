import math
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from fitting import FitParams, FitResult
from time_fitting import two_pass_time_fit


def _fit_result(params):
    return FitResult(FitParams(params), None, 0)


def test_default_time_bin_count():
    root = Path(__file__).resolve().parents[1]
    with open(root / "config.yaml", "r") as f:
        cfg = yaml.safe_load(f)
    plot_cfg = cfg.get("plotting", {})

    t_start = 0.0
    t_end = 5 * 3600 + 123.0
    times = np.linspace(t_start, t_end, num=10)

    centers, _ = analyze._ts_bin_centers_widths(times, plot_cfg, t_start, t_end)
    expected = math.floor((t_end - t_start) / plot_cfg["plot_time_bin_width_s"])
    assert len(centers) == expected


def test_two_pass_time_fit_rejects_invalid_second_pass():
    first = _fit_result({"E_Po214": 1.0, "nll": 10.0, "fit_valid": True})
    second = _fit_result({"E_Po214": 1.5, "nll": 9.0, "fit_valid": False})

    calls = iter([first, second])

    def fake_fit(*args, **kwargs):
        return next(calls)

    out = two_pass_time_fit(
        {"Po214": [0.0, 1.0]},
        0.0,
        10.0,
        {"fix_background_b_first_pass": True},
        fit_func=fake_fit,
    )

    assert out is first


def test_two_pass_time_fit_metadata_baseline_fixed():
    first = _fit_result({"E_Po214": 1.0, "nll": 10.0, "fit_valid": True})
    second = _fit_result({"E_Po214": 1.5, "nll": 9.0, "fit_valid": False})

    calls = iter([first, second])

    def fake_fit(*args, **kwargs):
        return next(calls)

    out = two_pass_time_fit(
        {"Po214": [0.0, 1.0]},
        0.0,
        10.0,
        {"fix_background_b_first_pass": True},
        baseline_rate=0.25,
        fit_func=fake_fit,
    )

    assert out.metadata.get("background_strategy") == "baseline_fixed"
    assert out.metadata.get("baseline_rate_Bq") == 0.25


def test_two_pass_time_fit_metadata_floated():
    first = _fit_result({"E_Po214": 1.0, "nll": 12.0, "fit_valid": True})
    second = _fit_result({"E_Po214": 1.5, "nll": 10.0, "fit_valid": True})

    calls = iter([first, second])

    def fake_fit(*args, **kwargs):
        return next(calls)

    out = two_pass_time_fit(
        {"Po214": [0.0, 1.0]},
        0.0,
        10.0,
        {"fix_background_b_first_pass": True},
        baseline_rate=0.5,
        fit_func=fake_fit,
    )

    assert out.metadata.get("background_strategy") == "floated"
    assert "baseline_rate_Bq" not in out.metadata

