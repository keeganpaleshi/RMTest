import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_utils import plot_time_series


def basic_config():
    return {
        "window_Po214": [7.5, 8.0],
        "eff_Po214": [1.0],
        "window_Po218": None,
        "time_bin_mode": "fixed",
        "time_bin_s": 1.0,
        "dump_time_series_json": False,
    }


def test_plot_time_series_fallback(tmp_path):
    times = np.array([1001.0, 1002.0, 1003.0])
    energies = np.array([7.7, 7.8, 7.6])
    out_png = tmp_path / "ts.png"
    plot_time_series(
        times,
        energies,
        {"E": 0.1, "B": 0.0, "N0": 0.0},
        1000.0,
        1005.0,
        basic_config(),
        str(out_png),
    )
    assert out_png.exists()


def test_plot_time_series_none_fit_results(tmp_path):
    times = np.array([1001.0, 1002.0])
    energies = np.array([7.6, 7.8])
    out_png = tmp_path / "ts2.png"
    plot_time_series(
        times,
        energies,
        None,
        1000.0,
        1005.0,
        basic_config(),
        str(out_png),
    )
    assert out_png.exists()
