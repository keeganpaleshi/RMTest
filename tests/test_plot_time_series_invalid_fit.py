import numpy as np
from pathlib import Path
import plot_utils
from plot_utils import plot_time_series


def test_plot_time_series_skips_invalid_fit(tmp_path, monkeypatch):
    times = np.array([1000.1, 1000.2, 1000.3])
    energies = np.array([7.7, 7.75, 7.7])
    cfg = {"window_po214": [7.6, 7.8], "eff_po214": [1.0]}

    labels = []
    orig_plot = plot_utils.plt.plot

    def wrapped_plot(*args, **kwargs):
        lbl = kwargs.get("label")
        if lbl:
            labels.append(lbl)
        return orig_plot(*args, **kwargs)

    monkeypatch.setattr(plot_utils.plt, "plot", wrapped_plot)

    fit_res = {
        "E_Po214": 1.0,
        "B_Po214": 0.0,
        "N0_Po214": 0.0,
        "fit_valid": False,
    }

    out_png = tmp_path / "po214.png"
    plot_time_series(times, energies, fit_res, 1000.0, 1001.0, cfg, str(out_png))

    assert out_png.exists()
    assert "Model Po214" not in labels
