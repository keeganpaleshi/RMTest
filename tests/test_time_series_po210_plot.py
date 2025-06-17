import matplotlib
matplotlib.use("Agg")

import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils import extract_time_series_events
import plot_utils
from plot_utils import plot_time_series


def test_extract_time_series_po210_count():
    df = pd.DataFrame(
        {
            "timestamp": [1.0, 2.0, 3.0, 4.0],
            "energy_MeV": [5.1, 5.3, 5.25, 5.5],
        }
    )
    cfg = {"time_fit": {"window_Po210": [5.2, 5.4]}}

    ts = extract_time_series_events(df, cfg)
    assert len(ts.get("Po210", [])) == 2


def test_plot_time_series_po210_png(tmp_path, monkeypatch):
    times = np.array([1000.1, 1000.2, 1000.3])
    energies = np.array([5.3, 5.25, 5.3])
    cfg = {"window_Po210": [5.2, 5.4], "eff_Po210": [1.0]}

    labels = []
    orig_plot = plot_utils.plt.plot

    def wrapped_plot(*args, **kwargs):
        lbl = kwargs.get("label")
        if lbl:
            labels.append(lbl)
        return orig_plot(*args, **kwargs)

    monkeypatch.setattr(plot_utils.plt, "plot", wrapped_plot)

    out_png = tmp_path / "po210.png"
    plot_time_series(times, energies, None, 1000.0, 1001.0, cfg, str(out_png))

    assert out_png.exists()
    assert "Model Po210" not in labels

