import numpy as np
import plot_utils
from plot_utils import plot_time_series


def test_default_fixed_time_bins(tmp_path, monkeypatch):
    times = np.array([100.0, 3700.0, 7400.0])
    energies = np.full_like(times, 5.3)
    cfg = {
        "window_po210": [5.2, 5.4],
        "eff_po210": [1.0],
        "plot_time_style": "lines",
    }

    captured = {}

    def fake_plot(x, y, *args, **kwargs):
        if kwargs.get("label") == "Data Po210":
            captured["n_bins"] = len(x)
        return type("obj", (), {})()

    monkeypatch.setattr(plot_utils.plt, "plot", fake_plot)
    monkeypatch.setattr(plot_utils.plt, "savefig", lambda *a, **k: None)

    t_start = 0.0
    t_end = 7500.0
    plot_time_series(times, energies, None, t_start, t_end, cfg, str(tmp_path / "ts.png"))

    expected_bins = int(np.floor((t_end - t_start) / 3600))
    assert captured.get("n_bins") == expected_bins
