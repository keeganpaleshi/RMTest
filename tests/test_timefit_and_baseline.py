import numpy as np
from pathlib import Path
import yaml
import plot_utils


def test_default_fixed_time_bins(tmp_path, monkeypatch):
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    plot_cfg = dict(cfg.get("time_fit", {}))
    plot_cfg.update(cfg.get("plotting", {}))
    # Remove None-valued entries and normalise scalar efficiencies to lists
    plot_cfg = {k: v for k, v in plot_cfg.items() if v is not None}
    if "eff_po214" in plot_cfg and not isinstance(plot_cfg["eff_po214"], list):
        plot_cfg["eff_po214"] = [plot_cfg["eff_po214"]]

    captured = {}

    def fake_hist(data, bins):
        captured["bins"] = np.array(bins)
        return np.zeros(len(bins) - 1), np.array(bins)

    monkeypatch.setattr(plot_utils.np, "histogram", fake_hist)
    monkeypatch.setattr(plot_utils, "get_targets", lambda cfg, out: {"png": Path(out)})
    monkeypatch.setattr(plot_utils.plt, "savefig", lambda p, dpi=300: Path(p).touch())

    times = np.array([100.0, 2000.0, 4000.0, 6000.0, 8000.0])
    energies = np.full(times.shape, 7.6)
    out_png = tmp_path / "ts.png"
    t_start, t_end = 0.0, 10000.0

    plot_utils.plot_time_series(times, energies, {}, t_start, t_end, plot_cfg, str(out_png))

    assert "bins" in captured
    expected = int(np.floor((t_end - t_start) / 3600))
    assert len(captured["bins"]) - 1 == expected
