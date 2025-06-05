import numpy as np
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_utils import plot_time_series, plot_spectrum


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


def test_plot_time_series_auto_fd(tmp_path):
    # 100 uniform events over 5 seconds
    times = 1000.0 + np.linspace(0, 5, 100)
    energies = np.full(100, 7.7)
    cfg = basic_config()
    cfg.update({
        "plot_time_binning_mode": "AUTO",
        "dump_time_series_json": True,
    })
    out_png = tmp_path / "ts_auto.png"
    plot_time_series(times, energies, None, 1000.0, 1005.0, cfg, str(out_png))
    js = out_png.with_name("ts_auto_ts.json")
    assert out_png.exists() and js.exists()

    import json
    with open(js) as f:
        data = json.load(f)

    centers = data["centers_s"]
    arr = times - 1000.0
    q25, q75 = np.percentile(arr[(arr >= 0) & (arr <= 5)], [25, 75])
    iqr = q75 - q25
    if iqr <= 0:
        expected = 1
    else:
        bw = 2 * iqr / (len(arr) ** (1.0 / 3.0))
        expected = max(1, int(np.ceil((arr.max() - arr.min()) / bw)))

    assert len(centers) == expected


def test_plot_spectrum_save_formats(tmp_path):
    energies = np.linspace(0, 10, 50)
    cfg = {"plot_save_formats": ["png", "pdf"], "plot_spectrum_binsize_adc": 1}
    out_png = tmp_path / "spec.png"
    plot_spectrum(energies, config=cfg, out_png=str(out_png))
    assert out_png.exists()
    assert out_png.with_suffix('.pdf').exists()


def test_plot_time_series_custom_half_life(tmp_path, monkeypatch):
    times = np.array([1000.1, 1000.2, 1001.1, 1001.8])
    energies = np.array([7.6, 7.7, 7.8, 7.7])
    cfg = basic_config()
    cfg["hl_Po214"] = [2.0]

    captured = {}

    def fake_plot(x, y, *args, **kwargs):
        label = kwargs.get("label")
        if label == "Model Po214":
            captured["y"] = np.array(y)
        return type("obj", (), {})()

    monkeypatch.setattr("plot_utils.plt.plot", fake_plot)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    plot_time_series(
        times,
        energies,
        {"E": 0.1, "B": 0.0, "N0": 0.0},
        1000.0,
        1002.0,
        cfg,
        str(tmp_path / "ts_custom.png"),
    )

    lam = np.log(2.0) / 2.0
    centers = np.array([0.5, 1.5])
    expected = 0.1 * (1.0 - np.exp(-lam * centers))
    assert "y" in captured
    assert np.allclose(captured["y"], expected, rtol=1e-4)


def test_plot_time_series_custom_half_life_po218(tmp_path, monkeypatch):
    times = np.array([1000.1, 1001.1, 1002.1])
    energies = np.array([5.9, 6.0, 5.8])
    cfg = basic_config()
    cfg.update({
        "window_Po218": [5.8, 6.3],
        "eff_Po218": [1.0],
        "hl_Po218": [4.0],
    })

    captured = {}

    def fake_plot(x, y, *args, **kwargs):
        if kwargs.get("label") == "Model Po218":
            captured["y"] = np.array(y)
        return type("obj", (), {})()

    monkeypatch.setattr("plot_utils.plt.plot", fake_plot)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    plot_time_series(
        times,
        energies,
        {"E": 0.1, "B": 0.0, "N0": 0.0},
        1000.0,
        1003.0,
        cfg,
        str(tmp_path / "ts_p218.png"),
    )

    lam = np.log(2.0) / 4.0
    centers = np.array([0.5, 1.5, 2.5])
    expected = 0.1 * (1.0 - np.exp(-lam * centers))
    assert np.allclose(captured.get("y"), expected, rtol=1e-4)

