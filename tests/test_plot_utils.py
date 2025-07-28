import numpy as np
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from constants import PO210
from plot_utils import plot_time_series, plot_spectrum, extract_time_series


def basic_config():
    return {
        "window_po214": [7.5, 8.0],
        "eff_po214": [1.0],
        "window_po218": None,
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
    assert "widths_s" in data
    assert "eff_Po214" in data
    assert "live_time_Po214_s" in data
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


def test_plot_spectrum_po210_xlim(tmp_path):
    energies = np.linspace(0, 10, 100)
    cfg = {"window_po210": [5.2, 5.4]}
    ax = plot_spectrum(energies, config=cfg, out_png=str(tmp_path / "spec2.png"))
    assert ax.get_xlim() == (5.2, 5.4)


def test_plot_spectrum_irregular_edges_residuals(tmp_path, monkeypatch):
    edges = np.array([0.0, 1.0, 3.0, 4.0])
    energies = np.concatenate([
        np.linspace(0.05, 0.95, 10),
        np.linspace(1.1, 2.9, 20),
        np.linspace(3.05, 3.95, 10),
    ])

    captured = []

    def fake_bar(self, x, height, *args, **kwargs):
        captured.append(np.array(height))
        return type("obj", (), {})()

    import matplotlib.axes

    monkeypatch.setattr(matplotlib.axes.Axes, "bar", fake_bar)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    fit_vals = {"b0": 10.0, "b1": 0.0}
    plot_spectrum(
        energies,
        fit_vals=fit_vals,
        bin_edges=edges,
        out_png=str(tmp_path / "spec_irreg.png"),
    )

    hist, _ = np.histogram(energies, bins=edges)
    width = np.diff(edges)
    centers = edges[:-1] + width / 2.0
    model_counts = (fit_vals["b0"] + fit_vals["b1"] * centers) * width
    expected = hist - model_counts

    assert len(captured) >= 2
    np.testing.assert_allclose(captured[1], expected)


def test_plot_time_series_custom_half_life(tmp_path, monkeypatch):
    times = np.array([1000.1, 1000.2, 1001.1, 1001.8])
    energies = np.array([7.6, 7.7, 7.8, 7.7])
    cfg = basic_config()
    cfg["hl_po214"] = [2.0]

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
        "window_po218": [5.8, 6.3],
        "eff_po218": [1.0],
        "hl_po218": [4.0],
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


def test_plot_time_series_time_fit_half_lives(tmp_path, monkeypatch):
    times = np.array([1000.1, 1000.9, 1001.1, 1001.9])
    energies = np.array([7.6, 5.9, 7.8, 6.0])
    cfg = basic_config()
    cfg.update({
        "window_po218": [5.8, 6.3],
        "eff_po218": [1.0],
        "time_fit": {"hl_po214": [2.0], "hl_po218": [4.0]},
    })

    captured = {}

    def fake_plot(x, y, *args, **kwargs):
        lbl = kwargs.get("label")
        if lbl == "Model Po214":
            captured["Po214"] = np.array(y)
        elif lbl == "Model Po218":
            captured["Po218"] = np.array(y)
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
        str(tmp_path / "ts_nested.png"),
    )

    centers = np.array([0.5, 1.5, 2.5])
    lam214 = np.log(2.0) / 2.0
    lam218 = np.log(2.0) / 4.0
    exp214 = 0.1 * (1.0 - np.exp(-lam214 * centers))
    exp218 = 0.1 * (1.0 - np.exp(-lam218 * centers))
    assert np.allclose(captured.get("Po214"), exp214, rtol=1e-4)
    assert np.allclose(captured.get("Po218"), exp218, rtol=1e-4)


def test_plot_time_series_invalid_half_life_po214(tmp_path):
    cfg = basic_config()
    cfg["hl_po214"] = [0.0]
    with pytest.raises(ValueError):
        plot_time_series(
            np.array([1000.1]),
            np.array([7.7]),
            None,
            1000.0,
            1001.0,
            cfg,
            str(tmp_path / "ts_bad214.png"),
        )


def test_plot_time_series_invalid_half_life_po218(tmp_path):
    cfg = basic_config()
    cfg.update({"window_po218": [5.8, 6.3], "eff_po218": [1.0], "hl_po218": [-2.0]})
    with pytest.raises(ValueError):
        plot_time_series(
            np.array([1000.1]),
            np.array([6.0]),
            None,
            1000.0,
            1001.0,
            cfg,
            str(tmp_path / "ts_bad218.png"),
        )


def test_plot_time_series_line_style(tmp_path, monkeypatch):
    times = np.array([1000.2, 1000.8])
    energies = np.array([7.7, 7.8])
    cfg = basic_config()
    cfg["plot_time_style"] = "lines"

    called = {}

    def fake_plot(*args, **kwargs):
        called["plot"] = True
        return type("obj", (), {})()

    def fake_step(*args, **kwargs):
        called["step"] = True
        return type("obj", (), {})()

    monkeypatch.setattr("plot_utils.plt.plot", fake_plot)
    monkeypatch.setattr("plot_utils.plt.step", fake_step)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    plot_time_series(
        times,
        energies,
        None,
        1000.0,
        1001.0,
        cfg,
        str(tmp_path / "ts_lines.png"),
    )

    assert called.get("plot") and "step" not in called


def test_plot_time_series_po210_no_model(tmp_path, monkeypatch):
    times = np.array([1000.1, 1000.2])
    energies = np.array([5.3, 5.25])
    cfg = basic_config()
    cfg.update({"window_po210": [5.2, 5.4], "eff_po210": [1.0]})

    labels = []

    def fake_plot(*args, **kwargs):
        lbl = kwargs.get("label")
        if lbl:
            labels.append(lbl)
        return type("obj", (), {})()

    monkeypatch.setattr("plot_utils.plt.plot", fake_plot)
    monkeypatch.setattr("plot_utils.plt.step", fake_plot)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    plot_time_series(
        times,
        energies,
        None,
        1000.0,
        1001.0,
        cfg,
        str(tmp_path / "ts_p210.png"),
    )

    assert "Model Po210" not in labels


def test_plot_time_series_po210_default_half_life(tmp_path, monkeypatch):
    times = np.array([1000.1, 1001.1, 1001.9])
    energies = np.array([5.3, 5.3, 5.3])
    cfg = basic_config()
    cfg.update({"window_po210": [5.2, 5.4], "eff_po210": [1.0]})

    captured = {}

    def fake_plot(x, y, *args, **kwargs):
        if kwargs.get("label") == "Model Po210":
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
        str(tmp_path / "ts_p210_def.png"),
    )

    lam = np.log(2.0) / PO210.half_life_s
    centers = np.array([0.5, 1.5])
    expected = 0.1 * (1.0 - np.exp(-lam * centers))
    assert "y" in captured
    assert np.allclose(captured["y"], expected, rtol=1e-4)


def test_plot_time_series_rate_normalisation(tmp_path, monkeypatch):
    times = np.array(
        [1000.5, 1001.0, 1002.1, 1002.9, 1004.0]
    )
    energies = np.full_like(times, 7.7)
    cfg = basic_config()
    cfg.update(
        {
            "plot_time_normalise_rate": True,
            "plot_time_binning_mode": "fixed",
            "plot_time_bin_width_s": 2.0,
        }
    )

    captured = {}

    def fake_step(x, y, *args, **kwargs):
        captured["y"] = np.array(y)
        return type("obj", (), {})()

    monkeypatch.setattr("plot_utils.plt.step", fake_step)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    plot_time_series(
        times,
        energies,
        None,
        1000.0,
        1006.0,
        cfg,
        str(tmp_path / "ts_norm.png"),
    )

    dt = cfg["plot_time_bin_width_s"]
    n_bins = int(np.floor((1006.0 - 1000.0) / dt))
    edges = np.arange(0, (n_bins + 1) * dt, dt, dtype=float)
    counts, _ = np.histogram(times - 1000.0, bins=edges)
    expected = counts / dt

    assert "y" in captured
    assert np.allclose(captured["y"], expected)


def test_plot_radon_activity_output(tmp_path):
    times = [0.0, 1.0, 2.0]
    activity = [1.0, 2.0, 3.0]
    errors = [0.1, 0.2, 0.3]
    out_png = tmp_path / "radon.png"

    from plot_utils import plot_radon_activity_full

    plot_radon_activity_full(times, activity, errors, str(out_png))

    assert out_png.exists()


def test_plot_equivalent_air_output(tmp_path):
    times = [0.0, 1.0, 2.0]
    volumes = [0.1, 0.2, 0.3]
    errors = [0.01, 0.02, 0.03]
    out_png = tmp_path / "air.png"

    from plot_utils import plot_equivalent_air

    plot_equivalent_air(times, volumes, errors, 5.0, str(out_png))

    assert out_png.exists()



def test_plot_radon_activity_array(tmp_path):
    times = np.array([0.0, 0.5, 1.0])
    activity = np.array([0.5, 1.0, 1.5])
    errors = np.array([0.05, 0.1, 0.15])
    out_png = tmp_path / "radon_arr.png"

    from plot_utils import plot_radon_activity_full

    plot_radon_activity_full(times, activity, errors, str(out_png))

    assert out_png.exists()


def test_plot_equivalent_air_array(tmp_path):
    times = np.array([0.0, 0.5, 1.0])
    volumes = np.array([0.05, 0.1, 0.15])
    errors = np.array([0.005, 0.01, 0.015])
    out_png = tmp_path / "air_arr.png"

    from plot_utils import plot_equivalent_air

    plot_equivalent_air(times, volumes, errors, 5.0, str(out_png))

def test_plot_equivalent_air_no_conc(tmp_path):
    times = [0.0, 1.0, 2.0]
    volumes = [1.0, 2.0, 3.0]
    errors = [0.1, 0.2, 0.3]
    out_png = tmp_path / "air_none.png"

    from plot_utils import plot_equivalent_air

    plot_equivalent_air(times, volumes, errors, None, str(out_png))


    assert out_png.exists()



def test_plot_radon_activity_time_variation(tmp_path, monkeypatch):
    times = np.array([0.0, 1.0, 2.0, 3.0])

    from radon_activity import radon_activity_curve
    activity, errors = radon_activity_curve(times, 1.0, 0.1, 2.0, 0.2, 5.0)

    captured = {}

    def fake_errorbar(x, y, *args, **kwargs):
        captured["y"] = np.array(y)
        return type("obj", (), {})()

    monkeypatch.setattr("plot_utils.plt.errorbar", fake_errorbar)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    from plot_utils import plot_radon_activity_full

    plot_radon_activity_full(times, activity, errors, str(tmp_path / "var.png"))

    assert "y" in captured
    assert not np.allclose(captured["y"], captured["y"][0])

def test_plot_radon_activity_small_array(tmp_path):
    times = np.array([0.0, 0.2, 0.4])
    activity = np.array([1.0, 1.1, 1.2])
    errors = np.array([0.1, 0.1, 0.1])
    out_png = tmp_path / "radon_small.png"

    from plot_utils import plot_radon_activity_full

    plot_radon_activity_full(times, activity, errors, str(out_png))

    assert out_png.exists()


def test_plot_equivalent_air_small_array(tmp_path):
    times = np.array([0.0, 0.5, 1.0])
    volumes = np.array([0.1, 0.15, 0.2])
    errors = np.array([0.01, 0.01, 0.02])
    out_png = tmp_path / "air_small.png"

    from plot_utils import plot_equivalent_air

    plot_equivalent_air(times, volumes, errors, 2.5, str(out_png))

    assert out_png.exists()


def test_plot_modeled_radon_activity_output(tmp_path):
    times = np.array([0.0, 1.0, 2.0])

    from plot_utils import plot_modeled_radon_activity

    out_png = tmp_path / "model.png"
    plot_modeled_radon_activity(times, 1.0, 0.1, 2.0, 0.2, 5.0, str(out_png))

    assert out_png.exists()


def test_plot_modeled_radon_activity_variation(tmp_path, monkeypatch):
    times = np.array([0.0, 1.0, 2.0, 3.0])

    captured = {}

    def fake_errorbar(x, y, *args, **kwargs):
        captured["y"] = np.asarray(y)
        return type("obj", (), {})()

    monkeypatch.setattr("plot_utils.plt.errorbar", fake_errorbar)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    from plot_utils import plot_modeled_radon_activity

    plot_modeled_radon_activity(times, 1.0, 0.1, 2.0, 0.2, 5.0, str(tmp_path / "var.png"))

    assert "y" in captured
    assert not np.allclose(captured["y"], captured["y"][0])


def test_plot_modeled_radon_activity_time_change(tmp_path, monkeypatch):
    times = np.array([0.0, 2.0, 4.0, 6.0])

    captured = {}

    def fake_errorbar(x, y, *args, **kwargs):
        captured["y"] = np.asarray(y)
        return type("obj", (), {})()

    monkeypatch.setattr("plot_utils.plt.errorbar", fake_errorbar)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    from plot_utils import plot_modeled_radon_activity

    plot_modeled_radon_activity(times, 0.5, 0.05, 1.0, 0.1, 3.0, str(tmp_path / "tc.png"))

    assert "y" in captured
    assert not np.allclose(captured["y"], captured["y"][0])
def test_plot_radon_activity_multiple_formats(tmp_path):
    times = np.array([0.0, 1.0, 2.0])
    activity = np.array([1.0, 1.1, 1.2])
    errors = np.array([0.1, 0.1, 0.1])
    out_png = tmp_path / "radon_multi.png"

    from plot_utils import plot_radon_activity_full

    plot_radon_activity_full(times, activity, errors, str(out_png), config={"plot_save_formats": ["png", "pdf"]})

    assert out_png.exists()
    assert out_png.with_suffix('.pdf').exists()


def test_plot_equivalent_air_multiple_formats(tmp_path):
    times = np.array([0.0, 0.5, 1.0])
    volumes = np.array([0.1, 0.2, 0.3])
    errors = np.array([0.01, 0.01, 0.02])
    out_png = tmp_path / "air_multi.png"

    from plot_utils import plot_equivalent_air

    plot_equivalent_air(times, volumes, errors, 1.0, str(out_png), config={"plot_save_formats": ["png", "pdf"]})

    assert out_png.exists()
    assert out_png.with_suffix('.pdf').exists()


def test_plot_radon_activity_default_extension(tmp_path):
    """Save radon activity plot when no extension is provided."""
    times = np.array([0.0, 0.5, 1.0])
    activity = np.array([1.0, 1.2, 1.4])
    errors = np.array([0.1, 0.1, 0.1])
    out_png = tmp_path / "radon_default"

    from plot_utils import plot_radon_activity_full

    plot_radon_activity_full(times, activity, errors, str(out_png))

    assert out_png.with_suffix('.png').exists()


def test_plot_equivalent_air_default_extension(tmp_path):
    """Save equivalent air plot when no extension is provided."""
    times = np.array([0.0, 0.5, 1.0])
    volumes = np.array([0.1, 0.2, 0.3])
    errors = np.array([0.01, 0.01, 0.01])
    out_png = tmp_path / "air_default"

    from plot_utils import plot_equivalent_air

    plot_equivalent_air(times, volumes, errors, 1.0, str(out_png))

    assert out_png.with_suffix('.png').exists()


def test_plot_time_series_bare_filename(tmp_path, monkeypatch):
    times = np.array([1001.0, 1002.0])
    energies = np.array([7.6, 7.8])
    monkeypatch.chdir(tmp_path)

    plot_time_series(
        times,
        energies,
        {"E": 0.1, "B": 0.0, "N0": 0.0},
        1000.0,
        1005.0,
        basic_config(),
        "bare_ts.png",
    )

    assert Path("bare_ts.png").exists()


def test_plot_spectrum_bare_filename(tmp_path, monkeypatch):
    energies = np.linspace(0, 10, 5)
    monkeypatch.chdir(tmp_path)

    plot_spectrum(energies, out_png="bare_spec.png")

    assert Path("bare_spec.png").exists()


def test_plot_radon_activity_bare_filename(tmp_path, monkeypatch):
    times = np.array([0.0, 1.0, 2.0])
    activity = np.array([1.0, 1.1, 1.2])
    errors = np.array([0.1, 0.1, 0.1])
    monkeypatch.chdir(tmp_path)

    from plot_utils import plot_radon_activity_full

    plot_radon_activity_full(times, activity, errors, "bare_radon.png")

    assert Path("bare_radon.png").exists()


def test_plot_equivalent_air_bare_filename(tmp_path, monkeypatch):
    times = np.array([0.0, 1.0, 2.0])
    volumes = np.array([0.1, 0.2, 0.3])
    errors = np.array([0.01, 0.01, 0.01])
    monkeypatch.chdir(tmp_path)

    from plot_utils import plot_equivalent_air

    plot_equivalent_air(times, volumes, errors, 1.0, "bare_air.png")

    assert Path("bare_air.png").exists()


def test_plot_radon_trend_output(tmp_path):
    from plot_utils import plot_radon_trend_full

    times = [0.0, 1.0, 2.0]
    activity = [1.0, 1.2, 1.4]
    out_png = tmp_path / "trend.png"

    plot_radon_trend_full(times, activity, str(out_png))

    assert out_png.exists()


def test_plot_radon_activity_po214(tmp_path):
    times = [0.0, 1.0, 2.0]
    activity = [1.0, 2.0, 3.0]
    errors = [0.1, 0.2, 0.3]
    out_png = tmp_path / "radon_activity_po214.png"

    from plot_utils import plot_radon_activity_full

    plot_radon_activity_full(times, activity, errors, str(out_png))

    assert out_png.exists()


def test_plot_equivalent_air_po214(tmp_path):
    times = [0.0, 1.0, 2.0]
    volumes = [0.1, 0.2, 0.3]
    errors = [0.01, 0.02, 0.03]
    out_png = tmp_path / "equivalent_air_po214.png"

    from plot_utils import plot_equivalent_air

    plot_equivalent_air(times, volumes, errors, 5.0, str(out_png))

    assert out_png.exists()


def test_extract_time_series_counts():
    times = np.array([1000.1, 1000.8, 1001.1, 1001.5, 1001.9])
    energies = np.full_like(times, 5.3)
    counts, edges = extract_time_series(times, energies, (5.2, 5.4), 1000.0, 1002.0)
    assert edges.tolist() == [0.0, 1.0, 2.0]
    assert counts.tolist() == [2, 3]


def test_extract_time_series_none_window():
    times = np.array([1000.5, 1001.5])
    energies = np.array([5.3, 5.3])
    counts, edges = extract_time_series(times, energies, None, 1000.0, 1002.0)
    assert counts.size == 0 and edges.size == 0


def test_plot_radon_activity_axis_labels(tmp_path, monkeypatch):
    times = [0.0, 1.0, 2.0]
    activity = [1.0, 2.0, 3.0]
    errors = [0.1, 0.2, 0.3]
    out_png = tmp_path / "radon_lbl.png"

    import matplotlib.pyplot as plt
    import matplotlib.axes

    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)
    monkeypatch.setattr("plot_utils.plt.close", lambda *a, **k: None)

    captured = {}
    orig_sec = matplotlib.axes.Axes.secondary_xaxis

    def wrapper(self, *args, **kwargs):
        sec = orig_sec(self, *args, **kwargs)
        captured["axis"] = sec
        return sec

    monkeypatch.setattr(matplotlib.axes.Axes, "secondary_xaxis", wrapper)

    from plot_utils import plot_radon_activity_full

    plot_radon_activity_full(times, activity, errors, str(out_png))

    ax = plt.gca()
    assert ax.get_xlabel() == "Time (UTC)"
    assert "axis" in captured
    assert captured["axis"].get_xlabel() == "Elapsed Time (s)"


def test_plot_time_series_uncertainty_band(tmp_path, monkeypatch):
    times = np.array([1000.1, 1000.6, 1001.2, 1001.8])
    energies = np.full_like(times, 7.7)
    cfg = basic_config()
    cfg.update({"plot_time_bin_width_s": 1.0})

    band_called = {}

    def fake_fill(x, y1, y2, *args, **kwargs):
        band_called["ok"] = True
        return type("obj", (), {})()

    monkeypatch.setattr("plot_utils.plt.fill_between", fake_fill)
    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    model_errs = {"Po214": np.full(3, 0.1)}

    plot_time_series(
        times,
        energies,
        {"E": 0.1, "B": 0.0, "N0": 0.0},
        1000.0,
        1003.0,
        cfg,
        str(tmp_path / "ts_band.png"),
        model_errors=model_errs,
    )

    assert band_called.get("ok")


def test_plot_time_series_datetime64(tmp_path):
    times = np.array(
        [
            np.datetime64("1970-01-01T00:00:01"),
            np.datetime64("1970-01-01T00:00:02"),
            np.datetime64("1970-01-01T00:00:03"),
        ]
    )
    energies = np.array([7.7, 7.8, 7.7])
    cfg = basic_config()
    cfg.update({"plot_time_binning_mode": "auto", "time_bins_fallback": 1})
    out_png = tmp_path / "ts_dt.png"

    plot_time_series(
        times,
        energies,
        None,
        times[0],
        times[-1] + np.timedelta64(1, "s"),
        cfg,
        str(out_png),
    )

    assert out_png.exists()


