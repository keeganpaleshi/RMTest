import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze


def test_summary_has_diagnostics(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {
            "method": "two-point",
            "peak_prominence": 0.0,
            "peak_width": 1,
            "nominal_adc": {"Po210": 1238, "Po218": 1300, "Po214": 1800},
            "peak_search_radius": 30,
            "fit_window_adc": 20,
            "use_emg": False,
            "init_sigma_adc": 5.0,
            "init_tau_adc": 1.0,
            "sanity_tolerance_mev": 1.0,
        },
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": []},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    data_path = Path(__file__).resolve().parents[1] / "example_input.csv"

    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: None)

    from io_utils import write_summary as real_write_summary

    captured = {}

    def capture_write(out_dir, summary, timestamp=None):
        p = real_write_summary(out_dir, summary, timestamp)
        captured["dir"] = p
        return str(p)

    monkeypatch.setattr(analyze, "write_summary", capture_write)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    analyze.main()

    summary_path = Path(captured["dir"]) / "summary.json"
    data = json.loads(summary_path.read_text())
    assert "diagnostics" in data
    diag = data["diagnostics"]
    for key in [
        "spectral_fit_fit_valid",
        "time_fit_po214_fit_valid",
        "time_fit_po218_fit_valid",
        "n_events_loaded",
        "n_events_discarded",
        "selected_analysis_modes",
        "warnings",
    ]:
        assert key in diag
