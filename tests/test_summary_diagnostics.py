import json
import sys
from pathlib import Path
import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze


def test_summary_diagnostics_block(tmp_path, monkeypatch):
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
        "spectral_fit": {"do_spectral_fit": False},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(yaml.safe_dump(cfg))

    data_path = Path(__file__).resolve().parents[1] / "example_input.csv"

    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(
        analyze, "plot_time_series", lambda *a, **k: Path(k.get("out_png", tmp_path / "x.png")).touch()
    )
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "plot_radon_activity", lambda *a, **k: Path(a[2]).touch())
    monkeypatch.setattr(analyze, "plot_radon_trend", lambda *a, **k: Path(a[2]).touch())

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

    result_dir = next(tmp_path.iterdir())
    summary_path = result_dir / "summary.json"
    data = json.loads(summary_path.read_text())
    diag = data.get("diagnostics")
    assert diag is not None
    for key in [
        "spectral_fit_fit_valid",
        "time_fit_po214_fit_valid",
        "n_events_loaded",
        "n_events_discarded",
        "selected_analysis_modes",
        "warnings",
    ]:
        assert key in diag
