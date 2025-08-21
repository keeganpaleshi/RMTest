import json
from pathlib import Path

import matplotlib.pyplot as plt

import analyze


def test_summary_diagnostics(tmp_path, monkeypatch):
    data_dir = Path(__file__).resolve().parent / "data" / "mini_run"
    csv = data_dir / "run.csv"
    cfg = data_dir / "config.yaml"

    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)

    analyze.main(["-i", str(csv), "-c", str(cfg), "-o", str(tmp_path)])

    summary_files = list(tmp_path.glob("*/summary.json"))
    assert len(summary_files) == 1
    data = json.loads(summary_files[0].read_text())
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
