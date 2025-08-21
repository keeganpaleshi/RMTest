import json
from pathlib import Path

from io_utils import Summary, write_summary
from reporting import Diagnostics
from dataclasses import asdict


def test_diagnostics_written(tmp_path):
    summary = Summary()
    summary.diagnostics = Diagnostics(
        spectral_fit_fit_valid=True,
        time_fit_po214_fit_valid=False,
        n_events_loaded=10,
        n_events_discarded=2,
        warnings=["example"]
    )

    results_dir = write_summary(tmp_path, summary)
    summary_path = Path(results_dir) / "summary.json"
    data = json.loads(summary_path.read_text())

    assert "diagnostics" in data
    diag = data["diagnostics"]
    for key in {
        "spectral_fit_fit_valid",
        "time_fit_po214_fit_valid",
        "n_events_loaded",
        "n_events_discarded",
        "warnings",
    }:
        assert key in diag


def test_missing_diagnostics_are_injected(tmp_path):
    results_dir = write_summary(tmp_path, {})
    summary_path = Path(results_dir) / "summary.json"
    data = json.loads(summary_path.read_text())

    assert "diagnostics" in data
    assert data["diagnostics"] == asdict(Diagnostics())
