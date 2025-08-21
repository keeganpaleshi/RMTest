import json
from pathlib import Path

import reporting


def test_diagnostics_written(tmp_path):
    diag = reporting.Diagnostics(
        spectral_fit_fit_valid=True,
        time_fit_po214_fit_valid=False,
        n_events_loaded=10,
        n_events_discarded=2,
        warnings=["something"],
    )
    result_dir = reporting.write_summary({}, tmp_path, diag, timestamp="20000101T000000Z")
    summary = json.loads((result_dir / "summary.json").read_text())
    assert "diagnostics" in summary
    block = summary["diagnostics"]
    expected_keys = {
        "spectral_fit_fit_valid",
        "time_fit_po214_fit_valid",
        "n_events_loaded",
        "n_events_discarded",
        "warnings",
    }
    assert expected_keys <= set(block)
