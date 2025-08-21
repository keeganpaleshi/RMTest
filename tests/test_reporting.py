import json
from pathlib import Path

from io_utils import Summary, write_summary
from reporting import Diagnostics


def test_summary_includes_diagnostics(tmp_path):
    diag = Diagnostics(
        spectral_fit_fit_valid=True,
        time_fit_po214_fit_valid=False,
        n_events_loaded=10,
        n_events_discarded=2,
        warnings=["w"],
    )
    summary = Summary(diagnostics=diag)
    results = write_summary(tmp_path, summary, "20000101T000000Z")
    data = json.load(open(Path(results) / "summary.json"))
    assert set(data["diagnostics"]) == {
        "spectral_fit_fit_valid",
        "time_fit_po214_fit_valid",
        "n_events_loaded",
        "n_events_discarded",
        "warnings",
    }
