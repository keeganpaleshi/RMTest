import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils import Summary, write_summary
from reporting import diagnostics_block


def test_diagnostics_written(tmp_path):
    summary = Summary(diagnostics=diagnostics_block(
        spectral_fit_fit_valid=True,
        time_fit_po214_fit_valid=False,
        n_events_loaded=10,
        n_events_discarded=2,
        warnings=["warn"],
    ))
    result_dir = write_summary(tmp_path, summary)
    data = json.loads((Path(result_dir) / "summary.json").read_text())
    diag = data.get("diagnostics", {})
    assert diag["spectral_fit_fit_valid"] is True
    assert diag["time_fit_po214_fit_valid"] is False
    assert diag["n_events_loaded"] == 10
    assert diag["n_events_discarded"] == 2
    assert diag["warnings"] == ["warn"]
