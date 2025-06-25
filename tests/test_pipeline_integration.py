import json
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
from dataclasses import asdict


def test_pipeline_integration(tmp_path, monkeypatch):
    data_dir = Path(__file__).resolve().parent / "data" / "mini_run"
    csv_path = data_dir / "mini.csv"
    cfg_path = data_dir / "cfg.json"

    monkeypatch.setattr("plot_utils.plt.savefig", lambda *a, **k: None)

    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    analyze.main(["-i", str(csv_path), "-c", str(cfg_path), "-o", str(tmp_path)])

    summary = captured.get("summary", {})
    rate = summary.get("time_fit", {}).get("Po214", {}).get("E_corrected")
    assert rate is not None and rate > 0

