import sys
from pathlib import Path
import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import radon_activity
from dataclasses import asdict


def _setup(monkeypatch):
    data_dir = Path(__file__).resolve().parent / "data" / "mini_run"
    csv = data_dir / "run.csv"
    cfg = data_dir / "config.json"
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(radon_activity, "compute_radon_activity", lambda *a, **k: (-1.0, 0.5))
    return csv, cfg


def test_negative_activity_exit(tmp_path, monkeypatch):
    csv, cfg = _setup(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        analyze.main(["-i", str(csv), "-c", str(cfg), "-o", str(tmp_path)])
    assert excinfo.value.code == 1


def test_negative_activity_allowed(tmp_path, monkeypatch):
    csv, cfg = _setup(monkeypatch)
    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)

    analyze.main([
        "-i",
        str(csv),
        "-c",
        str(cfg),
        "-o",
        str(tmp_path),
        "--allow-negative-activity",
    ])

    summary = captured.get("summary", {})
    radon_res = summary["radon_results"]
    assert radon_res["radon_concentration_Bq_per_L"]["value"] < 0.0
    assert radon_res["total_radon_in_sample_Bq"]["value"] == pytest.approx(0.0)
