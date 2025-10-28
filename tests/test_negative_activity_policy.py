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
    cfg = data_dir / "config.yaml"
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    monkeypatch.setattr(radon_activity, "compute_radon_activity", lambda *a, **k: (-1.0, 0.5))
    return csv, cfg


def test_negative_activity_exit(tmp_path, monkeypatch):
    csv, cfg = _setup(monkeypatch)
    with pytest.raises(SystemExit) as excinfo:
        analyze.main(["-i", str(csv), "-c", str(cfg), "-o", str(tmp_path)])
    assert excinfo.value.code == 1


def test_negative_activity_allowed(tmp_path, monkeypatch, caplog):
    csv, cfg = _setup(monkeypatch)
    captured = {}

    def fake_write(out_dir, summary, timestamp=None):
        captured["summary"] = asdict(summary)
        d = Path(out_dir) / (timestamp or "x")
        d.mkdir(parents=True, exist_ok=True)
        return str(d)

    monkeypatch.setattr(analyze, "write_summary", fake_write)
    monkeypatch.setattr(analyze, "copy_config", lambda *a, **k: None)
    monkeypatch.setattr(
        radon_activity,
        "compute_total_radon",
        lambda *a, **k: (0.0, 0.0, -5.0, 5e-6),
    )

    caplog.set_level("WARNING")

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
    total_entry = summary["radon_results"]["total_radon_in_sample_Bq"]
    assert total_entry["value"] == pytest.approx(-1.0, abs=5e-6)
    assert total_entry["uncertainty"] == pytest.approx(5e-6)
    assert any(
        "clipped to -1.0 Bq floor" in message
        or "Negative total radon in sample reported" in message
        for message in caplog.messages
    )
