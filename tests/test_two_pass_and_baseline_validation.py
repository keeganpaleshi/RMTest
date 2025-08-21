import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze

DATA_DIR = Path(__file__).resolve().parent / "data" / "mini_run"


def _paths():
    csv = DATA_DIR / "run.csv"
    cfg = DATA_DIR / "config.yaml"
    return csv, cfg


def test_two_pass_time_fit_invoked(tmp_path, monkeypatch):
    csv, cfg = _paths()
    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    calls = {"n": 0}
    orig = analyze.two_pass_time_fit

    def wrapped(*args, **kwargs):
        calls["n"] += 1
        return orig(*args, **kwargs)

    monkeypatch.setattr(analyze, "two_pass_time_fit", wrapped)
    analyze.main(["-i", str(csv), "-c", str(cfg), "-o", str(tmp_path)])
    assert calls["n"] > 0


def test_baseline_validation_fails_before_time_fit(tmp_path, monkeypatch):
    csv, cfg_path = _paths()
    with open(cfg_path) as f:
        cfg = json.load(f)
    cfg["baseline"] = {"range": ["2020-01-02T00:00:00Z", "2020-01-01T00:00:00Z"]}
    bad_cfg = tmp_path / "cfg_bad.yaml"
    with open(bad_cfg, "w") as f:
        json.dump(cfg, f)

    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)
    called = {"n": 0}

    def fake(*args, **kwargs):
        called["n"] += 1
        return None

    monkeypatch.setattr(analyze, "two_pass_time_fit", fake)

    with pytest.raises(ValueError):
        analyze.main(["-i", str(csv), "-c", str(bad_cfg), "-o", str(tmp_path)])

    assert called["n"] == 0

