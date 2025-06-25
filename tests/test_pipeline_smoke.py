import json
from pathlib import Path
import matplotlib.pyplot as plt
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze


def test_pipeline_smoke(tmp_path, monkeypatch):
    data_dir = Path(__file__).resolve().parent / "data" / "mini_run"
    csv = data_dir / "run.csv"
    cfg = data_dir / "config.json"

    monkeypatch.setattr(plt, "savefig", lambda *a, **k: None)

    analyze.main(["-i", str(csv), "-c", str(cfg), "-o", str(tmp_path)])

    summary_files = list(tmp_path.glob("*/summary.json"))
    assert len(summary_files) == 1
    with open(summary_files[0]) as f:
        json.load(f)
