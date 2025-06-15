import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from visualize import cov_heatmap, efficiency_bar


def test_cov_heatmap(tmp_path):
    cov = np.array([[1.0, 0.5], [0.5, 2.0]])
    out = tmp_path / "cov.png"
    cov_heatmap(cov, str(out), labels=["a", "b"])
    assert out.exists()


def test_efficiency_bar_annotations(tmp_path, monkeypatch):
    eff = {
        "sources": {
            "spike": {"value": 0.8, "error": 0.1},
            "assay": {"value": 0.7, "error": 0.05},
        },
        "combined": {"value": 0.75, "error": 0.05, "weights": [0.6, 0.4]},
    }

    texts = []
    monkeypatch.setattr("visualize.plt.text", lambda *a, **k: texts.append(a[2]))
    monkeypatch.setattr("visualize.plt.savefig", lambda path, **k: Path(path).touch())

    out = tmp_path / "eff.png"
    efficiency_bar(eff, str(out))
    assert out.exists()
    assert any("0.6" in t for t in texts) and any("0.4" in t for t in texts)
