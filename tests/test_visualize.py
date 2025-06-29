import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import plot_utils  # noqa: F401 - ensure matplotlib backend is set
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


def test_cov_heatmap_bare_filename(tmp_path, monkeypatch):
    cov = np.array([[1.0, 0.5], [0.5, 1.0]])
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("visualize.plt.savefig", lambda p, **k: Path(p).touch())

    cov_heatmap(cov, "foo.png")

    assert Path("foo.png").exists()


def test_efficiency_bar_bare_filename(tmp_path, monkeypatch):
    eff = {"sources": {"A": {"value": 0.5, "error": 0.1}}}
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("visualize.plt.savefig", lambda p, **k: Path(p).touch())

    efficiency_bar(eff, "foo.png")

    assert Path("foo.png").exists()
