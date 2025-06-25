from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from plot_utils.paths import get_targets


def test_get_targets_default_ext(tmp_path):
    targets = get_targets({}, tmp_path / "plot")
    expected = {"png": (tmp_path / "plot.png")}
    assert targets == expected


def test_get_targets_multiple_formats_root(tmp_path):
    cfg = {"plot_save_formats": ["png", "pdf"]}
    targets = get_targets(cfg, tmp_path / "sub" / "figure")
    expected = {
        "png": tmp_path / "sub" / "figure.png",
        "pdf": tmp_path / "sub" / "figure.pdf",
    }
    assert targets == expected


def test_get_targets_multiple_formats_nested(tmp_path):
    cfg = {"plotting": {"plots": {"formats": ["svg", "png"]}}}
    targets = get_targets(cfg["plotting"], tmp_path / "fig.data")
    base = tmp_path / "fig"
    expected = {
        "svg": base.with_suffix(".svg"),
        "png": base.with_suffix(".png"),
    }
    assert targets == expected
