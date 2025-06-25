from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from plot_utils.paths import get_targets


def test_get_targets_default(tmp_path):
    out = tmp_path / "plot"
    targets = get_targets(None, out)
    assert list(targets.keys()) == ["png"]
    assert targets["png"].suffix == ".png"


def test_get_targets_config_formats(tmp_path):
    out = tmp_path / "plot.png"
    cfg = {"plotting": {"plot_save_formats": ["png", "pdf"]}}
    targets = get_targets(cfg, out)
    assert set(targets.keys()) == {"png", "pdf"}
    assert targets["pdf"].suffix == ".pdf"
