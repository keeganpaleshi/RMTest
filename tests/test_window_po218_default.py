from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils import load_config


def test_window_po218_default():
    cfg_path = Path(__file__).resolve().parents[1] / "config.yaml"
    loaded = load_config(cfg_path)
    assert loaded["time_fit"]["window_po218"] == [5.90, 6.10]
