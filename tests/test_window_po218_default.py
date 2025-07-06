import json
from pathlib import Path

import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from io_utils import load_config


def test_window_po218_default(tmp_path):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "spectral_fit": {},
        "time_fit": {"do_time_fit": True},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    loaded = load_config(cfg_path)
    assert loaded["time_fit"]["window_po218"] == [5.922, 6.082]
