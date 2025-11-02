import pytest
import sys
from pathlib import Path

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Add src/ directory to path for rmtest package
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

import analyze
import io_utils

# Skip all tests when required dependencies are missing
_required = ["numpy", "scipy", "matplotlib", "pandas", "iminuit"]
for pkg in _required:
    pytest.importorskip(pkg, reason=f"Package '{pkg}' is required for tests")


# Ensure time-fit tests run with manageable statistics
_orig_load_config = io_utils.load_config


def _load_config_min_counts(path, *args, **kwargs):
    cfg = _orig_load_config(path, *args, **kwargs)
    tf = cfg.setdefault("time_fit", {})
    tf.setdefault("min_counts", 0)
    return cfg


io_utils.load_config = _load_config_min_counts
analyze.load_config = _load_config_min_counts
