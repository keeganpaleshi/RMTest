import json
import yaml
import pytest
import sys
from pathlib import Path
from collections.abc import Mapping

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
import io_utils

# Skip all tests when required dependencies are missing
_required = ["numpy", "scipy", "matplotlib", "pandas", "iminuit"]
for pkg in _required:
    pytest.importorskip(pkg, reason=f"Package '{pkg}' is required for tests")


# Ensure time-fit tests run with manageable statistics
_orig_load_config = io_utils.load_config


def _load_config_min_counts(path, *args, **kwargs):
    if isinstance(path, Mapping):
        raw = dict(path)
    else:
        p = Path(path)
        with open(p, "r", encoding="utf-8") as f:
            if p.suffix in {".yaml", ".yml"}:
                raw = yaml.safe_load(f) or {}
            else:
                raw = json.load(f, object_pairs_hook=io_utils._no_duplicates_object_pairs_hook)
    raw.setdefault("allow_fallback", False)
    raw.setdefault("allow_negative_baseline", False)
    raw.setdefault("allow_negative_activity", False)
    raw.setdefault("analysis_isotope", "radon")
    cfg = _orig_load_config(raw, *args, **kwargs)
    tf = cfg.setdefault("time_fit", {})
    tf.setdefault("min_counts", 0)
    return cfg


io_utils.load_config = _load_config_min_counts
analyze.load_config = _load_config_min_counts
