import os
import re
import shutil
import sys
import uuid
from pathlib import Path

import pytest

# Add repo root to path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

# Add src/ directory to path for rmtest package
SRC = ROOT / "src"
if SRC.is_dir():
    sys.path.insert(0, str(SRC))

TMP_BASE = ROOT / ".codex_pytest_tmp"
TMP_BASE.mkdir(exist_ok=True)
TMP_ROOT = TMP_BASE / f"pid_{os.getpid()}"
TMP_ROOT.mkdir(exist_ok=True)
PYTENSOR_BASE = TMP_ROOT / "pytensor"
PYTENSOR_COMPILEDIR = PYTENSOR_BASE / "compiledir"
PYTENSOR_COMPILEDIR.mkdir(parents=True, exist_ok=True)

# Keep temp-file consumers and PyTensor's artifacts inside the workspace rather
# than host-level cache roots, which are not reliably accessible here. Force
# the Python linker for tests so PyTensor avoids transient C build folders that
# also fail under this filesystem layer.
os.environ["TMP"] = str(TMP_ROOT)
os.environ["TEMP"] = str(TMP_ROOT)
_pytensor_flags = os.environ.get("PYTENSOR_FLAGS", "")
_extra_pytensor_flags = (
    f"base_compiledir={PYTENSOR_BASE},compiledir={PYTENSOR_COMPILEDIR},cxx=,linker=py"
)
os.environ["PYTENSOR_FLAGS"] = (
    f"{_pytensor_flags},{_extra_pytensor_flags}" if _pytensor_flags else _extra_pytensor_flags
)

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


@pytest.fixture(scope="session", autouse=True)
def _manage_repo_tmp_root():
    shutil.rmtree(TMP_ROOT, ignore_errors=True)
    TMP_ROOT.mkdir(parents=True, exist_ok=True)
    PYTENSOR_COMPILEDIR.mkdir(parents=True, exist_ok=True)
    yield
    shutil.rmtree(TMP_ROOT, ignore_errors=True)


@pytest.fixture
def tmp_path(request):
    name = re.sub(r"[^A-Za-z0-9_.-]+", "_", request.node.name).strip("._")
    path = TMP_ROOT / f"{name[:40] or 'tmp'}_{uuid.uuid4().hex[:8]}"
    path.mkdir()
    yield path
    shutil.rmtree(path, ignore_errors=True)
