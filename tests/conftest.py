import pytest
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))


# Skip all tests when required dependencies are missing
_required = ["numpy", "scipy", "matplotlib", "pandas", "iminuit"]
for pkg in _required:
    pytest.importorskip(pkg, reason=f"Package '{pkg}' is required for tests")


