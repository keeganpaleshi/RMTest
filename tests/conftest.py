import pytest

# Skip all tests when required dependencies are missing
_required = ["numpy", "scipy", "matplotlib", "pandas", "iminuit"]
for pkg in _required:
    pytest.importorskip(pkg, reason=f"Package '{pkg}' is required for tests")
