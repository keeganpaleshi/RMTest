import sys
from pathlib import Path
import pytest
import numpy
import scipy

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from version_check import check_versions


def test_supported_versions(monkeypatch):
    monkeypatch.setattr(numpy, "__version__", "1.26.4")
    monkeypatch.setattr(scipy, "__version__", "1.12.0")
    # Should not raise
    check_versions()


def test_numpy_too_high(monkeypatch):
    monkeypatch.setattr(numpy, "__version__", "2.0.0")
    monkeypatch.setattr(scipy, "__version__", "1.12.0")
    with pytest.raises(RuntimeError):
        check_versions()


def test_scipy_too_high(monkeypatch):
    monkeypatch.setattr(numpy, "__version__", "1.26.4")
    monkeypatch.setattr(scipy, "__version__", "1.13.0")
    with pytest.raises(RuntimeError):
        check_versions()
