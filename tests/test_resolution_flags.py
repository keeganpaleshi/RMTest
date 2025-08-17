import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
import pytest


def test_conflict_between_float_and_fix_sigma0():
    cfg = {"float_sigma_E": True, "flags": {"fix_sigma0": True}}
    with pytest.raises(ValueError):
        analyze._resolve_spectral_flags(cfg)


def test_no_conflict_when_flag_absent():
    cfg = {"float_sigma_E": True, "flags": {}}
    flags = analyze._resolve_spectral_flags(cfg)
    assert flags == {}
