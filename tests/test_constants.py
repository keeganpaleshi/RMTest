import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from constants import PO210, load_nuclide_overrides, load_half_life_overrides


def test_po210_default():
    assert PO210.half_life_s == pytest.approx(1.1956e7)


def test_po210_override():
    cfg = {"time_fit": {"hl_po210": [42.0]}}
    consts = load_nuclide_overrides(cfg)
    assert consts["Po210"].half_life_s == 42.0
    hl = load_half_life_overrides(cfg)
    assert hl["Po210"] == 42.0


def test_po214_override():
    cfg = {"time_fit": {"hl_po214": [0.5]}}
    consts = load_nuclide_overrides(cfg)
    assert consts["Po214"].half_life_s == 0.5
    hl = load_half_life_overrides(cfg)
    assert hl["Po214"] == 0.5


def test_po218_override():
    cfg = {"time_fit": {"hl_po218": [200.0]}}
    consts = load_nuclide_overrides(cfg)
    assert consts["Po218"].half_life_s == 200.0
    hl = load_half_life_overrides(cfg)
    assert hl["Po218"] == 200.0

