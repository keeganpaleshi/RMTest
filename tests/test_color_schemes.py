import warnings

import matplotlib.pyplot as plt
from matplotlib.colors import to_hex
import pytest

import color_schemes as cs

def test_po214_color_defined():
    assert hasattr(cs, "COLOR_SCHEMES") and "Po214" in cs.COLOR_SCHEMES["default"]


@pytest.mark.parametrize("name", cs.COLOR_SCHEMES.keys())
def test_palette_values_are_hex(name):
    palette = cs.COLOR_SCHEMES[name]
    for value in palette.values():
        color = to_hex(value)
        assert color.startswith("#") and len(color) == 7


def test_apply_palette_no_warnings():
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("error")
        cs.apply_palette("default")
    assert not w
    cycle = plt.rcParams["axes.prop_cycle"]
    assert cycle.keys == {"color"}
