import color_schemes as cs
import matplotlib.pyplot as plt

def test_po214_color_defined():
    assert hasattr(cs, "COLOR_SCHEMES") and "Po214" in cs.COLOR_SCHEMES["default"]


def test_apply_palette_sets_cycle():
    original = plt.rcParams["axes.prop_cycle"]
    try:
        palette = cs.apply_palette("colorblind")
        cycle_colors = plt.rcParams["axes.prop_cycle"].by_key().get("color", [])
        assert list(palette.values()) == cycle_colors
    finally:
        plt.rcParams["axes.prop_cycle"] = original
