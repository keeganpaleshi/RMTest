# Color palettes for plotting
# Each scheme maps element names to explicit hex color codes.

COLOR_SCHEMES = {
    "default": {
        "Po214": "#d62728",
        "Po218": "#1f77b4",
        "Po210": "#2ca02c",
        "radon_activity": "#9467bd",
        "equivalent_air": "#2ca02c",
        "efficiency_bar": "#1f77b4",
        "fit": "#ff0000",
        "hist": "#808080",
    },
    "colorblind": {
        "Po214": "#D55E00",  # orange-red
        "Po218": "#0072B2",  # blue
        "Po210": "#009E73",  # green
        "radon_activity": "#CC79A7",
        "equivalent_air": "#009E73",
        "efficiency_bar": "#0072B2",
        "fit": "#D55E00",
        "hist": "#808080",
    },
    "grayscale": {
        "Po214": "#000000",
        "Po218": "#696969",
        "Po210": "#808080",
        "radon_activity": "#000000",
        "equivalent_air": "#808080",
        "efficiency_bar": "#696969",
        "fit": "#000000",
        "hist": "#d3d3d3",
    },
}


def apply_palette(name: str = "default") -> None:
    """Apply the given palette to ``matplotlib``.

    Parameters
    ----------
    name:
        Name of the palette to apply. If the palette does not exist the
        ``default`` palette is used.
    """

    from cycler import cycler
    from matplotlib import pyplot as plt
    from matplotlib.colors import to_hex

    palette = COLOR_SCHEMES.get(name, COLOR_SCHEMES["default"])
    colors = [to_hex(c) for c in palette.values()]
    plt.rcParams["axes.prop_cycle"] = cycler(color=colors)

