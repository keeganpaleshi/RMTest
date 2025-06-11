# Color palettes for plotting
# Each scheme maps element names to matplotlib color names.

COLOR_SCHEMES = {
    "default": {
        "Po214": "tab:red",
        "Po218": "tab:blue",
        "Po210": "tab:green",
        "radon_activity": "tab:purple",
        "equivalent_air": "tab:green",
        "efficiency_bar": "tab:blue",
        "fit": "red",
        "hist": "gray",
    },
    "colorblind": {
        "Po214": "#D55E00",  # orange-red
        "Po218": "#0072B2",  # blue
        "Po210": "#009E73",  # green
        "radon_activity": "#CC79A7",
        "equivalent_air": "#009E73",
        "efficiency_bar": "#0072B2",
        "fit": "#D55E00",
        "hist": "gray",
    },
    "grayscale": {
        "Po214": "black",
        "Po218": "dimgray",
        "Po210": "gray",
        "radon_activity": "black",
        "equivalent_air": "gray",
        "efficiency_bar": "dimgray",
        "fit": "black",
        "hist": "lightgray",
    },
}
