import color_schemes as cs

def test_po214_color_defined():
    assert hasattr(cs, "COLOR_SCHEMES") and "Po214" in cs.COLOR_SCHEMES["default"]
