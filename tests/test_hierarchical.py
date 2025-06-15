import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hierarchical import fit_hierarchical_runs


def test_fit_hierarchical_runs_basic():
    runs = [
        {"half_life": 1.0, "dhalf_life": 0.1, "slope_MeV_per_ch": 0.5, "dslope": 0.1},
        {"half_life": 1.2, "dhalf_life": 0.1, "slope_MeV_per_ch": 0.55, "dslope": 0.1},
    ]

    res = fit_hierarchical_runs(runs, draws=50, tune=50, chains=1)

    assert "half_life" in res
    assert "mean" in res["half_life"]
    assert "slope_MeV_per_ch" in res
    assert "mean" in res["slope_MeV_per_ch"]
