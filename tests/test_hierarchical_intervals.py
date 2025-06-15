import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from hierarchical import fit_hierarchical_runs


def test_credible_intervals_from_summaries(tmp_path):
    # Create two fake run directories with summary.json
    for i, hl in enumerate([10.0, 12.0]):
        d = tmp_path / f"run{i}"
        d.mkdir()
        with open(d / "summary.json", "w") as f:
            json.dump(
                {
                    "half_life": hl,
                    "dhalf_life": 1.0,
                    "calibration": {"a": [1.0 + 0.1 * i, 0.1], "c": [0.1 * i, 0.1]},
                },
                f,
            )

    run_results = []
    for p in tmp_path.glob("*/summary.json"):
        with open(p, "r", encoding="utf-8") as f:
            dat = json.load(f)
        cal = dat.get("calibration", {})
        run_results.append(
            {
                "half_life": dat.get("half_life"),
                "dhalf_life": dat.get("dhalf_life"),
                "slope": cal.get("a", [None, None])[0],
                "dslope": cal.get("a", [None, None])[1],
                "intercept": cal.get("c", [None, None])[0],
                "dintercept": cal.get("c", [None, None])[1],
            }
        )

    res = fit_hierarchical_runs(
        run_results, draws=50, tune=50, chains=1, random_seed=42
    )

    for key in ("half_life", "slope", "intercept"):
        assert key in res
        assert "hdi" in res[key]
        hdi = res[key]["hdi"]
        assert len(hdi) == 2
        assert hdi[0] < hdi[1]
