import json, pathlib, math

def test_radon_block_present():
    summ = json.load(open(pathlib.Path("results/summary.json")))
    assert "radon" in summ, "Radon block missing in summary"
    assert math.isfinite(summ["radon"]["Rn_activity_Bq"])
    assert math.isfinite(summ["radon"]["stat_unc_Bq"])
