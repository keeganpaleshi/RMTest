import json
import pytest
import pathlib
import math

summary = json.load(open(pathlib.Path("results/summary.json")))

def test_radon_present():
    assert "radon" in summary and summary["radon"]["Rn_activity_Bq"] >= 0
    assert math.isfinite(summary["radon"]["stat_unc_Bq"])
