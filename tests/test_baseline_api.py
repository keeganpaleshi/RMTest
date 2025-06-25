import numpy as np
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import baseline
import baseline_utils


def test_baseline_api_alias():
    ts = pd.date_range("2020-01-01", periods=5, freq="s", tz="UTC")
    df = pd.DataFrame({"timestamp": ts, "adc": np.arange(5)})
    bins = np.arange(0, 6)
    t0, t1 = ts[0], ts[-1]

    out1 = baseline.subtract(df, df, bins, t0, t1)
    out2 = baseline_utils.subtract(df, df, bins, t0, t1)

    for a, b in zip(out1, out2):
        pd.testing.assert_frame_equal(a, b)
