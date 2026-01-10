import pandas as pd
import numpy as np
import analyze
import analysis_helpers


def test_auto_expand_window_recovers_counts():
    df = pd.DataFrame({
        "timestamp": pd.date_range("1970-01-01", periods=5, freq="s"),
        "energy_MeV": np.full(5, 7.72),
        "denergy_MeV": np.full(5, 0.01),
    })
    events, win = analysis_helpers.auto_expand_window(df, (7.71, 7.71), threshold=5, step=0.02)
    assert len(events) == 5
    assert win[0] < 7.71 and win[1] > 7.71

