import numpy as np
import pandas as pd
import pytest

from baseline_utils import apply_baseline_subtraction, BaselineError


def _df_from_seconds(seconds):
    timestamps = pd.to_datetime(seconds, unit="s", utc=True)
    return pd.DataFrame({"timestamp": timestamps, "adc": np.zeros(len(seconds))})


def test_apply_baseline_subtraction_outside_range_without_fallback():
    df = _df_from_seconds([0.0, 1.0, 2.0])
    bins = np.array([0.0, 1.0])

    with pytest.raises(BaselineError, match="outside data range"):
        apply_baseline_subtraction(
            df,
            df,
            bins,
            pd.Timestamp(10, unit="s", tz="UTC"),
            pd.Timestamp(12, unit="s", tz="UTC"),
            allow_fallback=False,
        )


def test_apply_baseline_subtraction_empty_slice_without_fallback():
    df = _df_from_seconds([0.0, 1.0, 2.0])
    bins = np.array([0.0, 1.0])

    with pytest.raises(BaselineError, match="matched no events"):
        apply_baseline_subtraction(
            df,
            df,
            bins,
            pd.Timestamp(0.1, unit="s", tz="UTC"),
            pd.Timestamp(0.2, unit="s", tz="UTC"),
            allow_fallback=False,
        )


def test_apply_baseline_subtraction_allows_fallback_when_enabled():
    df = _df_from_seconds([0.0, 1.0, 2.0])
    bins = np.array([0.0, 1.0])

    out_df, hist = apply_baseline_subtraction(
        df,
        df,
        bins,
        pd.Timestamp(10, unit="s", tz="UTC"),
        pd.Timestamp(12, unit="s", tz="UTC"),
        allow_fallback=True,
    )

    assert out_df.equals(df)
    assert hist.shape == (1,)
    assert hist[0] == pytest.approx(3.0)
