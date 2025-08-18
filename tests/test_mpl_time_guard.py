import numpy as np
import pytest
from plot_utils._time_utils import guard_mpl_times, to_mpl_times


def test_guard_mpl_times_normalizes_inputs():
    secs = [0, 3600]
    expected = to_mpl_times(secs)
    out_from_secs = guard_mpl_times(times=secs)
    out_from_mpl = guard_mpl_times(times_mpl=expected)
    np.testing.assert_allclose(out_from_secs, expected)
    np.testing.assert_allclose(out_from_mpl, expected)


def test_guard_mpl_times_rejects_aliases():
    with pytest.raises(AssertionError):
        guard_mpl_times(times_dt=[0, 1])
    with pytest.raises(ValueError):
        guard_mpl_times()
    with pytest.raises(ValueError):
        guard_mpl_times(times=[0], times_mpl=[1])
