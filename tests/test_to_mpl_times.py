from datetime import datetime, timezone
import numpy as np
from plot_utils._time_utils import to_mpl_times


def test_to_mpl_times_various_inputs():
    secs = [0, 3600]
    expected = to_mpl_times(secs)
    dt64 = np.array(secs, dtype="datetime64[s]")
    naive = [datetime.utcfromtimestamp(s) for s in secs]
    aware = [datetime.fromtimestamp(s, timezone.utc) for s in secs]
    np.testing.assert_allclose(to_mpl_times(dt64), expected)
    np.testing.assert_allclose(to_mpl_times(naive), expected)
    np.testing.assert_allclose(to_mpl_times(aware), expected)
