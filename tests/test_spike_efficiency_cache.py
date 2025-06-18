import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import utils
import efficiency


def test_spike_efficiency_cache_separate_by_error(monkeypatch):
    calls = []

    def fake_calc(counts, act, live):
        calls.append((counts, act, live))
        return 0.1

    monkeypatch.setattr(efficiency, "calc_spike_efficiency", fake_calc)

    utils._spike_eff_cache.clear()
    cfg1 = {"counts": 10, "activity_bq": 5, "live_time_s": 100, "error": 0.1}
    cfg2 = {"counts": 10, "activity_bq": 5, "live_time_s": 100, "error": 0.2}

    utils.get_spike_efficiency(cfg1)
    utils.get_spike_efficiency(cfg2)

    assert len(calls) == 2
    assert len(utils._spike_eff_cache) == 2
