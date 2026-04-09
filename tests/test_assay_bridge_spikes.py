import math
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from assay_bridge import fit_spike_periods


def _make_entry(ts: datetime, rate_hz: float, dt_s: float = 3600.0) -> dict[str, float]:
    counts = int(round(rate_hz * dt_s))
    return {"t": ts.timestamp(), "counts": counts, "dt": dt_s}


def test_fit_spike_periods_respects_bin_edges_and_fit_delay():
    t0 = datetime(2024, 1, 1, tzinfo=timezone.utc)
    lam = math.log(2) / (3.8235 * 86400.0)
    background = 0.5
    amplitude = 5.0

    # The first bin spans 00:00-01:00 and straddles the 30-minute fit delay.
    # It should therefore be excluded once the fitter uses bin edges.
    centers = [t0 + timedelta(minutes=30 + 60 * i) for i in range(6)]
    isotope_series = {"Po214": [], "Po218": []}
    for center in centers:
        rate = amplitude * math.exp(-lam * (center.timestamp() - t0.timestamp())) + background
        isotope_series["Po214"].append(_make_entry(center, rate * 0.6))
        isotope_series["Po218"].append(_make_entry(center, rate * 0.4))

    cfg = {
        "lucas_bridge": {
            "spike_periods": [
                {
                    "start": "2024-01-01T00:00:00Z",
                    "end": "2024-01-01T06:00:00Z",
                    "fit_delay_minutes": 30,
                    "label": "Synthetic spike",
                    "assay_date": "2024-01-02",
                }
            ]
        }
    }

    result = fit_spike_periods(isotope_series, cfg)
    period = result["periods"][0]

    assert "error" not in period
    assert period["fit_start"] == "2024-01-01T00:30:00Z"
    assert period["fit_delay_minutes"] == pytest.approx(30.0)
    assert period["n_bins"] == 5
