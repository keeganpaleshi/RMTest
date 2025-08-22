import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from calibration import CalibrationResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
import baseline_noise


def test_baseline_range_after_analysis_end(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "allow_fallback": True,
        "analysis": {"analysis_start_time": 0, "analysis_end_time": 5},
        "baseline": {
            "range": [7, 8],
            "monitor_volume_l": 605.0,
            "sample_volume_l": 0.0,
        },
        "calibration": {},
        "spectral_fit": {"do_spectral_fit": False, "expected_peaks": {"Po210": 0}},
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [pd.Timestamp(1.0, unit="s", tz="UTC")],
            "adc": [8.0],
            "fchannel": [1],
        }
    )
    data_path = tmp_path / "data.csv"
    df.to_csv(data_path, index=False)

    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={"Po210": {"centroid_adc": 10}},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(analyze, "derive_calibration_constants", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "derive_calibration_constants_auto", lambda *a, **k: cal_mock)
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "plot_time_series", lambda *a, **k: Path(k["out_png"]).touch())
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))
    monkeypatch.setattr(
        baseline_noise, "estimate_baseline_noise", lambda *a, **k: (None, {})
    )

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
    ]
    monkeypatch.setattr(sys, "argv", args)

    with pytest.raises(ValueError, match="baseline.*analysis_end_time"):
        analyze.main()
