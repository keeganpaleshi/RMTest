import sys
import json
from pathlib import Path

import pandas as pd
import numpy as np
from calibration import CalibrationResult

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
from constants import DEFAULT_ADC_CENTROIDS
from io_utils import load_config
from fitting import FitResult, FitParams


def test_expected_peaks_default(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": True,
            "mu_sigma": 0.05,
            "amp_prior_scale": 1.0,
            "S_bkg_prior": [0.0, 1.0],
            "beta1_prior": [0.0, 1.0],
            "peak_search_method": "cwt",
        },
        "time_fit": {"do_time_fit": False},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }

    cfg_path = tmp_path / "cfg.json"
    with open(cfg_path, "w") as f:
        json.dump(cfg, f)

    # Ensure load_config works without expected_peaks
    loaded = load_config(cfg_path)
    assert "expected_peaks" not in loaded.get("spectral_fit", {})

    df = pd.DataFrame({
        "fUniqueID": [1],
        "fBits": [0],
        "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")],
        "adc": [1000],
        "fchannel": [1],
    })
    data_path = tmp_path / "events.csv"
    df.to_csv(data_path, index=False)

    cal_mock = CalibrationResult(
        coeffs=[0.0, 1.0],
        cov=np.zeros((2, 2)),
        peaks={},
        sigma_E=1.0,
        sigma_E_error=0.0,
    )
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants",
        lambda *a, **k: cal_mock,
    )
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants_auto",
        lambda *a, **k: cal_mock,
    )
    monkeypatch.setattr(analyze, "fit_spectrum", lambda *a, **k: FitResult(FitParams({}), np.zeros((0, 0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))

    captured = {}

    def fake_find_adc_bin_peaks(adc_values, expected, **kwargs):
        captured["expected"] = expected
        captured["method"] = kwargs.get("method")
        return {k: float(v) for k, v in expected.items()}

    monkeypatch.setattr(analyze, "find_adc_bin_peaks", fake_find_adc_bin_peaks)

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

    analyze.main()

    assert captured.get("expected") == DEFAULT_ADC_CENTROIDS
    assert captured.get("method") == "cwt"

