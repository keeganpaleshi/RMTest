import yaml
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze
import likelihood_ext
from fitting import FitResult, FitParams
from feature_selectors import select_neg_loglike
from calibration import CalibrationResult


def test_cli_likelihood_extended(tmp_path, monkeypatch):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {},
        "spectral_fit": {
            "do_spectral_fit": True,
            "expected_peaks": {"Po210": 7600},
            "mu_sigma": 0.1,
            "amp_prior_scale": 1.0,
            "bkg_mode": "manual",
            "b0_prior": [0.0, 1.0],
            "b1_prior": [0.0, 1.0],
            "flags": {},
        },
        "time_fit": {"do_time_fit": False, "flags": {}},
        "systematics": {"enable": False},
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    with open(cfg_path, "w") as f:
        yaml.safe_dump(cfg, f)

    df = pd.DataFrame(
        {
            "fUniqueID": [1],
            "fBits": [0],
            "timestamp": [pd.Timestamp(0, unit="s", tz="UTC")],
            "adc": [7600],
            "fchannel": [1],
        }
    )
    data_path = tmp_path / "run.csv"
    df.to_csv(data_path, index=False)

    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants",
        lambda *a, **k: CalibrationResult(
            coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0
        ),
    )
    monkeypatch.setattr(
        analyze,
        "derive_calibration_constants_auto",
        lambda *a, **k: CalibrationResult(
            coeffs=[0.0, 1.0], cov=np.zeros((2, 2)), peaks={}, sigma_E=1.0, sigma_E_error=0.0
        ),
    )
    monkeypatch.setattr(analyze, "apply_calibration", lambda adc, a, c, quadratic_coeff=None: np.asarray(adc))
    monkeypatch.setattr(analyze, "find_adc_bin_peaks", lambda *a, **k: {"Po210": 7600})
    monkeypatch.setattr(analyze, "fit_time_series", lambda *a, **k: FitResult(FitParams({}), np.zeros((0, 0)), 0))
    monkeypatch.setattr(analyze, "plot_spectrum", lambda *a, **k: None)
    monkeypatch.setattr(analyze, "cov_heatmap", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "efficiency_bar", lambda *a, **k: Path(a[1]).touch())
    monkeypatch.setattr(analyze, "apply_burst_filter", lambda df, cfg, mode="rate": (df, 0))

    called = {"extended": False}

    def fake_fit_spectrum(energies, priors, flags=None, **kwargs):
        opts = SimpleNamespace(**(flags or {}))
        neg_ll = select_neg_loglike(opts)
        neg_ll(np.array([1.0]), lambda E, p: np.ones_like(E), {"area": 1.0}, area_keys=("area",))
        called["extended"] = neg_ll is likelihood_ext.neg_loglike_extended
        return FitResult(FitParams({"fit_valid": True}), np.zeros((0, 0)), 0)

    monkeypatch.setattr(analyze, "fit_spectrum", fake_fit_spectrum)

    args = [
        "analyze.py",
        "--config",
        str(cfg_path),
        "--input",
        str(data_path),
        "--output_dir",
        str(tmp_path),
        "--likelihood",
        "extended",
    ]
    monkeypatch.setattr(sys, "argv", args)
    analyze.main()

    assert called["extended"] is True
