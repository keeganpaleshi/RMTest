import json
import sys
from pathlib import Path

import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import analyze
from io_utils import Summary, copy_config, load_config, load_events, write_summary


def test_load_config_accepts_canonical_lowercase_keys(tmp_path):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "columns": {
            "unique_id": "uid",
            "bits": "bit_col",
            "timestamp": "time_col",
            "adc": "adc_col",
            "channel": "chan_col",
        },
        "calibration": {
            "slope_mev_per_ch": 0.0043,
            "sigma_e_init": 0.015,
            "intercept_mev": -0.12,
        },
        "spectral_fit": {
            "expected_peaks": {"Po210": 1250, "Po218": 1400, "Po214": 1800},
            "background_norm_points": 64,
            "s_bkg_prior": [0.0, 1.0],
            "sigma_e_prior_source": "config",
            "sigma_e_prior_sigma": 0.2,
            "float_sigma_e": False,
            "tau_po214_prior_mean": 0.01,
            "tau_po214_prior_sigma": 0.005,
            "flags": {"fix_f": True, "f_prior": [0.0, 0.1]},
        },
        "time_fit": {
            "do_time_fit": True,
            "window_Po214": [7.5, 8.0],
            "flags": {"fix_n0_po214": True},
        },
        "systematics": {
            "enable": False,
            "sigma_e_frac": 0.1,
            "energy_shift_kev": 1.0,
        },
        "plotting": {"plot_save_formats": ["png"]},
    }
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg), encoding="utf-8")

    loaded = load_config(cfg_path)

    assert loaded["columns"]["fUniqueID"] == "uid"
    assert loaded["columns"]["fBits"] == "bit_col"
    assert loaded["columns"]["fchannel"] == "chan_col"
    assert loaded["calibration"]["slope_MeV_per_ch"] == 0.0043
    assert loaded["calibration"]["sigma_E_init"] == 0.015
    assert loaded["calibration"]["intercept_MeV"] == -0.12
    assert loaded["spectral_fit"]["loglin_n_norm"] == 64
    assert loaded["spectral_fit"]["S_bkg_prior"] == [0.0, 1.0]
    assert loaded["spectral_fit"]["sigma_E_prior_source"] == "config"
    assert loaded["spectral_fit"]["sigma_E_prior_sigma"] == 0.2
    assert loaded["spectral_fit"]["float_sigma_E"] is False
    assert loaded["spectral_fit"]["flags"]["fix_F"] is True
    assert loaded["spectral_fit"]["flags"]["F_prior"] == [0.0, 0.1]
    assert loaded["time_fit"]["window_po214"] == [7.5, 8.0]
    assert loaded["time_fit"]["flags"]["fix_N0_po214"] is True
    assert loaded["systematics"]["sigma_E_frac"] == 0.1
    assert loaded["systematics"]["energy_shift_keV"] == 1.0


def test_load_events_accepts_canonical_column_map_keys(tmp_path):
    df = pd.DataFrame(
        {
            "uid": [1],
            "bit_col": [0],
            "time_col": ["1970-01-01T00:00:01Z"],
            "adc_col": [12],
            "chan_col": [3],
        }
    )
    csv_path = tmp_path / "events.csv"
    df.to_csv(csv_path, index=False)

    loaded = load_events(
        csv_path,
        column_map={
            "unique_id": "uid",
            "bits": "bit_col",
            "timestamp": "time_col",
            "adc": "adc_col",
            "channel": "chan_col",
        },
    )

    assert list(loaded.columns) == ["fUniqueID", "fBits", "timestamp", "adc", "fchannel"]
    assert loaded.loc[0, "fUniqueID"] == "1"
    assert loaded.loc[0, "fBits"] == "0"
    assert loaded.loc[0, "fchannel"] == "3"


def test_write_summary_adds_canonical_output_aliases(tmp_path):
    summary = Summary(
        calibration={"sigma_E": 1.0, "sigma_E_error": 0.1},
        baseline={"rate_Bq": {"Po214": 1.0}, "baseline_rate_Bq": 0.5},
        radon_results={"total_radon_in_sample_Bq": {"value": 1.0, "uncertainty": 0.1}},
        radon_combined={"activity_Bq": 1.0, "unc_Bq": 0.1},
        radon={"Rn_activity_Bq": 1.0, "stat_unc_Bq": 0.2},
    )

    results_dir = write_summary(tmp_path, summary, "19700101T000000Z")
    data = json.loads((Path(results_dir) / "summary.json").read_text(encoding="utf-8"))

    assert data["calibration"]["sigma_e"] == 1.0
    assert data["calibration"]["sigma_E"] == 1.0
    assert data["calibration"]["sigma_e_error"] == 0.1
    assert data["radon"]["rn_activity_bq"] == 1.0
    assert data["radon"]["Rn_activity_Bq"] == 1.0
    assert data["radon"]["stat_unc_bq"] == 0.2
    assert data["baseline"]["rate_bq"]["Po214"] == 1.0
    assert data["baseline"]["rate_Bq"]["Po214"] == 1.0
    assert data["radon_combined"]["activity_bq"] == 1.0
    assert data["radon_combined"]["unc_bq"] == 0.1
    assert data["radon_results"]["total_radon_in_sample_bq"]["value"] == 1.0


def test_copy_config_writes_canonical_keys(tmp_path):
    cfg = {
        "pipeline": {"log_level": "INFO"},
        "calibration": {"slope_MeV_per_ch": 0.0043, "intercept_MeV": -0.15},
        "spectral_fit": {
            "expected_peaks": {"Po210": 1},
            "loglin_n_norm": 64,
            "float_sigma_E": False,
            "flags": {"fix_F": True, "F_prior": [0.0, 0.1]},
        },
        "time_fit": {"do_time_fit": True, "flags": {"fix_N0_po214": True}},
        "systematics": {"enable": False, "sigma_E_frac": 0.1, "energy_shift_keV": 1.0},
        "plotting": {"plot_save_formats": ["png"]},
    }

    dest = copy_config(tmp_path / "results", cfg)
    data = json.loads(Path(dest).read_text(encoding="utf-8"))

    assert data["calibration"]["slope_mev_per_ch"] == 0.0043
    assert data["calibration"]["intercept_mev"] == -0.15
    assert "slope_MeV_per_ch" not in data["calibration"]
    assert data["spectral_fit"]["background_norm_points"] == 64
    assert data["spectral_fit"]["float_sigma_e"] is False
    assert data["spectral_fit"]["flags"]["fix_f"] is True
    assert data["spectral_fit"]["flags"]["f_prior"] == [0.0, 0.1]
    assert data["time_fit"]["flags"]["fix_n0_po214"] is True
    assert data["systematics"]["sigma_e_frac"] == 0.1
    assert data["systematics"]["energy_shift_kev"] == 1.0


def test_parse_args_accepts_hyphenated_and_underscore_aliases():
    hyphenated = analyze.parse_args(["--output-dir", "out", "--baseline-range", "1", "2"])
    with pytest.warns(DeprecationWarning, match="--output_dir is deprecated; use --output-dir"):
        underscored = analyze.parse_args(["--output_dir", "out2", "--baseline_range", "3", "4"])

    assert hyphenated.output_dir == "out"
    assert hyphenated.baseline_range == ["1", "2"]
    assert underscored.output_dir == "out2"
    assert underscored.baseline_range == ["3", "4"]



def test_parse_args_accepts_hidden_deprecated_aliases():
    with pytest.warns(DeprecationWarning, match="--analysis_start_time is deprecated; use --analysis-start-time"):
        args = analyze.parse_args(
            [
                "--analysis_start_time",
                "1",
                "--plot_time_binning_mode",
                "fd",
                "--plot_time_bin_width",
                "120",
            ]
        )

    assert args.analysis_start_time == "1"
    assert args.time_bin_mode == "fd"
    assert args.time_bin_width == 120.0



def test_help_prefers_canonical_flag_names(capsys):
    with pytest.raises(SystemExit):
        analyze.parse_args(["--help"])

    out = capsys.readouterr().out
    assert "Inputs and outputs:" in out
    assert "Time selection and baseline:" in out
    assert "Calibration and fit controls:" in out
    assert "Plotting, diagnostics, and reproducibility:" in out
    assert "--output-dir" in out
    assert "--baseline-range" in out
    assert "--output_dir" not in out
    assert "--baseline_range" not in out
    assert "--time-bin-mode" in out
