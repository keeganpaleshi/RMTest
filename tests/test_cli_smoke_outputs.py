import json
from pathlib import Path

import analyze


def _run(tmp_path: Path, unbinned: bool) -> Path:
    tmp_path.mkdir(parents=True, exist_ok=True)
    data_dir = Path(__file__).resolve().parent / "data" / "mini_run"
    csv = data_dir / "run.csv"
    cfg = json.loads((data_dir / "config.yaml").read_text())
    cfg["spectral_fit"].update({
        "do_spectral_fit": True,
        "unbinned_likelihood": unbinned,
        "expected_peaks": {"Po210": 0},
        "mu_sigma": 0.05,
        "amp_prior_scale": 1.0,
        "b0_prior": [0.0, 1.0],
        "b1_prior": [0.0, 1.0],
        "spectral_binning_mode": "adc",
        "adc_bin_width": 10,
        "flags": {"fix_sigma0": True, "fix_F": True},
        "mu_bounds": {"Po210": [5.2, 5.4]},
        "float_sigma_E": False,
    })
    cfg_path = tmp_path / "cfg.yaml"
    cfg_path.write_text(json.dumps(cfg))

    def fake_fit_spectrum(*args, unbinned=False, **kwargs):
        return {"likelihood_path": "unbinned_extended" if unbinned else "binned_poisson"}
    analyze.fit_spectrum = fake_fit_spectrum

    try:
        analyze.main(["-i", str(csv), "-c", str(cfg_path), "-o", str(tmp_path)])
    except SystemExit as e:
        assert e.code == 0

    out_dir = next(p for p in tmp_path.iterdir() if p.is_dir() and p.name != "__pycache__")
    return out_dir


def test_cli_smoke_outputs(tmp_path):
    out_dir = _run(tmp_path, unbinned=False)

    summary_path = out_dir / "summary.json"
    assert summary_path.exists()
    summary = json.loads(summary_path.read_text())
    assert "diagnostics" in summary
    assert summary["spectral_fit"]["likelihood_path"] == "binned_poisson"
    assert (out_dir / "spectrum.png").exists()
    assert any(out_dir.glob("time_series_*.png"))

    out_dir2 = _run(tmp_path / "unbinned", unbinned=True)
    summary2 = json.loads((out_dir2 / "summary.json").read_text())
    assert summary2["spectral_fit"]["likelihood_path"] == "unbinned_extended"
