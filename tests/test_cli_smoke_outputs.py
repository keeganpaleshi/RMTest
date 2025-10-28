import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from fitting import FitResult
import analyze


def _run(tmp_path: Path, monkeypatch, unbinned: bool = False):
    data_dir = Path(__file__).resolve().parent / "data" / "mini_run"
    csv = data_dir / "run.csv"
    with open(data_dir / "config.yaml", "r", encoding="utf-8") as f:
        cfg = json.load(f)
    cfg.setdefault("spectral_fit", {})
    spec_cfg = cfg["spectral_fit"]
    spec_cfg["do_spectral_fit"] = True
    spec_cfg.setdefault("expected_peaks", {"Po210": 0})
    spec_cfg.setdefault("amp_prior_scale", 1.0)
    spec_cfg.setdefault("mu_sigma", 1.0)
    spec_cfg.setdefault("b0_prior", [0.0, 1.0])
    spec_cfg.setdefault("b1_prior", [0.0, 1.0])
    spec_cfg.setdefault("bkg_mode", "manual")
    if unbinned:
        spec_cfg["unbinned_likelihood"] = True
    else:
        spec_cfg.pop("unbinned_likelihood", None)
    plotting_cfg = cfg.setdefault("plotting", {})
    plotting_cfg["plot_save_formats"] = ["pdf"]
    cfg_path = tmp_path / ("cfg_unbinned.yaml" if unbinned else "cfg.yaml")
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(cfg, f)
    def fake_fit_spectrum(
        energies,
        priors,
        flags=None,
        bins=None,
        bin_edges=None,
        bounds=None,
        unbinned=False,
        strict=False,
    ):
        params = {"likelihood_path": "unbinned_extended" if unbinned else "binned_poisson", "fit_valid": True}
        return FitResult(params, np.zeros((1, 1)), 0)

    monkeypatch.setattr(analyze, "fit_spectrum", fake_fit_spectrum)
    monkeypatch.setattr(
        analyze,
        "_save_stub_spectrum_plot",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("stub spectrum plot should not be used")),
    )
    monkeypatch.setattr(
        analyze,
        "plot_spectrum_comparison",
        lambda *a, out_png, **k: Path(out_png).touch(),
    )
    monkeypatch.setattr(
        analyze,
        "plot_time_series",
        lambda *a, out_png, **k: Path(out_png).touch(),
    )

    analyze.main(["-i", str(csv), "-c", str(cfg_path), "-o", str(tmp_path)])
    summary_files = list(tmp_path.glob("*/summary.json"))
    assert summary_files
    summary_file = max(summary_files, key=lambda p: p.stat().st_mtime)
    with open(summary_file, "r", encoding="utf-8") as f:
        summary = json.load(f)
    return summary, summary_file.parent


def test_cli_smoke_outputs(tmp_path, monkeypatch):
    summary, out_dir = _run(tmp_path, monkeypatch, unbinned=False)
    assert "diagnostics" in summary
    assert summary.get("spectral_fit", {}).get("likelihood_path") == "binned_poisson"
    assert (out_dir / "spectrum.png").exists()
    assert len(list(out_dir.glob("time_series_*.png"))) > 0

    summary2, _ = _run(tmp_path, monkeypatch, unbinned=True)
    assert summary2.get("spectral_fit", {}).get("likelihood_path") == "unbinned_extended"
