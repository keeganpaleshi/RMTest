import numpy as np
from pathlib import Path

from plot_utils import plot_spectrum_comparison


def test_pre_post_fixed_bins(monkeypatch, tmp_path):
    def boom(*a, **k):  # pragma: no cover - ensures function is not used
        raise AssertionError("histogram_bin_edges called")

    monkeypatch.setattr(np, "histogram_bin_edges", boom)

    pre = np.linspace(0.0, 1.0, 10)
    post = np.linspace(0.0, 1.0, 10)
    out = tmp_path / "pp.png"
    plot_spectrum_comparison(pre, post, out_png=str(out))
    assert out.exists()
