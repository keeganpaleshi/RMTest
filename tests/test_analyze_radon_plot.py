import numpy as np

import analyze
import radon_activity


def test_radon_activity_curve_uses_defaults_for_missing_uncertainties(monkeypatch):
    calls: list[tuple[float, float, float, float]] = []

    def fake_curve(times, E, dE, N0, dN0, hl, cov=0.0):  # pragma: no cover - stub
        calls.append((E, dE, N0, dN0))
        return np.zeros_like(times, dtype=float), np.ones_like(times, dtype=float)

    monkeypatch.setattr(radon_activity, "radon_activity_curve", fake_curve)

    t_rel = np.array([0.0, 1.0, 2.0], dtype=float)
    cfg: dict[str, object] = {"time_fit": {}}

    fit_po214 = {
        "E_corrected": 5.0,
        "dE_corrected": None,
        "N0_Po214": 10.0,
        "dN0_Po214": None,
    }
    fit_po218 = {
        "E_corrected": None,
        "E_Po218": 7.0,
        "dE_corrected": None,
        "dE_Po218": None,
        "N0_Po218": 11.0,
        "dN0_Po218": None,
    }

    act214, err214 = analyze._radon_activity_curve_from_fit(
        "Po214", {}, fit_po214, t_rel, cfg
    )
    act218, err218 = analyze._radon_activity_curve_from_fit(
        "Po218", {}, fit_po218, t_rel, cfg
    )

    assert len(calls) == 2
    assert calls[0] == (5.0, 0.0, 10.0, 0.0)
    assert calls[1] == (7.0, 0.0, 11.0, 0.0)
    assert act214.shape == t_rel.shape
    assert err214.shape == t_rel.shape
    assert act218.shape == t_rel.shape
    assert err218.shape == t_rel.shape
