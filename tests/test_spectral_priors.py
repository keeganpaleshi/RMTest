import sys
from pathlib import Path

# Add the ``src`` directory so the ``rmtest`` package can be imported
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from rmtest.spectral.priors import apply_width_prior


def test_priors_copied_and_keys_removed_when_flag_true():
    priors = {"sigma0": 1, "F": 2, "A": 3}
    cfg = {"spectral_fit": {"no_width_prior": True}}
    out = apply_width_prior(priors, cfg)
    # Ensure original dictionary unchanged and copy returned
    assert priors == {"sigma0": 1, "F": 2, "A": 3}
    assert out is not priors
    assert "sigma0" not in out and "F" not in out
    assert "A" in out


def test_keys_remain_when_flag_absent():
    priors = {"sigma0": 1, "F": 2}
    cfg = {}
    out = apply_width_prior(priors, cfg)
    assert out["sigma0"] == 1
    assert out["F"] == 2


def test_keys_remain_when_flag_false():
    priors = {"sigma0": 1, "F": 2}
    cfg = {"spectral_fit": {"no_width_prior": False}}
    out = apply_width_prior(priors, cfg)
    assert out["sigma0"] == 1
    assert out["F"] == 2
