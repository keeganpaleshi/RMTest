# tests/test_config_clip_floor.py
"""Tests for clip_floor configuration loading and validation."""
import json
import pytest
from io_utils import load_config


def test_clip_floor_default(tmp_path):
    """Test that clip_floor defaults to 1e-300 when not specified."""
    cfg = {
        "spectral_fit": {},
        "pipeline": {"log_level": "INFO"},
    }
    p = tmp_path / "cfg.yaml"
    import yaml
    p.write_text(yaml.dump(cfg))
    loaded = load_config(str(p))
    assert 0.0 < loaded["spectral_fit"]["clip_floor"] <= 1e-6
    assert loaded["spectral_fit"]["clip_floor"] == 1e-300


def test_clip_floor_override(tmp_path):
    """Test that clip_floor can be overridden with a valid value."""
    cfg = {
        "spectral_fit": {"clip_floor": 1e-200},
        "pipeline": {"log_level": "INFO"},
    }
    p = tmp_path / "cfg.yaml"
    import yaml
    p.write_text(yaml.dump(cfg))
    loaded = load_config(str(p))
    assert loaded["spectral_fit"]["clip_floor"] == 1e-200


def test_clip_floor_invalid_zero_raises(tmp_path):
    """Test that clip_floor=0.0 raises a ValueError."""
    cfg = {
        "spectral_fit": {"clip_floor": 0.0},
        "pipeline": {"log_level": "INFO"},
    }
    p = tmp_path / "cfg.yaml"
    import yaml
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="spectral_fit.clip_floor must be in"):
        load_config(str(p))


def test_clip_floor_invalid_negative_raises(tmp_path):
    """Test that negative clip_floor raises a ValueError."""
    cfg = {
        "spectral_fit": {"clip_floor": -1e-300},
        "pipeline": {"log_level": "INFO"},
    }
    p = tmp_path / "cfg.yaml"
    import yaml
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="spectral_fit.clip_floor must be in"):
        load_config(str(p))


def test_clip_floor_invalid_too_large_raises(tmp_path):
    """Test that clip_floor > 1e-6 raises a ValueError."""
    cfg = {
        "spectral_fit": {"clip_floor": 1e-5},
        "pipeline": {"log_level": "INFO"},
    }
    p = tmp_path / "cfg.yaml"
    import yaml
    p.write_text(yaml.dump(cfg))
    with pytest.raises(ValueError, match="spectral_fit.clip_floor must be in"):
        load_config(str(p))
