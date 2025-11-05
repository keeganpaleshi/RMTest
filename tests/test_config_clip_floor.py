"""Tests for clip_floor configuration loading and validation."""
import yaml
import pytest
from io_utils import load_config


def test_clip_floor_default(tmp_path):
    """Verify clip_floor gets a safe default when not specified."""
    cfg = {"spectral_fit": {}, "plotting": {"plot_save_formats": ["png"]}}
    # Add required keys for validation
    cfg["pipeline"] = {"log_level": "INFO"}
    cfg["time_fit"] = {}
    cfg["systematics"] = {"enable": False}

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    loaded = load_config(str(p))
    assert 0.0 < loaded["spectral_fit"]["clip_floor"] <= 1e-6
    assert loaded["spectral_fit"]["clip_floor"] == 1e-300


def test_clip_floor_override(tmp_path):
    """Verify clip_floor can be overridden."""
    cfg = {
        "spectral_fit": {"clip_floor": 1e-200},
        "plotting": {"plot_save_formats": ["png"]},
    }
    # Add required keys for validation
    cfg["pipeline"] = {"log_level": "INFO"}
    cfg["time_fit"] = {}
    cfg["systematics"] = {"enable": False}

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    loaded = load_config(str(p))
    assert loaded["spectral_fit"]["clip_floor"] == 1e-200


def test_clip_floor_invalid_raises(tmp_path):
    """Verify clip_floor=0 or negative values raise an error."""
    cfg = {
        "spectral_fit": {"clip_floor": 0.0},
        "plotting": {"plot_save_formats": ["png"]},
    }
    # Add required keys for validation
    cfg["pipeline"] = {"log_level": "INFO"}
    cfg["time_fit"] = {}
    cfg["systematics"] = {"enable": False}

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=r"spectral_fit\.clip_floor must be"):
        load_config(str(p))


def test_clip_floor_too_large_raises(tmp_path):
    """Verify clip_floor > 1e-6 raises an error."""
    cfg = {
        "spectral_fit": {"clip_floor": 1e-5},
        "plotting": {"plot_save_formats": ["png"]},
    }
    # Add required keys for validation
    cfg["pipeline"] = {"log_level": "INFO"}
    cfg["time_fit"] = {}
    cfg["systematics"] = {"enable": False}

    p = tmp_path / "cfg.yaml"
    p.write_text(yaml.safe_dump(cfg))
    with pytest.raises(ValueError, match=r"spectral_fit\.clip_floor must be"):
        load_config(str(p))
