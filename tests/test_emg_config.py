"""Tests for centralized EMG configuration constants and helpers.

This module tests the EMG configuration helpers in rmtest.emg_constants to
ensure that config-driven EMG settings are properly resolved and applied.
"""

import pytest
import numpy as np

from rmtest.emg_constants import (
    EMG_STABLE_MODE,
    EMG_MIN_TAU,
    EMG_DEFAULT_METHOD,
    emg_min_tau_from_config,
    emg_stable_mode_from_config,
    emg_method_from_config,
    emg_use_emg_from_config,
    clamp_tau,
)


class TestEMGDefaults:
    """Test that default EMG constants are defined and reasonable."""

    def test_emg_defaults_exist(self):
        """Verify that all default constants exist and have valid values."""
        # Defaults should exist
        assert EMG_STABLE_MODE is not None
        assert EMG_MIN_TAU is not None
        assert EMG_DEFAULT_METHOD is not None

        # Defaults should have expected types
        assert isinstance(EMG_STABLE_MODE, bool)
        assert isinstance(EMG_MIN_TAU, (int, float))
        assert isinstance(EMG_DEFAULT_METHOD, str)

        # Defaults should have reasonable values
        assert EMG_MIN_TAU > 0
        assert EMG_DEFAULT_METHOD in ("erfcx", "direct")

    def test_emg_stable_mode_default(self):
        """Verify stable mode is enabled by default."""
        assert EMG_STABLE_MODE is True

    def test_emg_min_tau_positive(self):
        """Verify minimum tau is positive."""
        assert EMG_MIN_TAU > 0


class TestConfigHelpers:
    """Test config helper functions with various inputs."""

    def test_emg_min_tau_from_empty_config(self):
        """Test that empty config returns default min_tau."""
        cfg = {}
        result = emg_min_tau_from_config(cfg)
        assert result == EMG_MIN_TAU
        assert result > 0

    def test_emg_min_tau_from_none(self):
        """Test that None config returns default min_tau."""
        result = emg_min_tau_from_config(None)
        assert result == EMG_MIN_TAU

    def test_emg_min_tau_from_config_override(self):
        """Test that config overrides default min_tau."""
        cfg = {"fitting": {"emg": {"min_tau": 1e-3}}}
        result = emg_min_tau_from_config(cfg)
        assert result == pytest.approx(1e-3)

    def test_emg_stable_mode_from_empty_config(self):
        """Test that empty config returns default stable mode."""
        cfg = {}
        result = emg_stable_mode_from_config(cfg)
        assert result is True

    def test_emg_stable_mode_from_none(self):
        """Test that None config returns default stable mode."""
        result = emg_stable_mode_from_config(None)
        assert result is True

    def test_emg_stable_mode_from_config_false(self):
        """Test that config can disable stable mode."""
        cfg = {"fitting": {"emg": {"stable_mode": False}}}
        result = emg_stable_mode_from_config(cfg)
        assert result is False

    def test_emg_stable_mode_from_config_true(self):
        """Test that config can explicitly enable stable mode."""
        cfg = {"fitting": {"emg": {"stable_mode": True}}}
        result = emg_stable_mode_from_config(cfg)
        assert result is True

    def test_emg_method_from_empty_config(self):
        """Test that empty config returns default method."""
        cfg = {}
        result = emg_method_from_config(cfg)
        assert result == EMG_DEFAULT_METHOD

    def test_emg_method_from_config_erfcx(self):
        """Test that config can select erfcx method."""
        cfg = {"fitting": {"emg": {"method": "erfcx"}}}
        result = emg_method_from_config(cfg)
        assert result == "erfcx"

    def test_emg_method_from_config_direct(self):
        """Test that config can select direct method."""
        cfg = {"fitting": {"emg": {"method": "direct"}}}
        result = emg_method_from_config(cfg)
        assert result == "direct"

    def test_emg_use_emg_from_empty_config(self):
        """Test that empty config returns False for use_emg."""
        cfg = {}
        result = emg_use_emg_from_config(cfg)
        assert result is False

    def test_emg_use_emg_from_config_bool_true(self):
        """Test that config can enable EMG globally."""
        cfg = {"fitting": {"emg": {"use_emg": True}}}
        result = emg_use_emg_from_config(cfg)
        assert result is True

    def test_emg_use_emg_from_config_bool_false(self):
        """Test that config can disable EMG globally."""
        cfg = {"fitting": {"emg": {"use_emg": False}}}
        result = emg_use_emg_from_config(cfg)
        assert result is False

    def test_emg_use_emg_from_config_mapping(self):
        """Test that config can specify per-isotope EMG settings."""
        cfg = {"fitting": {"emg": {"use_emg": {"Po210": True, "Po218": False}}}}
        result = emg_use_emg_from_config(cfg)
        assert isinstance(result, dict)
        assert result["Po210"] is True
        assert result["Po218"] is False


class TestClampTau:
    """Test tau clamping functionality."""

    def test_clamp_tau_below_minimum(self):
        """Test that tau below minimum is clamped."""
        tau = 1e-10
        cfg = {"fitting": {"emg": {"min_tau": 1e-4}}}
        result = clamp_tau(tau, cfg)
        assert result == pytest.approx(1e-4)

    def test_clamp_tau_above_minimum(self):
        """Test that tau above minimum is not clamped."""
        tau = 1e-2
        cfg = {"fitting": {"emg": {"min_tau": 1e-4}}}
        result = clamp_tau(tau, cfg)
        assert result == pytest.approx(1e-2)

    def test_clamp_tau_exactly_at_minimum(self):
        """Test that tau exactly at minimum is not clamped."""
        tau = 1e-4
        cfg = {"fitting": {"emg": {"min_tau": 1e-4}}}
        result = clamp_tau(tau, cfg)
        assert result == pytest.approx(1e-4)

    def test_clamp_tau_with_explicit_min(self):
        """Test that explicit min_tau parameter overrides config."""
        tau = 1e-10
        cfg = {"fitting": {"emg": {"min_tau": 1e-4}}}
        result = clamp_tau(tau, cfg, min_tau=1e-3)
        assert result == pytest.approx(1e-3)

    def test_clamp_tau_with_none_config(self):
        """Test that None config uses default minimum."""
        tau = 1e-10
        result = clamp_tau(tau, None)
        assert result >= EMG_MIN_TAU


class TestPerIsotopeConfig:
    """Test per-isotope EMG configuration scenarios."""

    def test_emg_per_isotope_override(self):
        """Test that per-isotope EMG settings work correctly."""
        cfg = {
            "fitting": {
                "emg": {
                    "use_emg": {"Po210": True, "Po218": False, "Po214": False},
                    "min_tau": 1e-3,
                }
            }
        }

        # Verify use_emg mapping
        use_emg = emg_use_emg_from_config(cfg)
        assert isinstance(use_emg, dict)
        assert use_emg["Po210"] is True
        assert use_emg["Po218"] is False
        assert use_emg["Po214"] is False

        # Verify min_tau
        min_tau = emg_min_tau_from_config(cfg)
        assert min_tau == pytest.approx(1e-3)

        # Simulate building a Po210 peak with tau clamping
        tau_prior = 0.005  # 5ms tail
        tau_clamped = clamp_tau(tau_prior, cfg)
        assert tau_clamped == pytest.approx(0.005)
        assert tau_clamped >= min_tau

    def test_emg_tau_clamping(self):
        """Test that tau is properly clamped to minimum."""
        cfg = {
            "fitting": {
                "emg": {
                    "min_tau": 1e-3,
                }
            }
        }

        # Tau below minimum should be clamped
        tau_too_small = 1e-6
        tau_clamped = clamp_tau(tau_too_small, cfg)
        assert tau_clamped == pytest.approx(1e-3)

        # Tau above minimum should not be clamped
        tau_ok = 0.005
        tau_clamped = clamp_tau(tau_ok, cfg)
        assert tau_clamped == pytest.approx(0.005)


class TestBaselineHelpers:
    """Test baseline-related helper functions."""

    def test_summarize_baseline_allows_negative(self):
        """Test baseline_allows_negative helper function."""
        from baseline_utils import baseline_allows_negative

        # Default should be False
        cfg = {}
        assert baseline_allows_negative(cfg) is False

        # Explicit True
        cfg = {"baseline": {"allow_negative_baseline": True}}
        assert baseline_allows_negative(cfg) is True

        # Explicit False
        cfg = {"baseline": {"allow_negative_baseline": False}}
        assert baseline_allows_negative(cfg) is False

    def test_summarize_baseline_clips_negative(self):
        """Test that summarize_baseline clips negative values when configured."""
        from baseline_utils import summarize_baseline

        # Create a config that disallows negative baselines
        cfg = {
            "baseline": {
                "allow_negative_baseline": False,
                "rate_Bq": {"Po210": 5.0},
                "scales": {"Po210": 1.0},
            },
            "time_fit": {
                "Po210": {"E_Po210": 3.0}  # Raw < baseline, would be negative
            },
        }

        result = summarize_baseline(cfg, ["Po210"])
        assert "Po210" in result
        raw, baseline, corrected = result["Po210"]
        assert raw == pytest.approx(3.0)
        assert baseline == pytest.approx(5.0)
        assert corrected == pytest.approx(0.0)  # Clipped to 0

    def test_summarize_baseline_allows_negative_when_configured(self):
        """Test that summarize_baseline allows negative values when configured."""
        from baseline_utils import summarize_baseline

        # Create a config that allows negative baselines
        cfg = {
            "baseline": {
                "allow_negative_baseline": True,
                "rate_Bq": {"Po210": 5.0},
                "scales": {"Po210": 1.0},
            },
            "time_fit": {
                "Po210": {"E_Po210": 3.0}  # Raw < baseline, will be negative
            },
        }

        result = summarize_baseline(cfg, ["Po210"])
        assert "Po210" in result
        raw, baseline, corrected = result["Po210"]
        assert raw == pytest.approx(3.0)
        assert baseline == pytest.approx(5.0)
        assert corrected == pytest.approx(-2.0)  # Allowed to be negative


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
