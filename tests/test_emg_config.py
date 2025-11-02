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


class TestRegressionGuards:
    """Tests to prevent regressions in constant handling and alias mapping."""

    def test_constants_tau_min_not_overwritten(self):
        """Ensure constants._TAU_MIN uses centralized value, not legacy override.

        This test guards against regression of accidentally overriding the
        centralized EMG_MIN_TAU with a legacy hardcoded value like 1e-8.
        """
        import constants

        # The constants module should use the centralized value (5e-4 default)
        # NOT the old legacy value (1e-8)
        assert constants._TAU_MIN >= 5e-4
        assert constants._TAU_MIN != pytest.approx(1e-8)

        # Verify it actually matches the centralized constant
        from rmtest.emg_constants import EMG_MIN_TAU
        assert constants._TAU_MIN == pytest.approx(EMG_MIN_TAU)

    def test_emg_method_alias_direct_resolves_correctly(self):
        """Ensure 'direct' method alias resolves to intended strategy.

        This guards against accidental changes to the method alias table.
        The 'direct' method should map to 'scipy_safe' strategy.
        """
        from calibration import _normalize_emg_mode

        # Test that "direct" maps to "scipy_safe"
        assert _normalize_emg_mode("direct") == "scipy_safe"

        # Also test other common aliases for consistency
        assert _normalize_emg_mode("erfcx") == "erfcx_exact"
        assert _normalize_emg_mode("erfcx_exact") == "erfcx_exact"
        assert _normalize_emg_mode("legacy") == "legacy"
        assert _normalize_emg_mode("scipy_safe") == "scipy_safe"
        assert _normalize_emg_mode("") == "scipy_safe"  # default

    def test_emg_stable_uses_runtime_tau_floor(self):
        """Ensure emg_stable.py uses runtime-configurable tau floor, not frozen constant.

        This tests that emg_stable._get_tau_min() queries constants._TAU_MIN
        instead of using a frozen module-level constant.
        """
        import constants
        from emg_stable import _get_tau_min

        # Get the original value
        original_tau = getattr(constants, '_TAU_MIN', None)

        try:
            # Set a test value in constants
            test_tau = 1e-3
            constants._TAU_MIN = test_tau

            # emg_stable should pick up the new value
            assert _get_tau_min() == pytest.approx(test_tau)

        finally:
            # Restore original value
            if original_tau is not None:
                constants._TAU_MIN = original_tau

    def test_calibration_uses_single_tau_min(self):
        """Ensure calibration.py doesn't have confusing dual tau min paths.

        This guards against regression where _EMG_TAU_MIN and _TAU_MIN
        coexist with different values.
        """
        import calibration

        # Should have _TAU_MIN
        assert hasattr(calibration, '_TAU_MIN')

        # Should NOT have separate _EMG_TAU_MIN (that was removed)
        # If it exists, it should be the same as _TAU_MIN
        if hasattr(calibration, '_EMG_TAU_MIN'):
            assert calibration._EMG_TAU_MIN == calibration._TAU_MIN

        # get_emg_tau_min() should return _TAU_MIN
        from calibration import get_emg_tau_min
        assert get_emg_tau_min() == pytest.approx(calibration._TAU_MIN)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
