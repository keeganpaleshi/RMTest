from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

from analyze import _normalise_mu_bounds
import analyze
from calibration import apply_calibration
from utils.time_utils import parse_timestamp


def test_parse_timestamp_normalises_to_utc():
    assert parse_timestamp(42) == pd.Timestamp(42, unit="s", tz="UTC")
    assert parse_timestamp("1970-01-01T01:00:00+01:00") == pd.Timestamp(
        "1970-01-01T00:00:00Z"
    )
    assert parse_timestamp(
        datetime(1970, 1, 1, 1, tzinfo=timezone(timedelta(hours=1)))
    ) == pd.Timestamp("1970-01-01T00:00:00Z")


def test_normalise_mu_bounds_adc_uses_calibration():
    slope = 0.005
    intercept = 0.02
    bounds_adc = {"Po210": [1000, 1010]}

    out = _normalise_mu_bounds(
        bounds_adc,
        units="adc",
        slope=slope,
        intercept=intercept,
        quadratic_coeff=0.0,
    )

    expected = apply_calibration([1000, 1010], slope, intercept)
    assert np.allclose(out["Po210"], expected)


def test_shared_shift_anchor_auto_excludes_po210_and_unknown1():
    anchors = analyze._resolve_template_shared_shift_anchor_isotopes(
        ["Po210", "Po214", "Po218", "Unknown1"],
        {},
    )

    assert anchors == ["Po214", "Po218"]


def test_merge_template_lower_floor_priors_prefers_stronger_local_floor():
    merged = analyze._merge_template_lower_floor_priors(
        {"S_Po214": [100.0, 35.0]},
        {"S_Po214": [180.0, 40.0], "S_Po210": [50.0, 18.0]},
    )

    assert merged["S_Po214"] == [180.0, 40.0]
    assert merged["S_Po210"] == [50.0, 18.0]


def test_build_template_from_aggregate_freezes_split_peak_shape_by_default():
    aggregate_params = {
        "mu_Po214": 7.687,
        "S_Po214": 120.0,
        "f_gauss2_shared": 0.29,
        "sigma_gauss2_ratio_shared": 1.4,
        "delta_E_broad": 0.02,
        "b0": 0.0,
        "b1": 0.0,
        "b2": 0.0,
        "b3": 0.0,
        "S_bkg": 10.0,
    }
    priors = {
        "mu_Po214": (7.687, 0.01),
        "S_Po214": (120.0, 20.0),
        "f_gauss2_shared": (0.29, 0.10),
        "sigma_gauss2_ratio_shared": (1.4, 0.5),
        "delta_E_broad": (0.02, 0.02),
        "b0": (0.0, 0.1),
        "b1": (0.0, 1.0),
        "b2": (0.0, 0.1),
        "b3": (0.0, 0.1),
        "S_bkg": (10.0, 5.0),
    }

    _, tpl_flags = analyze._build_template_from_aggregate(
        aggregate_params,
        priors,
        {},
        {"time_fit": {"template_shape_mode": "soft", "float_centroids": True}},
    )

    assert tpl_flags["fix_f_gauss2_shared"] is True
    assert tpl_flags["fix_sigma_gauss2_ratio_shared"] is True
    assert tpl_flags["fix_delta_E_broad"] is True


def test_template_independent_centroid_spread_kev_detects_unphysical_split():
    spread = analyze._template_independent_centroid_spread_kev(
        {
            "mu_Po218": 6.01,
            "mu_Po216": 6.17,
            "mu_Po214": 7.68,
            "S_Po218": 20.0,
            "S_Po216": 15.0,
            "S_Po214": 100.0,
        },
        {
            "mu_Po218": 6.002,
            "mu_Po216": 6.778,
            "mu_Po214": 7.687,
        },
        anchor_isotopes=["Po218", "Po216", "Po214"],
    )

    assert spread > 100.0


def test_relax_mu_bounds_around_seed_respects_max_expand_cap():
    relaxed = analyze._relax_mu_bounds_around_seed(
        (5.93, 6.07),
        6.17,
        margin_kev=50.0,
        max_expand_kev=35.0,
    )

    assert relaxed == (5.93, 6.07)
