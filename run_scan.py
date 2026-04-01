#!/usr/bin/env python3
"""Automated scan runner for shape parameter and DNL optimization.

Phases A-C: Test shared vs per-isotope freedom for each shape parameter
group (tail/shelf/halo) with DNL disabled - isolates model structure.

Phase D: DNL correction methods (Fourier/bandpass) with physically-motivated
shape sharing. Tau always shared (detector charge-collection property).

Phase E: DNL x binning exploration at bin_width=1,2 with Fourier, bandpass,
and hybrid DNL modes. Tests iteration count, smooth window, and period sets.

Phase F: Two-stage pipeline - Fourier DNL at full ADC resolution (bin_width=1),
then rebin to coarser widths before the final fit.

Phase G: Fourier period sweep - progressively adds shorter periods to find the
minimal basis that passes crossvalidation. G6 uses per-period crossval
auto-selection.

Usage:
    python run_scan.py [--scans A1,A2,...] [--config config.yaml]

Each scan takes ~1-20 min depending on free parameter count and DNL mode.
"""

import argparse
import copy
import csv
import json
import logging
import os
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).parent
SCAN_CONFIG_DIR = BASE_DIR / "scan_configs"
RESULTS_DIR = BASE_DIR / "results"

# ── Isotopes that participate in shape param sharing ──────────
# Po216 has EMG/shelf/halo disabled; Po212 has low stats → fixed in
# per-isotope scans.  Only Po210, Po218, Po214 get per-isotope freedom.
FREE_ISOTOPES = ["Po210", "Po218", "Po214"]
ALL_SHAPE_ISOTOPES = ["Po210", "Po218", "Po214", "Po212"]

# ── Fix flag patterns ──────────────────────────────────────────
TAIL_PARAMS = ["tau"]
SHELF_PARAMS = ["f_shelf", "sigma_shelf"]
HALO_PARAMS = ["f_halo", "sigma_halo", "tau_halo"]


def _make_fix_overrides(free_types, mode, isotopes=None):
    """Generate fix flag overrides.

    Parameters
    ----------
    free_types : list[str]
        Which param groups to free: "tail", "shelf", "halo"
    mode : str
        "shared" = use shared param (unfix the shared, keep per-iso fixed)
        "per_isotope" = unfix per-isotope for FREE_ISOTOPES only
    isotopes : list[str] or None
        Which isotopes to free (default: FREE_ISOTOPES for per_isotope,
        ALL_SHAPE_ISOTOPES for shared)
    """
    fix_flags = {}
    shared_flags = {}

    param_map = {
        "tail": TAIL_PARAMS,
        "shelf": SHELF_PARAMS,
        "halo": HALO_PARAMS,
    }

    for group in free_types:
        params = param_map[group]
        if mode == "shared":
            for p in params:
                shared_flags[p] = True  # enable shared param
                # Keep all per-isotope fixed (default from base config)
        elif mode == "per_isotope":
            isos = isotopes or FREE_ISOTOPES
            for p in params:
                for iso in isos:
                    fix_flags[f"fix_{p}_{iso}"] = False  # unfix

    return fix_flags, shared_flags


# ── 18 Scan Definitions ────────────────────────────────────────
SCANS = {}


def _define_scan(scan_id, description, free_types_modes):
    """Helper to define a scan.

    free_types_modes: list of (group, mode) tuples
        e.g. [("tail", "shared"), ("shelf", "per_isotope")]
    """
    fix_flags = {}
    shared_flags = {}
    for group, mode in free_types_modes:
        ff, sf = _make_fix_overrides([group], mode)
        fix_flags.update(ff)
        shared_flags.update(sf)
    SCANS[scan_id] = {
        "description": description,
        "fix_flags": fix_flags,
        "shared_flags": shared_flags,
    }


# Phase 1: Single mechanism
_define_scan("A1", "Shared tau only", [("tail", "shared")])
_define_scan("A2", "Per-isotope tau only", [("tail", "per_isotope")])
_define_scan("A3", "Shared shelf only", [("shelf", "shared")])
_define_scan("A4", "Per-isotope shelf only", [("shelf", "per_isotope")])
_define_scan("A5", "Shared halo only", [("halo", "shared")])
_define_scan("A6", "Per-isotope halo only", [("halo", "per_isotope")])

# Phase 2: Two mechanisms combined
_define_scan("B1", "Shared tau + shared shelf",
             [("tail", "shared"), ("shelf", "shared")])
_define_scan("B2", "Shared tau + per-iso shelf",
             [("tail", "shared"), ("shelf", "per_isotope")])
_define_scan("B3", "Per-iso tau + per-iso shelf",
             [("tail", "per_isotope"), ("shelf", "per_isotope")])
_define_scan("B4", "Shared tau + shared halo",
             [("tail", "shared"), ("halo", "shared")])
_define_scan("B5", "Per-iso shelf + shared halo",
             [("shelf", "per_isotope"), ("halo", "shared")])
_define_scan("B6", "Per-iso tau + shared halo",
             [("tail", "per_isotope"), ("halo", "shared")])

# Phase 3: All three mechanisms
_define_scan("C1", "All shared (tail+shelf+halo)",
             [("tail", "shared"), ("shelf", "shared"), ("halo", "shared")])
_define_scan("C2", "Shared tail+halo, per-iso shelf",
             [("tail", "shared"), ("shelf", "per_isotope"), ("halo", "shared")])
_define_scan("C3", "Per-iso tail+shelf, shared halo",
             [("tail", "per_isotope"), ("shelf", "per_isotope"), ("halo", "shared")])
_define_scan("C4", "All per-isotope (max freedom)",
             [("tail", "per_isotope"), ("shelf", "per_isotope"), ("halo", "per_isotope")])

# Phase 4: DNL + shape sharing (tau always shared)
# Tests DNL correction methods with physically-motivated sharing.
# tau is shared in all D-scans (charge collection = detector property).
# Shelf and halo sharing are varied; DNL method is Fourier or bandpass.

def _define_dnl_scan(scan_id, description, free_types_modes, dnl_mode):
    """Define a D-series scan with DNL correction.

    Parameters
    ----------
    scan_id : str
    description : str
    free_types_modes : list of (group, mode) tuples
    dnl_mode : str
        "fourier" = parameterized Fourier DNL (~10 DOF)
        "bandpass" = self-estimated bandpass DNL (~182 DOF)
    """
    fix_flags = {}
    shared_flags = {}
    for group, mode in free_types_modes:
        ff, sf = _make_fix_overrides([group], mode)
        fix_flags.update(ff)
        shared_flags.update(sf)
    SCANS[scan_id] = {
        "description": description,
        "fix_flags": fix_flags,
        "shared_flags": shared_flags,
        "dnl_mode": dnl_mode,
    }


# Fourier DNL series (D1-D4)
_define_dnl_scan("D1", "All shared + Fourier DNL",
                 [("tail", "shared"), ("shelf", "shared"), ("halo", "shared")],
                 dnl_mode="fourier")
_define_dnl_scan("D2", "Shared tau+halo, per-iso shelf + Fourier DNL",
                 [("tail", "shared"), ("shelf", "per_isotope"), ("halo", "shared")],
                 dnl_mode="fourier")
_define_dnl_scan("D3", "Shared tau+shelf, per-iso halo + Fourier DNL",
                 [("tail", "shared"), ("shelf", "shared"), ("halo", "per_isotope")],
                 dnl_mode="fourier")
_define_dnl_scan("D4", "Shared tau, per-iso shelf+halo + Fourier DNL",
                 [("tail", "shared"), ("shelf", "per_isotope"), ("halo", "per_isotope")],
                 dnl_mode="fourier")

# Bandpass DNL series (D5-D6)
_define_dnl_scan("D5", "All shared + Bandpass DNL",
                 [("tail", "shared"), ("shelf", "shared"), ("halo", "shared")],
                 dnl_mode="bandpass")
_define_dnl_scan("D6", "Shared tau+halo, per-iso shelf + Bandpass DNL",
                 [("tail", "shared"), ("shelf", "per_isotope"), ("halo", "shared")],
                 dnl_mode="bandpass")


# Phase 5: DNL × binning exploration (E-series)
# Tests whether finer ADC binning reveals real DNL structure that 3ch
# averaging destroys.  All shapes shared (C1 equivalent) to isolate
# the DNL effect.  Compare within each binning level.

def _define_binning_scan(scan_id, description, adc_bin_width,
                         dnl_mode="none", crossval=True,
                         dnl_overrides=None, config_overrides=None):
    """Define an E-series scan with binning + DNL overrides.

    Shape model: all shared (tail + shelf + halo), matching C1.

    Parameters
    ----------
    dnl_overrides : dict, optional
        Extra keys merged into spectral_fit.dnl_correction, e.g.
        {"iterations": 3, "smooth_window": 31,
         "fourier_periods_codes": [4,8,16,32,64,128,256,512]}.
    config_overrides : dict, optional
        Top-level config section overrides, e.g.
        {"calibration": {"use_quadratic": "cubic"}}.
    """
    fix_flags = {}
    shared_flags = {}
    for group in ("tail", "shelf", "halo"):
        ff, sf = _make_fix_overrides([group], "shared")
        fix_flags.update(ff)
        shared_flags.update(sf)
    SCANS[scan_id] = {
        "description": description,
        "fix_flags": fix_flags,
        "shared_flags": shared_flags,
        "adc_bin_width": adc_bin_width,
        "dnl_mode": dnl_mode,
        "crossval": crossval,
        "dnl_overrides": dnl_overrides or {},
        "config_overrides": config_overrides or {},
    }


# bin_width=1: single-channel resolution  - maximum DNL sensitivity
_define_binning_scan("E1", "bin1 control (no DNL)",
                     adc_bin_width=1, dnl_mode="none")
_define_binning_scan("E2", "bin1 + Fourier DNL (crossval)",
                     adc_bin_width=1, dnl_mode="fourier", crossval=True)
_define_binning_scan("E3", "bin1 + Fourier DNL (forced)",
                     adc_bin_width=1, dnl_mode="fourier", crossval=False)
_define_binning_scan("E4", "bin1 + Bandpass DNL (crossval)",
                     adc_bin_width=1, dnl_mode="bandpass", crossval=True)

# bin_width=2: intermediate  - partial DNL averaging
_define_binning_scan("E5", "bin2 control (no DNL)",
                     adc_bin_width=2, dnl_mode="none")
_define_binning_scan("E6", "bin2 + Fourier DNL (crossval)",
                     adc_bin_width=2, dnl_mode="fourier", crossval=True)

# E7-E10: DNL parameter exploration at bin_width=1
# All forced (no crossval) for fast iteration, comparing against E1 control.

# E7: 3 iterations of Fourier DNL  - does iterating improve convergence?
_define_binning_scan("E7", "bin1 + Fourier DNL 3iter (forced)",
                     adc_bin_width=1, dnl_mode="fourier", crossval=False,
                     dnl_overrides={"iterations": 3})

# E8: Extended periods  - add non-power-of-2 harmonics for richer basis
_define_binning_scan("E8", "bin1 + Fourier DNL extended (forced)",
                     adc_bin_width=1, dnl_mode="fourier", crossval=False,
                     dnl_overrides={
                         "fourier_periods_codes": [
                             3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32,
                             48, 64, 96, 128, 256, 512,
                         ],
                     })

# E9: Narrower smooth window  - preserves more medium-scale structure
_define_binning_scan("E9", "bin1 + Fourier DNL narrow-window (forced)",
                     adc_bin_width=1, dnl_mode="fourier", crossval=False,
                     dnl_overrides={"smooth_window": 31})

# E10: Kitchen sink  - 3 iter + extended periods + narrow window
_define_binning_scan("E10", "bin1 + Fourier DNL max (forced)",
                     adc_bin_width=1, dnl_mode="fourier", crossval=False,
                     dnl_overrides={
                         "iterations": 3,
                         "smooth_window": 31,
                         "fourier_periods_codes": [
                             3, 4, 5, 6, 8, 10, 12, 16, 20, 24, 32,
                             48, 64, 96, 128, 256, 512,
                         ],
                     })


# E11-E13: Hybrid DNL (Fourier + bandpass) at bin_width=1
# Iteration 1: Fourier captures periodic ADC structure
# Iteration 2+: bandpass captures individual bin spikes

_define_binning_scan("E11", "bin1 + hybrid DNL 2iter (crossval)",
                     adc_bin_width=1, dnl_mode="hybrid", crossval=True,
                     dnl_overrides={"iterations": 2})

_define_binning_scan("E12", "bin1 + hybrid DNL 3iter (forced)",
                     adc_bin_width=1, dnl_mode="hybrid", crossval=False,
                     dnl_overrides={"iterations": 3})

_define_binning_scan("E13", "bin1 + hybrid DNL 2iter narrow (forced)",
                     adc_bin_width=1, dnl_mode="hybrid", crossval=False,
                     dnl_overrides={"iterations": 2, "smooth_window": 31})

# E14: Pure bandpass at bin1 for comparison (no Fourier step)
_define_binning_scan("E14", "bin1 + bandpass DNL 1iter (crossval)",
                     adc_bin_width=1, dnl_mode="bandpass", crossval=True)

# E15-E19: Overfitting reduction  - wider smooth windows for bandpass/hybrid
# Current E11/E14 use smooth_window=61 → overfitting ~2500 NLL.
# Wider windows reduce effective DOF → less model-misfit absorption.

_define_binning_scan("E15", "bin1 + hybrid 2iter sw=121 (crossval)",
                     adc_bin_width=1, dnl_mode="hybrid", crossval=True,
                     dnl_overrides={"iterations": 2, "smooth_window": 121})

_define_binning_scan("E16", "bin1 + hybrid 2iter sw=181 (crossval)",
                     adc_bin_width=1, dnl_mode="hybrid", crossval=True,
                     dnl_overrides={"iterations": 2, "smooth_window": 181})

_define_binning_scan("E17", "bin1 + hybrid 2iter sw=301 (crossval)",
                     adc_bin_width=1, dnl_mode="hybrid", crossval=True,
                     dnl_overrides={"iterations": 2, "smooth_window": 301})

_define_binning_scan("E18", "bin1 + bandpass 1iter sw=121 (crossval)",
                     adc_bin_width=1, dnl_mode="bandpass", crossval=True,
                     dnl_overrides={"smooth_window": 121})

_define_binning_scan("E19", "bin1 + bandpass 1iter sw=181 (crossval)",
                     adc_bin_width=1, dnl_mode="bandpass", crossval=True,
                     dnl_overrides={"smooth_window": 181})


# F-series: Two-stage pipeline  - Fourier DNL at full resolution, then rebin
# Estimates DNL at bin_width=1 (captures periodic SAR ADC structure),
# applies correction, then rebins to suppress incoherent per-bin spikes.

_define_binning_scan("F1", "Full-res Fourier + FD rebin (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={"post_dnl_rebin": "fd"})

_define_binning_scan("F2", "Full-res Fourier + rebin 3 (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={"post_dnl_rebin": 3})

_define_binning_scan("F3", "Full-res Fourier + rebin 5 (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={"post_dnl_rebin": 5})

_define_binning_scan("F4", "Full-res Fourier + rebin 10 (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={"post_dnl_rebin": 10})

_define_binning_scan("F5", "Full-res Fourier + 5w bin width (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={"post_dnl_rebin": "5w"})


# G-series: Fourier period sweep  - find the minimal basis that passes crossval.
# All use full-res Fourier DNL + rebin 3 (matching F2 for direct comparison).
# Progressively add shorter periods to find where crossval starts failing.

_define_binning_scan("G1", "Fourier [128,256,512] + rebin3 (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={
                         "post_dnl_rebin": 3,
                         "fourier_periods_codes": [128, 256, 512],
                     })

_define_binning_scan("G2", "Fourier [64,128,256,512] + rebin3 (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={
                         "post_dnl_rebin": 3,
                         "fourier_periods_codes": [64, 128, 256, 512],
                     })

_define_binning_scan("G3", "Fourier [32,64,128,256,512] + rebin3 (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={
                         "post_dnl_rebin": 3,
                         "fourier_periods_codes": [32, 64, 128, 256, 512],
                     })

_define_binning_scan("G4", "Fourier [16,32,64,128,256,512] + rebin3 (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={
                         "post_dnl_rebin": 3,
                         "fourier_periods_codes": [16, 32, 64, 128, 256, 512],
                     })

_define_binning_scan("G5", "Fourier [256,512] + rebin3 (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={
                         "post_dnl_rebin": 3,
                         "fourier_periods_codes": [256, 512],
                     })

# G6: Auto-select  - give full period set, let per-period crossval pick winners
_define_binning_scan("G6", "Fourier AUTO-SELECT + rebin3 (crossval)",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={
                         "post_dnl_rebin": 3,
                         "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
                     })

# G7: Extended period set including non-power-of-2 candidates
# SAR ADC DNL is expected at base-2 periods, but capacitor mismatch or
# layout effects can produce structure at intermediate periods too.
# Let the per-period crossval sort out which are real.
_define_binning_scan("G7", "Fourier AUTO-SELECT extended periods + rebin3",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides={
                         "post_dnl_rebin": 3,
                         "fourier_periods_codes": [
                             4, 6, 8, 10, 12, 16, 20, 24, 32,
                             40, 48, 64, 96, 128, 192, 256, 384, 512,
                         ],
                     })


# ═══════════════════════════════════════════════════════════════════════
# Phase H: Calibration polynomial order
# ═══════════════════════════════════════════════════════════════════════
# Test whether nonlinear calibration absorbs the smooth residual structure
# that the Fourier DNL can't (and shouldn't) capture. All use the validated
# 2-term Fourier DNL from G6 as baseline.

_H_DNL = {
    "post_dnl_rebin": 3,
    "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
}

# H1: Linear calibration baseline (same as G6 for comparison)
_define_binning_scan("H1", "Linear cal + Fourier auto-select",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides=_H_DNL)

# H2: Quadratic calibration (Po210 + Po218 + Po214 = 3 points)
_define_binning_scan("H2", "Quadratic cal + Fourier auto-select",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides=_H_DNL,
                     config_overrides={"calibration": {"use_quadratic": True}})

# H3: Cubic calibration (Po210 + Po218 + Po216 + Po214 = 4 points)
_define_binning_scan("H3", "Cubic cal + Fourier auto-select",
                     adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
                     dnl_overrides=_H_DNL,
                     config_overrides={"calibration": {"use_quadratic": "cubic"}})

# H4: Quadratic calibration, no DNL (isolate calibration effect)
_define_binning_scan("H4", "Quadratic cal, no DNL",
                     adc_bin_width=10, dnl_mode="none",
                     config_overrides={"calibration": {"use_quadratic": True}})

# H5: Cubic calibration, no DNL
_define_binning_scan("H5", "Cubic cal, no DNL",
                     adc_bin_width=10, dnl_mode="none",
                     config_overrides={"calibration": {"use_quadratic": "cubic"}})


# ══════════════════════════════════════════════════════════════════
# I-series: Shape parameter scan
# Baseline: quadratic calibration + full-res Fourier DNL (auto-select) + rebin3
# Systematically free shape params (tau, shelf, halo) in shared vs per-isotope mode
# ══════════════════════════════════════════════════════════════════

_I_BASE_OVERRIDES = {
    "calibration": {"use_quadratic": True},
}
_I_DNL_OVERRIDES = {
    "post_dnl_rebin": 3,
    "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
}


def _define_shape_scan(scan_id, description, free_types_modes):
    """Define an I-series shape param scan with quad cal + Fourier DNL baseline."""
    fix_flags = {}
    shared_flags = {}
    for group, mode in free_types_modes:
        ff, sf = _make_fix_overrides([group], mode)
        fix_flags.update(ff)
        shared_flags.update(sf)
    SCANS[scan_id] = {
        "description": description,
        "fix_flags": fix_flags,
        "shared_flags": shared_flags,
        "adc_bin_width": 1,
        "dnl_mode": "full_res_fourier",
        "crossval": True,
        "dnl_overrides": dict(_I_DNL_OVERRIDES),
        "config_overrides": dict(_I_BASE_OVERRIDES),
    }


# I1: Baseline - all shapes fixed
_define_shape_scan("I1", "Quad+DNL baseline (all fixed)", [])

# Single mechanism, shared
_define_shape_scan("I2", "Free tau (shared)", [("tail", "shared")])
_define_shape_scan("I3", "Free shelf (shared)", [("shelf", "shared")])
_define_shape_scan("I4", "Free halo (shared)", [("halo", "shared")])

# Single mechanism, per-isotope
_define_shape_scan("I5", "Free tau (per-iso)", [("tail", "per_isotope")])
_define_shape_scan("I6", "Free shelf (per-iso)", [("shelf", "per_isotope")])
_define_shape_scan("I7", "Free halo (per-iso)", [("halo", "per_isotope")])

# Two mechanisms, shared
_define_shape_scan("I8", "Free tau+shelf (shared)",
                   [("tail", "shared"), ("shelf", "shared")])
_define_shape_scan("I9", "Free tau+halo (shared)",
                   [("tail", "shared"), ("halo", "shared")])
_define_shape_scan("I10", "Free shelf+halo (shared)",
                   [("shelf", "shared"), ("halo", "shared")])

# Two mechanisms, per-isotope
_define_shape_scan("I11", "Free tau+shelf (per-iso)",
                   [("tail", "per_isotope"), ("shelf", "per_isotope")])
_define_shape_scan("I12", "Free tau+halo (per-iso)",
                   [("tail", "per_isotope"), ("halo", "per_isotope")])
_define_shape_scan("I13", "Free shelf+halo (per-iso)",
                   [("shelf", "per_isotope"), ("halo", "per_isotope")])

# All three, shared and per-isotope
_define_shape_scan("I14", "Free all (shared)",
                   [("tail", "shared"), ("shelf", "shared"), ("halo", "shared")])
_define_shape_scan("I15", "Free all (per-iso)",
                   [("tail", "per_isotope"), ("shelf", "per_isotope"),
                    ("halo", "per_isotope")])

# Mixed: shared for one, per-isotope for another (best of both)
_define_shape_scan("I16", "Free tau(shared)+shelf(per-iso)",
                   [("tail", "shared"), ("shelf", "per_isotope")])
_define_shape_scan("I17", "Free tau(shared)+halo(per-iso)",
                   [("tail", "shared"), ("halo", "per_isotope")])
_define_shape_scan("I18", "Free tau(per-iso)+shelf(shared)+halo(shared)",
                   [("tail", "per_isotope"), ("shelf", "shared"), ("halo", "shared")])


def _load_base_config(config_path):
    """Load and return base config as dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _apply_scan_overrides(base_cfg, scan_def):
    """Apply scan-specific overrides to a deep copy of base config."""
    cfg = copy.deepcopy(base_cfg)
    sp = cfg.setdefault("spectral_fit", {})
    flags = sp.setdefault("flags", {})

    # ADC bin width override (E-series binning exploration)
    if "adc_bin_width" in scan_def:
        sp["adc_bin_width"] = scan_def["adc_bin_width"]

    # DNL configuration: Phase A-C disable DNL; Phase D/E enables it
    dnl = sp.setdefault("dnl_correction", {})
    dnl_mode = scan_def.get("dnl_mode", "none")
    crossval = scan_def.get("crossval", True)  # default True
    if dnl_mode == "fourier":
        dnl["enabled"] = True
        dnl["parameterized"] = True
        dnl["crossval"] = crossval
    elif dnl_mode == "bandpass":
        dnl["enabled"] = True
        dnl["parameterized"] = False
        dnl["crossval"] = crossval
    elif dnl_mode == "hybrid":
        dnl["enabled"] = True
        dnl["parameterized"] = "hybrid"
        dnl["crossval"] = crossval
    elif dnl_mode == "full_res_fourier":
        dnl["enabled"] = True
        dnl["parameterized"] = True
        dnl["full_resolution_estimate"] = True
        dnl["crossval"] = crossval
        # post_dnl_rebin is set via dnl_overrides
    else:
        dnl["enabled"] = False

    # Apply extra DNL config overrides (iterations, smooth_window, etc.)
    dnl_overrides = scan_def.get("dnl_overrides", {})
    for k, v in dnl_overrides.items():
        dnl[k] = v

    # Apply shared shape param flags
    shared = sp.setdefault("shared_shape_params", {})
    # Reset all to False first
    for p in ["tau", "f_shelf", "sigma_shelf", "f_halo", "sigma_halo", "tau_halo"]:
        shared[p] = False
    # Then set scan-specific flags
    for p, val in scan_def["shared_flags"].items():
        shared[p] = val

    # Apply fix flag overrides (unfix specific per-isotope params)
    for flag_name, flag_val in scan_def["fix_flags"].items():
        flags[flag_name] = flag_val

    # Disable expensive extra-fit diagnostics for scans (split-half = ~7
    # extra fits, model comparison = ~5 extra fits).  These are for final
    # model validation, not parameter exploration.
    sp["split_half_validation"] = False
    # Skip MINOS profile likelihood for scans (3-5x faster, Hesse is adequate)
    sp["skip_minos"] = True

    # Apply arbitrary top-level config overrides (calibration, analysis, etc.)
    for section_key, overrides in scan_def.get("config_overrides", {}).items():
        section = cfg.setdefault(section_key, {})
        if isinstance(overrides, dict):
            section.update(overrides)
        else:
            cfg[section_key] = overrides

    return cfg


def _write_scan_config(cfg, scan_id):
    """Write scan config to scan_configs/<scan_id>_config.yaml."""
    SCAN_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    path = SCAN_CONFIG_DIR / f"{scan_id}_config.yaml"
    with open(path, "w") as f:
        yaml.dump(cfg, f, default_flow_style=False, sort_keys=False)
    return path


def _run_scan(scan_id, config_path):
    """Run analyze.py with scan config and return output dir."""
    output_dir = RESULTS_DIR / f"scan_{scan_id}"
    cmd = [
        sys.executable, str(BASE_DIR / "analyze.py"),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
    ]
    logger.info("Running scan %s: %s", scan_id, " ".join(cmd))
    t0 = time.time()
    result = subprocess.run(
        cmd, capture_output=True, text=True, cwd=str(BASE_DIR),
    )
    elapsed = time.time() - t0
    logger.info("Scan %s completed in %.1f min (exit code %d)",
                scan_id, elapsed / 60, result.returncode)
    if result.returncode != 0:
        logger.error("Scan %s FAILED:\nstdout: %s\nstderr: %s",
                     scan_id, result.stdout[-2000:], result.stderr[-2000:])
    # Always save stderr for debugging (logger.info messages from analyze.py)
    stderr_path = output_dir / f"{scan_id}_stderr.log"
    try:
        stderr_path.write_text(result.stderr[-50000:] if result.stderr else "", encoding="utf-8")
    except Exception:
        pass
    return output_dir, result.returncode, elapsed


def _find_summary(output_dir):
    """Find summary.json in the output directory (may be in a timestamped subdir)."""
    for root, dirs, files in os.walk(output_dir):
        if "summary.json" in files:
            return Path(root) / "summary.json"
    return None


def _extract_metrics(summary_path):
    """Extract key metrics from summary.json."""
    with open(summary_path, "r") as f:
        s = json.load(f)

    # Metrics may be at top level or nested under spectral_fit
    sf = s.get("spectral_fit", {})
    def _get(key, default=float("nan")):
        return s.get(key, sf.get(key, default))

    n_free = _get("n_free_params", 0)
    nll = _get("nll", float("nan"))
    chi2_ndf = _get("chi2_ndf", float("nan"))
    ndf = _get("ndf", 0)
    aic = 2 * n_free + 2 * nll if nll == nll else float("nan")  # nan check

    pull = _get("pull_diagnostics", {})
    if isinstance(pull, dict):
        pull_sigma = pull.get("pull_sigma", float("nan"))
        acf_lag1 = pull.get("autocorr_lag1", float("nan"))
        dw = pull.get("durbin_watson", float("nan"))
        runs_z = pull.get("runs_test_z", float("nan"))
        rebin_pull_sigma = pull.get("rebin10_pull_sigma", float("nan"))
    else:
        pull_sigma = acf_lag1 = dw = runs_z = rebin_pull_sigma = float("nan")

    cov_method = _get("covariance_method", "unknown")
    hesse_ok = cov_method in ("hesse", "numerical_hessian")

    split = _get("split_half_validation", {})
    max_z_shape = split.get("max_z_shape", float("nan")) if isinstance(split, dict) else float("nan")

    # Fit validation diagnostics (GoF p-values, deviance, per-peak)
    fv = _get("fit_validation", {})
    if not isinstance(fv, dict):
        fv = {}
    chi2_pvalue = fv.get("chi2_pvalue", float("nan"))
    deviance_ndf = fv.get("deviance_ndf", float("nan"))
    deviance_pvalue = fv.get("deviance_pvalue", float("nan"))
    bc_ndf = fv.get("baker_cousins_ndf", float("nan"))
    bc_pvalue = fv.get("baker_cousins_pvalue", float("nan"))
    n_at_bound = fv.get("n_at_bound", 0)

    # Fit timing breakdown
    ft = _get("fit_timers", {})
    if not isinstance(ft, dict):
        ft = {}

    # Centroid refit tracking
    cr = _get("centroid_refit", {})
    if not isinstance(cr, dict):
        cr = {}

    return {
        "n_free": n_free,
        "nll": nll,
        "aic": aic,
        "chi2_ndf": chi2_ndf,
        "ndf": ndf,
        "chi2_pvalue": chi2_pvalue,
        "deviance_ndf": deviance_ndf,
        "deviance_pvalue": deviance_pvalue,
        "bc_ndf": bc_ndf,
        "bc_pvalue": bc_pvalue,
        "pull_sigma": pull_sigma,
        "rebin10_pull_sigma": rebin_pull_sigma,
        "acf_lag1": acf_lag1,
        "dw_statistic": dw,
        "runs_test_z": runs_z,
        "cov_method": cov_method,
        "hesse_ok": hesse_ok,
        "n_at_bound": n_at_bound,
        "max_z_shape": max_z_shape,
        "fit_time_initial_s": ft.get("initial_migrad_s", float("nan")),
        "fit_time_dnl_s": ft.get("dnl_s", float("nan")),
        "fit_time_cov_minos_s": ft.get("covariance_minos_s", float("nan")),
        "fit_time_total_s": ft.get("total_s", float("nan")),
        "fit_nll_calls": ft.get("total_nll_calls", 0),
        "minos_method": ft.get("minos_method", "unknown"),
        "centroid_refit_triggered": cr.get("centroid_refit_triggered", False),
        "centroid_refit_accepted": cr.get("centroid_refit_accepted", False),
    }


def main():
    parser = argparse.ArgumentParser(description="Run shape parameter scans")
    parser.add_argument("--config", default=str(BASE_DIR / "config.yaml"),
                        help="Base config file")
    parser.add_argument("--scans", default=None,
                        help="Comma-separated scan IDs (default: all)")
    args = parser.parse_args()

    base_cfg = _load_base_config(args.config)

    # Determine which scans to run
    if args.scans:
        scan_ids = [s.strip() for s in args.scans.split(",")]
    else:
        scan_ids = list(SCANS.keys())

    # Run scans
    all_results = []
    for scan_id in scan_ids:
        if scan_id not in SCANS:
            logger.warning("Unknown scan ID: %s, skipping", scan_id)
            continue

        scan_def = SCANS[scan_id]
        logger.info("=" * 60)
        logger.info("SCAN %s: %s", scan_id, scan_def["description"])
        logger.info("=" * 60)

        cfg = _apply_scan_overrides(base_cfg, scan_def)
        config_path = _write_scan_config(cfg, scan_id)
        output_dir, returncode, elapsed = _run_scan(scan_id, config_path)

        result = {
            "scan_id": scan_id,
            "description": scan_def["description"],
            "returncode": returncode,
            "elapsed_min": round(elapsed / 60, 1),
        }

        if returncode == 0:
            summary_path = _find_summary(output_dir)
            if summary_path:
                metrics = _extract_metrics(summary_path)
                result.update(metrics)
                _ft_total = metrics.get("fit_time_total_s", float("nan"))
                _ft_minos = metrics.get("minos_method", "?")
                _cr = "REFIT" if metrics.get("centroid_refit_triggered") else "no"
                logger.info(
                    "Scan %s: chi2/ndf=%.2f, NLL=%.1f, AIC=%.1f, "
                    "n_free=%d, pull_sigma=%.3f, rebin10=%.3f, "
                    "cov=%s, max_z_shape=%.2f, "
                    "fit_time=%.0fs (init=%.0fs dnl=%.0fs cov+minos=%.0fs [%s]), "
                    "centroid_refit=%s",
                    scan_id, metrics["chi2_ndf"], metrics["nll"],
                    metrics["aic"], metrics["n_free"],
                    metrics["pull_sigma"], metrics["rebin10_pull_sigma"],
                    metrics["cov_method"], metrics["max_z_shape"],
                    _ft_total,
                    metrics.get("fit_time_initial_s", float("nan")),
                    metrics.get("fit_time_dnl_s", float("nan")),
                    metrics.get("fit_time_cov_minos_s", float("nan")),
                    _ft_minos, _cr,
                )
            else:
                logger.warning("No summary.json found for scan %s", scan_id)
        else:
            logger.error("Scan %s failed with exit code %d", scan_id, returncode)

        all_results.append(result)

    # Write CSV summary
    csv_path = BASE_DIR / "scan_results.csv"
    if all_results:
        fieldnames = list(all_results[0].keys())
        # Ensure all fields present
        for r in all_results:
            for k in r:
                if k not in fieldnames:
                    fieldnames.append(k)
        with open(csv_path, "w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(all_results)
        logger.info("Scan results written to %s", csv_path)

    # Print ranked AIC table
    ranked = [r for r in all_results if "aic" in r and r.get("aic") == r.get("aic")]
    ranked.sort(key=lambda r: r.get("aic", float("inf")))

    print("\n" + "=" * 90)
    print("SCAN RESULTS - Ranked by AIC")
    print("=" * 90)
    print(f"{'Rank':>4} {'ID':>4} {'Description':<40} {'n_free':>6} "
          f"{'chi2/ndf':>8} {'AIC':>14} {'pull_σ':>7} {'rb10_σ':>7} "
          f"{'cov':>8} {'z_max':>6}")
    print("-" * 90)
    for i, r in enumerate(ranked, 1):
        print(f"{i:>4} {r['scan_id']:>4} {r['description']:<40} "
              f"{r.get('n_free', '?'):>6} "
              f"{r.get('chi2_ndf', float('nan')):>8.2f} "
              f"{r.get('aic', float('nan')):>14.1f} "
              f"{r.get('pull_sigma', float('nan')):>7.3f} "
              f"{r.get('rebin10_pull_sigma', float('nan')):>7.3f} "
              f"{r.get('cov_method', '?'):>8} "
              f"{r.get('max_z_shape', float('nan')):>6.2f}")
    print("=" * 90)

    if ranked:
        best = ranked[0]
        print(f"\nBest scan: {best['scan_id']}  - {best['description']}")
        print(f"  AIC={best.get('aic', '?'):.1f}, chi2/ndf={best.get('chi2_ndf', '?'):.2f}, "
              f"n_free={best.get('n_free', '?')}")
        if len(ranked) > 1:
            delta_aic = ranked[1].get("aic", 0) - best.get("aic", 0)
            print(f"  ΔAIC to next best ({ranked[1]['scan_id']}): {delta_aic:.1f}")
        print(f"\nTo run Phase D (DNL) scans:")
        print(f"  python run_scan.py --scans D1,D5")
        print(f"  python run_scan.py --scans D1,D2,D3,D4,D5,D6")


if __name__ == "__main__":
    main()
