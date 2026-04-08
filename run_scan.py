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


# ══════════════════════════════════════════════════════════════════
# J-series: Comprehensive binning × DNL sweep with current model
# ══════════════════════════════════════════════════════════════════
# Preserves the base config's model (tau linear, asym quadratic, beta
# coincidence, shared shelf/halo, right tail, Unknown1, Po216 passive).
# Only varies binning and DNL settings.  Diagnostics enabled for key runs.
#
# Estimated runtime: ~24 hours total
# ══════════════════════════════════════════════════════════════════

def _define_j_scan(scan_id, description, adc_bin_width,
                   dnl_mode="none", crossval=True,
                   dnl_overrides=None, config_overrides=None,
                   enable_diagnostics=False):
    """Define a J-series scan that preserves the base config model."""
    SCANS[scan_id] = {
        "description": description,
        "fix_flags": {},
        "shared_flags": {},
        "preserve_shared": True,      # don't reset shared shape params
        "adc_bin_width": adc_bin_width,
        "dnl_mode": dnl_mode,
        "crossval": crossval,
        "dnl_overrides": dnl_overrides or {},
        "config_overrides": config_overrides or {},
        "enable_diagnostics": True,   # always full diagnostics for J-series
    }


# ── J1-J7: Binning sweep, no DNL ─────────────────────────────────
# Establishes baseline chi2/ndf vs bin width. Wider bins average out
# DNL but lose spectral resolution; narrow bins expose DNL artifacts.

_define_j_scan("J1",  "bin1 no-DNL",  adc_bin_width=1)
_define_j_scan("J2",  "bin2 no-DNL",  adc_bin_width=2)
_define_j_scan("J3",  "bin3 no-DNL",  adc_bin_width=3)
_define_j_scan("J4",  "bin5 no-DNL",  adc_bin_width=5)
_define_j_scan("J5",  "bin10 no-DNL (baseline)", adc_bin_width=10,
               enable_diagnostics=True)
_define_j_scan("J6",  "bin15 no-DNL", adc_bin_width=15)
_define_j_scan("J7",  "bin20 no-DNL", adc_bin_width=20)

# ── J8-J14: Fourier DNL with crossval at each binning ────────────
# Tests whether periodic ADC structure improves the fit at each bin width.

_define_j_scan("J8",  "bin1 Fourier DNL (xval)",  adc_bin_width=1,
               dnl_mode="fourier", crossval=True)
_define_j_scan("J9",  "bin2 Fourier DNL (xval)",  adc_bin_width=2,
               dnl_mode="fourier", crossval=True)
_define_j_scan("J10", "bin3 Fourier DNL (xval)",  adc_bin_width=3,
               dnl_mode="fourier", crossval=True)
_define_j_scan("J11", "bin5 Fourier DNL (xval)",  adc_bin_width=5,
               dnl_mode="fourier", crossval=True)
_define_j_scan("J12", "bin10 Fourier DNL (xval)", adc_bin_width=10,
               dnl_mode="fourier", crossval=True)
_define_j_scan("J13", "bin15 Fourier DNL (xval)", adc_bin_width=15,
               dnl_mode="fourier", crossval=True)
_define_j_scan("J14", "bin20 Fourier DNL (xval)", adc_bin_width=20,
               dnl_mode="fourier", crossval=True)

# ── J15-J22: Full-resolution Fourier + rebin ─────────────────────
# Two-stage: estimate DNL at bin_width=1, apply, rebin to coarser.
# This is the most physically motivated approach.

_define_j_scan("J15", "FullRes Fourier + rebin 3 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 3})
_define_j_scan("J16", "FullRes Fourier + rebin 5 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 5})
_define_j_scan("J17", "FullRes Fourier + rebin 10 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10},
               enable_diagnostics=True)
_define_j_scan("J18", "FullRes Fourier + rebin 15 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 15})
_define_j_scan("J19", "FullRes Fourier + rebin 20 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 20})
_define_j_scan("J20", "FullRes Fourier + FD rebin (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": "fd"})

# ── J21-J24: Full-res Fourier + rebin, period subsets ────────────
# Which Fourier harmonics matter? Auto-select via per-period crossval.

_define_j_scan("J21", "FullRes Fourier [256,512] + rb10 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [256, 512]})
_define_j_scan("J22", "FullRes Fourier [128,256,512] + rb10 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [128, 256, 512]})
_define_j_scan("J23", "FullRes Fourier [64..512] + rb10 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [64, 128, 256, 512]})
_define_j_scan("J24", "FullRes AUTO-SELECT all periods + rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512]},
               enable_diagnostics=True)

# ── J25-J28: Three-stage binning (full-res → shape → physics) ────

_define_j_scan("J25", "3-stage rb3/rb15 FullRes Fourier (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 3},
               config_overrides={"spectral_fit": {
                   "three_stage_binning": {"enabled": True,
                                           "stage2_rebin": 3,
                                           "stage3_rebin": 15}}})
_define_j_scan("J26", "3-stage rb5/rb15 FullRes Fourier (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 5},
               config_overrides={"spectral_fit": {
                   "three_stage_binning": {"enabled": True,
                                           "stage2_rebin": 5,
                                           "stage3_rebin": 15}}})
_define_j_scan("J27", "3-stage rb3/rb10 FullRes Fourier (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 3},
               config_overrides={"spectral_fit": {
                   "three_stage_binning": {"enabled": True,
                                           "stage2_rebin": 3,
                                           "stage3_rebin": 10}}})
_define_j_scan("J28", "3-stage rb5/rb20 FullRes Fourier (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 5},
               config_overrides={"spectral_fit": {
                   "three_stage_binning": {"enabled": True,
                                           "stage2_rebin": 5,
                                           "stage3_rebin": 20}}})

# ── J29-J30: Extended Fourier periods (non-power-of-2) ───────────

_define_j_scan("J29", "FullRes extended periods + rb10 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [
                                  4, 6, 8, 10, 12, 16, 20, 24, 32,
                                  40, 48, 64, 96, 128, 192, 256, 384, 512]})
_define_j_scan("J30", "FullRes extended periods + rb5 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 5,
                              "fourier_periods_codes": [
                                  4, 6, 8, 10, 12, 16, 20, 24, 32,
                                  40, 48, 64, 96, 128, 192, 256, 384, 512]})

# ── J31-J34: Calibration polynomial sweep ────────────────────────
# Does nonlinear calibration help at the current model?

_define_j_scan("J31", "bin10 Quadratic cal, no DNL", adc_bin_width=10,
               config_overrides={"calibration": {"use_quadratic": True}})
_define_j_scan("J32", "bin10 Cubic cal, no DNL", adc_bin_width=10,
               config_overrides={"calibration": {"use_quadratic": "cubic"}})
_define_j_scan("J33", "FullRes Fourier + rb10 Quad cal (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512]},
               config_overrides={"calibration": {"use_quadratic": True}},
               enable_diagnostics=True)
_define_j_scan("J34", "FullRes Fourier + rb10 Cubic cal (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512]},
               config_overrides={"calibration": {"use_quadratic": "cubic"}})

# ── J35-J36: DNL iteration sweep ─────────────────────────────────

_define_j_scan("J35", "FullRes Fourier 2iter + rb10 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10, "iterations": 2,
                              "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512]})
_define_j_scan("J36", "FullRes Fourier 3iter + rb10 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10, "iterations": 3,
                              "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512]})

# ── J37-J38: Diagnostics runs on best candidates ─────────────────
# Full diagnostics (MINOS + split-half) on the most promising configs.

_define_j_scan("J37", "bin10 no-DNL FULL DIAG", adc_bin_width=10,
               enable_diagnostics=True)
_define_j_scan("J38", "FullRes Fourier + rb5 FULL DIAG",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 5,
                              "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512]},
               enable_diagnostics=True)

# J39: FullRes Fourier + rebin 1 (no rebinning — stay at full ADC resolution)
_define_j_scan("J39", "FullRes Fourier + rebin 1 (xval)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 1})

# ── J40-J42: Forced single/pair periods, NO crossval ─────────────
# Bypass auto-selection entirely — just apply these periods and see
# what each one does to the fit.

_define_j_scan("J40", "FullRes [256] forced rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [256]})
_define_j_scan("J41", "FullRes [512] forced rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [512]})
_define_j_scan("J42", "FullRes [256,512] forced rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [256, 512]})


_define_j_scan("J43", "FullRes [16] forced rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]})
_define_j_scan("J44", "FullRes [1024] forced rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [1024]})
_define_j_scan("J45", "FullRes [2048] forced rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [2048]})

# ── J46-J48: Combination period scans ────────────────────────────
# Test whether 256/512 add value on top of period 16.
_define_j_scan("J46", "FullRes [16,256] forced rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16, 256]})
_define_j_scan("J47", "FullRes [16,512] forced rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16, 512]})
_define_j_scan("J48", "FullRes [16,256,512] forced rb10",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16, 256, 512]})

# ── J51-J52: FullRes rebin with fixed bkg handoff ───────────────
# Test the fix_bkg_from_prelim fix.
_define_j_scan("J51", "FullRes crossval rb10 + fix_bkg",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"post_dnl_rebin": 10},
               config_overrides={"spectral_fit": {"fix_bkg_from_prelim": "fix"}})
_define_j_scan("J52", "FullRes [16] rb10 + fix_bkg",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {"fix_bkg_from_prelim": "fix"}})

# ── J53-J54: Fix ALL params from prelim bin1 ────────────────────
# Lock bkg + peaks + shapes from bin1 fit. Only DNL correction changes.
_define_j_scan("J53", "FullRes [16] rb10 + fix ALL from prelim",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "seed_peaks_from_prelim": "fix",
                   "seed_shape_from_prelim": "fix",
               }})
_define_j_scan("J54", "FullRes no-DNL rb10 + fix ALL from prelim",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": []},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "seed_peaks_from_prelim": "fix",
                   "seed_shape_from_prelim": "fix",
               }})

# ── J55-J56: Partial handoff tests ────────────────────────────────
# J55: fix bkg + peaks from prelim, let shapes seed (not fixed)
_define_j_scan("J55", "FullRes [16] rb10 + fix bkg+peaks",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "seed_peaks_from_prelim": "fix",
                   "seed_shape_from_prelim": "seed",
               }})
# J56: fix peaks only from prelim, let bkg and shapes float/seed
_define_j_scan("J56", "FullRes [16] rb10 + fix peaks only",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "seed",
                   "seed_peaks_from_prelim": "fix",
                   "seed_shape_from_prelim": "seed",
               }})

# ── J57-J60: Physics model improvements ───────────────────────────
# J57: Baseline with Unknown1 independent shapes, NO Bi212, NO Po212 beta
# Rerun of J52 concept with updated code (Unknown1 no longer sharing shapes).
_J57_extra_peaks = {
    "Unknown1": {
        "energy": 15.8, "sigma": [0.5, 0.1], "amplitude": [500, 300],
        "mu_sigma": 0.3, "mu_bounds": [13.0, 19.0],
        "use_emg": True, "tau": [0.01, 0.01],
        "use_shelf": True, "f_shelf": [0.02, 0.02], "sigma_shelf": [0.1, 0.05],
        "fix": {"sigma": True},
    },
}
_define_j_scan("J57", "FullRes [16] rb10 + fix_bkg (no Bi212)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "extra_peaks": _J57_extra_peaks,
                   "f_beta_Po212_prior": None,
                   "lambda_beta_Po212_prior": None,
                   "share_beta": False,
               }})
# J58: Add Bi-212 (shared shapes), no Po-212 beta yet
_define_j_scan("J58", "FullRes [16] rb10 + fix_bkg + Bi212 (shared shapes)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "f_beta_Po212_prior": None,
                   "lambda_beta_Po212_prior": None,
                   "share_beta": False,
               }})
# J59: Bi-212 + Po-212 beta tail (full default config)
_define_j_scan("J59", "FullRes [16] rb10 + fix_bkg + Bi212 + Po212 beta",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
               }})
# J60: Full model + fix ALL from prelim
_define_j_scan("J60", "FullRes [16] rb10 + fix ALL + Bi212 + shared beta",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "seed_peaks_from_prelim": "fix",
                   "seed_shape_from_prelim": "fix",
               }})
# J61: Full model + fix bkg+peaks (partial handoff)
_define_j_scan("J61", "FullRes [16] rb10 + fix bkg+peaks + Bi212 + shared beta",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "seed_peaks_from_prelim": "fix",
               }})
# J62: Full model at bin_width=1 (no rebin) — ground truth reference
_define_j_scan("J62", "bin1 [16] + Bi212 + shared beta",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {}})
# J64: Shared beta but NO Bi-212 — isolate beta tail effect
_define_j_scan("J64", "FullRes [16] rb10 + fix_bkg + shared beta (no Bi212)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "extra_peaks": _J57_extra_peaks,
               }})


# ── J65-J69: Regression & robustness ──────────────────────────────
# J65: Bi-212 (tight mu) + shared beta + fix_bkg — retest with constrained Bi-212
_define_j_scan("J65", "FullRes [16] rb10 + fix_bkg + Bi212(tight) + shared beta",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
               }})
# J66: Same as J65 + fix ALL from prelim
_define_j_scan("J66", "FullRes [16] rb10 + fix ALL + Bi212(tight) + shared beta",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "seed_peaks_from_prelim": "fix",
                   "seed_shape_from_prelim": "fix",
               }})
# J67: Shared beta + fix_bkg, NO DNL — measure DNL impact
_define_j_scan("J67", "FullRes no-DNL rb10 + fix_bkg + shared beta (no Bi212)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": []},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "extra_peaks": _J57_extra_peaks,
               }})
# J68: Shared beta + fix_bkg + [16,256] — does 256 still hurt?
_define_j_scan("J68", "FullRes [16,256] rb10 + fix_bkg + shared beta (no Bi212)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 10,
                              "fourier_periods_codes": [16, 256]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "extra_peaks": _J57_extra_peaks,
               }})
# J69: Shared beta + fix_bkg + rb16 (multiple of DNL period 16)
_define_j_scan("J69", "FullRes [16] rb16 + fix_bkg + shared beta (no Bi212)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"post_dnl_rebin": 16,
                              "fourier_periods_codes": [16]},
               config_overrides={"spectral_fit": {
                   "fix_bkg_from_prelim": "fix",
                   "extra_peaks": _J57_extra_peaks,
               }})


# ── J70-J74: Bin1 only (no rebin), Fourier DNL [16] ──────────────
# Po-216 now has use_emg/shelf/halo=true, receives shared shapes.
# No post_dnl_rebin — fit directly at bin_width=1 after DNL correction.

# J70: bin1 + [16] DNL + Bi-212 + shared beta + Po-216 full shapes
_define_j_scan("J70", "bin1 [16] DNL + Bi212 + shared beta + Po216 shapes",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"fourier_periods_codes": [16],
                              "post_dnl_rebin": 1},
               config_overrides={"spectral_fit": {}})
# J71: bin1 + [16] DNL + shared beta only (no Bi-212)
_define_j_scan("J71", "bin1 [16] DNL + shared beta (no Bi212)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"fourier_periods_codes": [16],
                              "post_dnl_rebin": 1},
               config_overrides={"spectral_fit": {
                   "extra_peaks": _J57_extra_peaks,
               }})
# J72: bin1 + [16] DNL + Bi-212 + shared beta + NO Unknown1
_J72_extra_peaks = {
    "Bi212": {
        "energy": 6.051, "sigma": [0.12, 0.03], "mu_sigma": 0.005,
        "mu_bounds": [6.03, 6.07],
        "use_emg": True, "tau": [0.08, 0.05],
        "use_shelf": True, "f_shelf": [0.05, 0.05], "sigma_shelf": [0.25, 0.15],
        "use_halo": True, "f_halo": [0.04, 0.10], "sigma_halo": [0.24, 0.12],
        "tau_halo": [0.05, 0.05],
        "amplitude": [500, 500],
        "amplitude_linked": {"reference": "Po212", "ratio": 0.5609},
        "fix": {"sigma": True},
    },
}
_define_j_scan("J72", "bin1 [16] DNL + Bi212 + shared beta (no Unknown1)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"fourier_periods_codes": [16],
                              "post_dnl_rebin": 1},
               config_overrides={"spectral_fit": {
                   "extra_peaks": _J72_extra_peaks,
               }})
# J73: bin1 + NO DNL + Bi-212 + shared beta (baseline without DNL)
_define_j_scan("J73", "bin1 no-DNL + Bi212 + shared beta",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"fourier_periods_codes": [],
                              "post_dnl_rebin": 1},
               config_overrides={"spectral_fit": {}})
# J74: bin1 + [16] DNL, no Bi-212, no shared beta (minimal model)
_define_j_scan("J74", "bin1 [16] DNL only (no Bi212, no beta sharing)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"fourier_periods_codes": [16],
                              "post_dnl_rebin": 1},
               config_overrides={"spectral_fit": {
                   "extra_peaks": _J57_extra_peaks,
                   "share_beta": False,
                   "f_beta_Po212_prior": None,
                   "lambda_beta_Po212_prior": None,
               }})


# ── J75: Full auto-select crossval at bin1 ────────────────────────
# Same model as J70 (Bi-212 + shared beta + Po-216 shapes) but with
# the full SAR ADC period set [4..512] and per-period crossval to
# auto-select which frequencies are real hardware DNL vs noise.
_define_j_scan("J75", "bin1 AUTO-SELECT DNL [4..512] + Bi212 + shared beta",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
                              "post_dnl_rebin": 1},
               config_overrides={"spectral_fit": {}})

# ── J76-J77: Quadratic calibration tests ─────────────────────────
# The linear calibration has 16 keV RMS residual (4 peaks excl Po216).
# Quadratic drops to 4 keV RMS (F=14.2, significant).  a2 < 0 means
# dE/dADC DECREASES at higher ADC — energy bins are ~5% narrower at
# Po-212 vs Po-210.  This mismatch could create smooth systematic
# residuals that long-period DNL (256, 512) absorbs.

# J76: Quadratic cal + full auto-select DNL
_define_j_scan("J76", "bin1 QUAD-CAL + AUTO-SELECT DNL [4..512]",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=True,
               dnl_overrides={"fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
                              "post_dnl_rebin": 1},
               config_overrides={
                   "calibration": {"use_quadratic": True},
                   "spectral_fit": {},
               })

# J77: Quadratic cal + period [16] only (compare vs J70)
_define_j_scan("J77", "bin1 QUAD-CAL + [16] DNL",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"fourier_periods_codes": [16],
                              "post_dnl_rebin": 1},
               config_overrides={
                   "calibration": {"use_quadratic": True},
                   "spectral_fit": {},
               })

# J78: Quadratic cal + NO DNL (isolate calibration effect)
_define_j_scan("J78", "bin1 QUAD-CAL + no DNL",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={"fourier_periods_codes": [],
                              "post_dnl_rebin": 1},
               config_overrides={
                   "calibration": {"use_quadratic": True},
                   "spectral_fit": {},
               })


# ═══════════════════════════════════════════════════════════════════════
# J79-J85: Peak shape improvement scans
# ═══════════════════════════════════════════════════════════════════════
# Long-wavelength residual oscillation (period ~160 codes) matches the
# Po210-Po218 peak spacing (161 codes).  This is peak shape model error,
# not DNL.  Test whether better shapes absorb the structure.
#
# All use bin1 + [16] DNL (same as J70 baseline) to isolate shape effects.

_J_SHAPE_DNL = {"fourier_periods_codes": [16], "post_dnl_rebin": 1}

# --- Strategy 1: Per-isotope shape parameters (un-share) ---

# J79: Per-isotope shelf (f_shelf, sigma_shelf free per isotope)
# Un-share shelf params: each isotope gets its own f_shelf, sigma_shelf.
# This adds 2*(n_iso-1) free params but lets each peak's low-energy shelf
# match independently.
_define_j_scan("J79", "bin1 [16] DNL + per-isotope SHELF",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "shared_shape_params": {
                       "tau": False, "f_shelf": False, "sigma_shelf": False,
                       "f_halo": True, "sigma_halo": True, "tau_halo": True,
                   },
               }})

# J80: Per-isotope halo (f_halo, sigma_halo, tau_halo free per isotope)
_define_j_scan("J80", "bin1 [16] DNL + per-isotope HALO",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "shared_shape_params": {
                       "tau": False, "f_shelf": True, "sigma_shelf": True,
                       "f_halo": False, "sigma_halo": False, "tau_halo": False,
                   },
               }})

# J81: Per-isotope shelf AND halo (everything un-shared)
_define_j_scan("J81", "bin1 [16] DNL + per-isotope SHELF+HALO",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "shared_shape_params": {
                       "tau": False, "f_shelf": False, "sigma_shelf": False,
                       "f_halo": False, "sigma_halo": False, "tau_halo": False,
                   },
               }})

# --- Strategy 2: Additional shape component (second Gaussian) ---
# Not yet implemented in fitting.py — placeholder for when we add it.
# J82: reserved for second Gaussian component

# --- Strategy 3: Energy-dependent sigma model ---

# J83: Higher-order sigma asymmetry polynomial (add asym_3 cubic term)
# Currently have asym_0 + asym_1*(E-Eref) + asym_2*(E-Eref)^2.
# Adding cubic allows more flexibility in how sigma_right varies.
_define_j_scan("J83", "bin1 [16] DNL + cubic sigma asymmetry (asym_3)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "asym_energy_model": "cubic",
                   "asym_3_prior": [0.0, 0.02],
               }})

# J84: Per-isotope shelf + cubic sigma asymmetry (combine 1 + 3)
_define_j_scan("J84", "bin1 [16] DNL + per-iso SHELF + cubic asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "shared_shape_params": {
                       "tau": False, "f_shelf": False, "sigma_shelf": False,
                       "f_halo": True, "sigma_halo": True, "tau_halo": True,
                   },
                   "asym_energy_model": "cubic",
                   "asym_3_prior": [0.0, 0.02],
               }})

# J85: Per-isotope shelf + halo + cubic asym (kitchen sink)
_define_j_scan("J85", "bin1 [16] DNL + per-iso SHELF+HALO + cubic asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "shared_shape_params": {
                       "tau": False, "f_shelf": False, "sigma_shelf": False,
                       "f_halo": False, "sigma_halo": False, "tau_halo": False,
                   },
                   "asym_energy_model": "cubic",
                   "asym_3_prior": [0.0, 0.02],
               }})


# --- Strategy 4: Physics-informed per-isotope model ---
# PIN photodiode physics:
# - tau (EMG tail): charge trapping depends on where alpha stops in
#   depletion region. Higher E → deeper penetration → different field.
#   Linear energy model (tau_0 + tau_slope*E) is physically motivated. KEEP.
# - f_shelf: fraction hitting dead-layer edges → geometry → SHARED.
# - sigma_shelf: range of partial collection → energy-dependent → PER-ISOTOPE.
# - f_halo: nuclear scattering cross section → Rutherford ∝ 1/E² → PER-ISOTOPE.
#   (should decrease at higher energies)
# - sigma_halo: scattering angle distribution → weakly energy-dep → SHARED.
# - tau_halo: halo tail → SHARED (same trapping physics).

# J86: Physics-informed: shared f_shelf + tau_halo + sigma_halo,
#       per-isotope sigma_shelf + f_halo
_define_j_scan("J86", "bin1 [16] DNL + physics: per-iso sigma_shelf + f_halo",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "shared_shape_params": {
                       "tau": False,          # linear energy model
                       "f_shelf": True,        # geometry → shared
                       "sigma_shelf": False,   # energy-dependent → per-isotope
                       "f_halo": False,         # Rutherford 1/E² → per-isotope
                       "sigma_halo": True,     # angle distribution → shared
                       "tau_halo": True,       # trapping physics → shared
                   },
               }})

# J87: Like J86 but also un-share sigma_halo (scattering kinematics
#       could make the angular distribution energy-dependent too)
_define_j_scan("J87", "bin1 [16] DNL + physics: per-iso sigma_shelf+f_halo+sigma_halo",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": False,
                       "f_halo": False,
                       "sigma_halo": False,
                       "tau_halo": True,
                   },
               }})

# J88: J86 + cubic sigma asymmetry (physics-informed + energy-dep sigma)
_define_j_scan("J88", "bin1 [16] DNL + physics per-iso + cubic asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": False,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
                   "asym_energy_model": "cubic",
                   "asym_3_prior": [0.0, 0.02],
               }})


# ── J89-J95: Linear energy model scans ──
# These use param(E) = param_0 + param_slope * (E - E_ref) instead of
# per-isotope or shared parameters.  More parameter-efficient than per-isotope
# (2 params vs N_iso) while capturing energy-dependent physics.

# J89: f_halo linear energy model (Rutherford scattering ∝ 1/E², expect negative slope)
# Everything else shared.  tau already has linear model from base config.
_define_j_scan("J89", "bin1 [16] DNL + f_halo linear energy model",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "f_halo_energy_model": "linear",
                   "f_halo_0_prior": [0.10, 0.08],
                   "f_halo_slope_prior": [0.0, 0.05],
                   "shared_shape_params": {
                       "tau": False,  # uses tau linear model
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,  # superseded by linear model
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J90: sigma_shelf linear energy model (alpha range depends on energy)
# f_shelf shared (geometry), f_halo shared.
_define_j_scan("J90", "bin1 [16] DNL + sigma_shelf linear energy model",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "sigma_shelf_energy_model": "linear",
                   "sigma_shelf_0_prior": [0.20, 0.15],
                   "sigma_shelf_slope_prior": [0.0, 0.05],
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": False,  # superseded by linear model
                       "f_halo": True,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J91: f_halo + sigma_shelf both linear (physics-informed pair)
_define_j_scan("J91", "bin1 [16] DNL + f_halo + sigma_shelf linear",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "f_halo_energy_model": "linear",
                   "f_halo_0_prior": [0.10, 0.08],
                   "f_halo_slope_prior": [0.0, 0.05],
                   "sigma_shelf_energy_model": "linear",
                   "sigma_shelf_0_prior": [0.20, 0.15],
                   "sigma_shelf_slope_prior": [0.0, 0.05],
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": False,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J92: ALL shape params linear energy models (maximum energy-dep, minimum params)
# tau(E), f_shelf(E), sigma_shelf(E), f_halo(E), sigma_halo(E), tau_halo(E)
# Only 12 params (6 intercepts + 6 slopes) vs ~30 for full per-isotope
_define_j_scan("J92", "bin1 [16] DNL + ALL shapes linear energy model",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "f_shelf_energy_model": "linear",
                   "f_shelf_0_prior": [0.08, 0.05],
                   "f_shelf_slope_prior": [0.0, 0.03],
                   "sigma_shelf_energy_model": "linear",
                   "sigma_shelf_0_prior": [0.20, 0.15],
                   "sigma_shelf_slope_prior": [0.0, 0.05],
                   "f_halo_energy_model": "linear",
                   "f_halo_0_prior": [0.10, 0.08],
                   "f_halo_slope_prior": [0.0, 0.05],
                   "sigma_halo_energy_model": "linear",
                   "sigma_halo_0_prior": [0.30, 0.20],
                   "sigma_halo_slope_prior": [0.0, 0.05],
                   "tau_halo_energy_model": "linear",
                   "tau_halo_0_prior": [0.10, 0.08],
                   "tau_halo_slope_prior": [0.0, 0.05],
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": False,
                       "sigma_shelf": False,
                       "f_halo": False,
                       "sigma_halo": False,
                       "tau_halo": False,
                   },
               }})

# J93: Physics-informed linear: f_halo(E) + sigma_shelf(E) + cubic asym
_define_j_scan("J93", "bin1 [16] DNL + f_halo+sigma_shelf linear + cubic asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "f_halo_energy_model": "linear",
                   "f_halo_0_prior": [0.10, 0.08],
                   "f_halo_slope_prior": [0.0, 0.05],
                   "sigma_shelf_energy_model": "linear",
                   "sigma_shelf_0_prior": [0.20, 0.15],
                   "sigma_shelf_slope_prior": [0.0, 0.05],
                   "asym_energy_model": "cubic",
                   "asym_3_prior": [0.0, 0.02],
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": False,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J94: ALL linear + cubic asym (kitchen sink energy model)
_define_j_scan("J94", "bin1 [16] DNL + ALL linear + cubic asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "f_shelf_energy_model": "linear",
                   "f_shelf_0_prior": [0.08, 0.05],
                   "f_shelf_slope_prior": [0.0, 0.03],
                   "sigma_shelf_energy_model": "linear",
                   "sigma_shelf_0_prior": [0.20, 0.15],
                   "sigma_shelf_slope_prior": [0.0, 0.05],
                   "f_halo_energy_model": "linear",
                   "f_halo_0_prior": [0.10, 0.08],
                   "f_halo_slope_prior": [0.0, 0.05],
                   "sigma_halo_energy_model": "linear",
                   "sigma_halo_0_prior": [0.30, 0.20],
                   "sigma_halo_slope_prior": [0.0, 0.05],
                   "tau_halo_energy_model": "linear",
                   "tau_halo_0_prior": [0.10, 0.08],
                   "tau_halo_slope_prior": [0.0, 0.05],
                   "asym_energy_model": "cubic",
                   "asym_3_prior": [0.0, 0.02],
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": False,
                       "sigma_shelf": False,
                       "f_halo": False,
                       "sigma_halo": False,
                       "tau_halo": False,
                   },
               }})

# J95: f_shelf linear (test if shelf fraction varies with energy)
_define_j_scan("J95", "bin1 [16] DNL + f_shelf linear energy model",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "f_shelf_energy_model": "linear",
                   "f_shelf_0_prior": [0.08, 0.05],
                   "f_shelf_slope_prior": [0.0, 0.03],
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": False,  # superseded by linear model
                       "sigma_shelf": True,
                       "f_halo": True,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})


# ── J96-J99: Peak shape model experiments ──
# Addressing the unphysical sigma_asym (bifurcated Gaussian).
# Po212 has asym=0.80 → right side 80% wider — no physics basis in PIN diode.
# All scans use physics-informed sharing:
#   tau: linear energy model (penetration depth depends on energy)
#   f_shelf: SHARED (dead-layer geometry, same for all alphas)
#   sigma_shelf: SHARED (dead-layer thickness is a detector property)
#   sigma_halo: SHARED (scattering geometry)
#   tau_halo: SHARED (charge trapping material property)
#   f_halo: per-isotope (Rutherford scattering cross-section varies with E)

# J96: Drop sigma_asym entirely.  All shapes shared (match J70 but no asym).
# Tests whether asym was doing anything useful or just absorbing error.
_define_j_scan("J96", "bin1 [16] DNL + NO sigma_asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J97: No asym + Rutherford f_halo scaling: f_halo(E) = f_halo_ref × (E_ref/E)²
# One parameter captures the physics exactly. Everything else shared.
_define_j_scan("J97", "bin1 [16] DNL + NO asym + Rutherford f_halo (1/E²)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,  # superseded by rutherford model
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J98: WITH asym (baseline) + Rutherford f_halo.
# Tests whether 1/E² scaling helps even keeping the broken asym.
_define_j_scan("J98", "bin1 [16] DNL + asym + Rutherford f_halo (1/E²)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,  # superseded by rutherford model
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J99: No asym + Rutherford f_halo + per-iso f_shelf.
# Tests if f_shelf needs per-iso freedom on top of the Rutherford halo.
_define_j_scan("J99", "bin1 [16] DNL + NO asym + Ruth f_halo + per-iso f_shelf",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": False,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})


# ── J100-J103: Second Gaussian component experiments ──
# Replace unphysical sigma_asym with a symmetric broad Gaussian shoulder.
# f_gauss2 and sigma_gauss2 are shared (detector charge collection property).

# J100: Second Gaussian, NO asym.  Drop the broken asym and replace with
# the physically-correct symmetric broad component.
_define_j_scan("J100", "bin1 [16] DNL + 2nd Gaussian, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.15, 0.10],
                   "sigma_gauss2_prior": [0.25, 0.15],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J101: Second Gaussian + asym (keep both to see if gauss2 absorbs what
# asym was doing, making asym redundant)
_define_j_scan("J101", "bin1 [16] DNL + 2nd Gaussian + asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.15, 0.10],
                   "sigma_gauss2_prior": [0.25, 0.15],
               }})

# J102: Second Gaussian, NO asym + Rutherford f_halo scaling.
# Full physics model: gauss2 for charge collection, 1/E² for scattering.
_define_j_scan("J102", "bin1 [16] DNL + 2nd Gauss + Ruth f_halo, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.15, 0.10],
                   "sigma_gauss2_prior": [0.25, 0.15],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J103: Second Gaussian, NO asym, Rutherford f_halo + per-iso f_shelf
_define_j_scan("J103", "bin1 [16] DNL + 2nd Gauss + Ruth + per-iso f_shelf",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.15, 0.10],
                   "sigma_gauss2_prior": [0.25, 0.15],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": False,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})


# ── J104-J115: Double EMG, Skew-Normal, Combinations ──────────────
# All are ALTERNATIVES to sigma_asym (not used in combination).
# Physics: PIN photodiode produces only LEFT-side asymmetry from charge
# trapping, dead-layer energy loss, and incomplete charge collection.
# sigma_asym was unphysically broadening the RIGHT side; these replace it
# with physically-motivated left-side (or smooth asymmetric) models.

# --- Double EMG: Two left-side trapping time constants ---
# PIN physics: bulk Si charge trapping (fast τ₁~0.05 MeV) plus
# surface/interface state trapping (slow τ₂~0.2-0.4 MeV).
# Both are detector material properties → f_tail2 and tau2 shared.

# J104: Double EMG only, shared tau2 (constant across peaks).
# Simplest double-tail model: one extra fraction + one extra time constant.
_define_j_scan("J104", "bin1 [16] DNL + double EMG, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_double_emg": True,
                   "f_tail2_prior": [0.10, 0.08],
                   "tau2_shared_prior": [0.30, 0.20],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J105: Double EMG with linear tau2 energy model, NO asym.
# Deeper-penetrating alphas (higher E) traverse more silicon, so the
# slow trapping time constant may increase with energy.
# tau2(E) = tau2_0 + tau2_slope × (E - E_ref)
_define_j_scan("J105", "bin1 [16] DNL + double EMG + linear tau2, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_double_emg": True,
                   "f_tail2_prior": [0.10, 0.08],
                   "tau2_energy_model": "linear",
                   "tau2_0_prior": [0.30, 0.20],
                   "tau2_slope_prior": [0.0, 0.05],
                   "tau2_energy_model_E_ref": 7.0,
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J106: Double EMG + Rutherford f_halo, NO asym.
# Full physics model: double trapping + 1/E² scattering.
_define_j_scan("J106", "bin1 [16] DNL + double EMG + Ruth f_halo, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_double_emg": True,
                   "f_tail2_prior": [0.10, 0.08],
                   "tau2_shared_prior": [0.30, 0.20],
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# --- Skew-Normal: Smooth continuous asymmetry ---
# Replaces the discontinuous bifurcated Gaussian with a smooth skew-normal.
# For EMG isotopes: EMG below peak (exponential charge-trapping tail) +
# skew-normal above peak (smooth right-side shaping).
# α < 0 = left-skew (physical for PIN diode), α > 0 = right-skew.
# PIN physics predicts α ~ constant (detector geometry) or mild E-dependence.

# J107: Skew-normal only, shared α, NO asym.
# Single shared skewness parameter — simplest continuous asymmetry.
_define_j_scan("J107", "bin1 [16] DNL + skew-normal (shared), NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_skew_normal": True,
                   "skew_alpha_prior": [0.0, 3.0],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J108: Skew-normal with linear energy model, NO asym.
# α(E) = α₀ + α_slope × (E - E_ref)
# Higher-energy alphas penetrate deeper → different charge collection
# geometry → potentially different skewness.
_define_j_scan("J108", "bin1 [16] DNL + skew-normal (linear E), NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_skew_normal": True,
                   "skew_energy_model": "linear",
                   "skew_alpha_0_prior": [0.0, 3.0],
                   "skew_alpha_slope_prior": [0.0, 2.0],
                   "skew_energy_model_E_ref": 7.0,
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J109: Skew-normal + Rutherford f_halo, NO asym.
_define_j_scan("J109", "bin1 [16] DNL + skew-normal + Ruth f_halo, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_skew_normal": True,
                   "skew_alpha_prior": [0.0, 3.0],
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# --- Combinations: Multiple physics components ---

# J110: Double EMG + skew-normal, NO asym.
# Two trapping tails on left + smooth asymmetric right-side shape.
# Models the most complete picture: two left tails + continuous right shape.
_define_j_scan("J110", "bin1 [16] DNL + double EMG + skew-normal, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_double_emg": True,
                   "f_tail2_prior": [0.10, 0.08],
                   "tau2_shared_prior": [0.30, 0.20],
                   "use_skew_normal": True,
                   "skew_alpha_prior": [0.0, 3.0],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J111: Double EMG + skew-normal + Ruth f_halo, NO asym.
# Most complete physics model without 2nd Gaussian: two trapping tails
# on the left + smooth asymmetric shape + Rutherford scattering.
_define_j_scan("J111", "bin1 [16] DNL + dblEMG + skewN + Ruth, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_double_emg": True,
                   "f_tail2_prior": [0.10, 0.08],
                   "tau2_shared_prior": [0.30, 0.20],
                   "use_skew_normal": True,
                   "skew_alpha_prior": [0.0, 3.0],
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J114: Skew-normal (linear E model) + Ruth f_halo, NO asym.
# Energy-dependent asymmetry + Rutherford scattering — 4 params total.
_define_j_scan("J114", "bin1 [16] DNL + skewN(linE) + Ruth, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_skew_normal": True,
                   "skew_energy_model": "linear",
                   "skew_alpha_0_prior": [0.0, 3.0],
                   "skew_alpha_slope_prior": [0.0, 2.0],
                   "skew_energy_model_E_ref": 7.0,
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J115: Double EMG (linear tau2) + Ruth f_halo, NO asym.
# Energy-dependent second trapping time + Rutherford scattering.
_define_j_scan("J115", "bin1 [16] DNL + dblEMG(linTau2) + Ruth, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_double_emg": True,
                   "f_tail2_prior": [0.10, 0.08],
                   "tau2_energy_model": "linear",
                   "tau2_0_prior": [0.30, 0.20],
                   "tau2_slope_prior": [0.0, 0.05],
                   "tau2_energy_model_E_ref": 7.0,
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})


# ── J116-J119: Beta model improvements ──────────────────────────────
# The beta coincidence model now includes:
#   - Right-EMG shape (Gaussian-smeared onset instead of hard step)
#   - Left-EMG charge trapping (reuses existing tau — zero new params)
#   - Optional backscatter LET continuum (f_beta_bs_shared)
# These improvements are active automatically in all scans (right-EMG +
# left-EMG convolution use existing params).  J116-J119 ADD the
# backscatter continuum and test in combination with best shape models.

# J116: Double EMG + beta backscatter continuum, NO asym.
_define_j_scan("J116", "bin1 [16] DNL + dblEMG + beta backscatter, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_double_emg": True,
                   "f_tail2_prior": [0.10, 0.08],
                   "tau2_shared_prior": [0.30, 0.20],
                   "f_beta_bs_shared_prior": [0.01, 0.02],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J117: Skew-normal + beta backscatter, NO asym.
_define_j_scan("J117", "bin1 [16] DNL + skewN + beta backscatter, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_skew_normal": True,
                   "skew_alpha_prior": [0.0, 3.0],
                   "f_beta_bs_shared_prior": [0.01, 0.02],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J118: Double EMG + skew-normal + Ruth + beta BS, NO asym.
# Most complete physics model: all left-side corrections.
_define_j_scan("J118", "bin1 [16] DNL + dblEMG+skN+Ruth+betaBS, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_double_emg": True,
                   "f_tail2_prior": [0.10, 0.08],
                   "tau2_shared_prior": [0.30, 0.20],
                   "use_skew_normal": True,
                   "skew_alpha_prior": [0.0, 3.0],
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "f_beta_bs_shared_prior": [0.01, 0.02],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J119: Baseline (no asym) + beta backscatter only.
# Tests the beta model improvements in isolation (no new peak shapes).
_define_j_scan("J119", "bin1 [16] DNL + beta backscatter only, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "f_beta_bs_shared_prior": [0.01, 0.02],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})


# ── J120-J127: Improved 2nd Gaussian (oblique incidence model) ──────
# Literature-informed model: the broad component represents oblique-
# incidence alphas traversing more dead layer (more straggling, more
# energy loss).  Key improvements over J100:
#   - Broad component now has EMG tail (same tau as core — same trapping)
#   - Optional LEFT offset delta_E_broad (extra dead layer energy loss)
#   - Optional right-side tail (channeling/Landau high-energy tailing)
# Sharing: f_gauss2, sigma_gauss2, delta_E_broad all SHARED (geometry).

# J120: 2nd Gaussian with EMG tail, NO asym (no offset yet).
# Same as J100 but the broad component now has left-EMG applied.
# Tests whether the EMG on the broad component matters.
_define_j_scan("J120", "bin1 [16] DNL + 2ndG(EMG), NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.20, 0.10],
                   "sigma_gauss2_prior": [0.15, 0.10],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J121: 2nd Gaussian with EMG + LEFT offset, NO asym.
# delta_E_broad lets the broad component centroid shift left.
# Physics: oblique alphas lose more dead layer energy.
_define_j_scan("J121", "bin1 [16] DNL + 2ndG(EMG+offset), NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.20, 0.10],
                   "sigma_gauss2_prior": [0.15, 0.10],
                   "delta_E_broad_prior": [0.03, 0.03],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J122: 2nd Gaussian(EMG+offset) + right-side tail, NO asym.
# Right-tail replaces sigma_asym: physically models channeling
# and Landau fluctuations (Pomme 2015, Shi 2019).
_define_j_scan("J122", "bin1 [16] DNL + 2ndG(EMG+off) + R-tail, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.20, 0.10],
                   "sigma_gauss2_prior": [0.15, 0.10],
                   "delta_E_broad_prior": [0.03, 0.03],
                   "use_tail_right": True,
                   "f_tail_right_prior": [0.03, 0.03],
                   "tau_tail_right_prior": [0.10, 0.08],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})

# J123: 2nd Gaussian(EMG+offset) + Ruth f_halo, NO asym.
_define_j_scan("J123", "bin1 [16] DNL + 2ndG(EMG+off) + Ruth, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.20, 0.10],
                   "sigma_gauss2_prior": [0.15, 0.10],
                   "delta_E_broad_prior": [0.03, 0.03],
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J124: Full physics model: 2ndG(EMG+off) + R-tail + Ruth + beta BS.
# Everything physically motivated, no sigma_asym.
_define_j_scan("J124", "bin1 [16] DNL + 2ndG+Rtail+Ruth+betaBS, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.20, 0.10],
                   "sigma_gauss2_prior": [0.15, 0.10],
                   "delta_E_broad_prior": [0.03, 0.03],
                   "use_tail_right": True,
                   "f_tail_right_prior": [0.03, 0.03],
                   "tau_tail_right_prior": [0.10, 0.08],
                   "f_halo_energy_model": "rutherford",
                   "f_halo_ref_prior": [0.10, 0.08],
                   "f_halo_energy_model_E_ref": 7.0,
                   "f_beta_bs_shared_prior": [0.01, 0.02],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
                   "shared_shape_params": {
                       "tau": False,
                       "f_shelf": True,
                       "sigma_shelf": True,
                       "f_halo": False,
                       "sigma_halo": True,
                       "tau_halo": True,
                   },
               }})

# J125: 2nd Gaussian(EMG+offset) + asym (keep asym to see if
# the right-tail replaces it or if asym still adds value).
_define_j_scan("J125", "bin1 [16] DNL + 2ndG(EMG+off) + asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.20, 0.10],
                   "sigma_gauss2_prior": [0.15, 0.10],
                   "delta_E_broad_prior": [0.03, 0.03],
               }})

# J126: 2nd Gaussian(EMG, no offset) + right-tail only, NO asym.
# Tests right-tail without offset to isolate right-tail contribution.
_define_j_scan("J126", "bin1 [16] DNL + 2ndG(EMG) + R-tail, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.20, 0.10],
                   "sigma_gauss2_prior": [0.15, 0.10],
                   "use_tail_right": True,
                   "f_tail_right_prior": [0.03, 0.03],
                   "tau_tail_right_prior": [0.10, 0.08],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})


# J127: Final J121 model + per-isotope beta (unshared).
# Tests whether unsharing beta for Po-212 (Bi-212 beta endpoint 2.25 MeV)
# vs Po-214 (Bi-214 beta endpoint 3.27 MeV) closes the gap to J125.
# sigma_asym was compensating: sigma_right/sigma = 2.0 at Po-212 energy.
_define_j_scan("J127", "bin1 [16] DNL + 2ndG(EMG+off) + per-iso beta, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   "use_gauss2": True,
                   "f_gauss2_prior": [0.29, 0.10],
                   "sigma_gauss2_prior": [0.14, 0.08],
                   "delta_E_broad_prior": [0.02, 0.02],
                   "share_beta": False,
                   "f_beta_Po214_prior": [0.04, 0.04],
                   "lambda_beta_Po214_prior": [0.13, 0.08],
                   "f_beta_Po212_prior": [0.05, 0.05],
                   "lambda_beta_Po212_prior": [0.25, 0.15],
                   "asym_energy_model": "",
                   "use_sigma_asym": False,
               }})


# ── J128-J132: Full-Fourier DNL + unfrozen/frozen shape ablation ───
# J128 expands Fourier DNL from single period [16] to all SAR bit-cycling
# periods [4..512], unfreezes shapes from prelim (seed not fix), and
# raises the f_beta_Po212 prior to [0.30, 0.20].  J129-J132 are controls.

_J128_DNL = {"fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
             "post_dnl_rebin": 1}

# Shared config for J128-J132: 2ndG + per-iso beta + dropped tau_halo + shared constant tau
# NOTE: shared_shape_params.tau=true and tau_shared_prior are in the base config.
# tau_energy_model=constant disables the linear tau_0+tau_slope model.
_J128_SPEC = {
    "tau_energy_model": "constant",          # disables linear tau_0+tau_slope
    "tau_shared_prior": [0.097, 0.05],       # from J127 fitted tau_0
    "tau_halo_shared_prior": [0.0, 0.001],
    "use_gauss2": True,
    "f_gauss2_prior": [0.29, 0.10],
    "sigma_gauss2_ratio_prior": [1.4, 0.5],
    "delta_E_broad_prior": [0.02, 0.02],
    "share_beta": False,
    "f_beta_Po214_prior": [0.04, 0.04],
    "lambda_beta_Po214_prior": [0.13, 0.08],
    # Po-212: Bi-212 beta endpoint 2.25 MeV (shorter lambda than Po-214)
    "f_beta_Po212_prior": [0.04, 0.04],
    "lambda_beta_Po212_prior": [0.10, 0.08],
    "asym_energy_model": "",
    "use_sigma_asym": False,
    "flags": {"fix_tau_halo_shared": True},
}

# J128: Full Fourier + seeded (unfrozen) shapes
_define_j_scan("J128",
               "bin1 full-Fourier DNL + 2ndG(EMG+off) + per-iso beta + unfrozen shapes, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={"spectral_fit": {
                   **_J128_SPEC,
                   "seed_shape_from_prelim": "seed",
               }})

# J129: NO DNL control — same shapes/beta as J128 but DNL disabled.
_define_j_scan("J129",
               "bin1 NO DNL + 2ndG(EMG+off) + per-iso beta + unfrozen shapes, NO asym",
               adc_bin_width=1, dnl_mode="none", crossval=False,
               config_overrides={"spectral_fit": {
                   **_J128_SPEC,
                   "seed_shape_from_prelim": "seed",
               }})

# J130: Period-16-only DNL control — same as J128 but single Fourier period.
_define_j_scan("J130",
               "bin1 period-16-only DNL + 2ndG(EMG+off) + per-iso beta + unfrozen shapes, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J_SHAPE_DNL,
               config_overrides={"spectral_fit": {
                   **_J128_SPEC,
                   "seed_shape_from_prelim": "seed",
               }})

# J131: Fixed shapes control — same as J127 but with raised f_beta cap
# and full Fourier DNL.  Shapes fixed from prelim (not seeded).
_define_j_scan("J131",
               "bin1 full-Fourier DNL + 2ndG(EMG+off) + per-iso beta + FIXED shapes, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={"spectral_fit": {
                   **_J128_SPEC,
                   "seed_shape_from_prelim": "fix",
               }})

# J132: Shapes OFF prelim — main fit finds shapes independently (no seeding).
_define_j_scan("J132",
               "bin1 full-Fourier DNL + 2ndG(EMG+off) + per-iso beta + shapes OFF prelim, NO asym",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={"spectral_fit": {
                   **_J128_SPEC,
                   "seed_shape_from_prelim": "off",
               }})

# ── J133-J136: Post-DNL rebin sweep ──────────────────────────────────
# Same model as J128 (full Fourier + seeded shapes) but with increasing
# post_dnl_rebin factors.  DNL correction is always estimated at full ADC
# resolution (bin_width=1), then the corrected histogram is rebinned before
# the main fit.  This shows how much DNL structure gets averaged out and
# how the smooth model misfit looks when rebinned.

for _rb_id, _rb_factor in [("J133", 5), ("J134", 10), ("J135", 16), ("J136", 32)]:
    _define_j_scan(
        _rb_id,
        f"full-Fourier DNL + rebin x{_rb_factor} + 2ndG(EMG+off) + per-iso beta + seed shapes",
        adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
        dnl_overrides={
            "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
            "post_dnl_rebin": _rb_factor,
        },
        config_overrides={"spectral_fit": {
            **_J128_SPEC,
            "seed_shape_from_prelim": "seed",
        }},
    )

# ── J137: Spectral fit only — validates Unknown1 auto-fix + diagnostics period fix
# Same model as J128 but with the code changes: auto-fix extra-peak shapes
# when rebinning, and code-domain diagnostics now use validated periods.
# Disables bridge and radon inference for speed.
_define_j_scan("J137",
               "bin1 full-Fourier DNL + 2ndG(EMG+off) + per-iso beta + seed shapes (fixes test)",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={
                   "spectral_fit": {
                       **_J128_SPEC,
                       "seed_shape_from_prelim": "seed",
                   },
                   "radon_inference": {"enabled": False},
                   "lucas_bridge": {"enabled": False},
               })

# ── J138: Full pipeline + lucas bridge (no ambient file)
# Same spectral fit as J137, plus lucas cell assay bridge and efficiency.
# Radon inference enabled but external_rn set to constant fallback (no pico file).
_define_j_scan("J138",
               "bin1 full-Fourier DNL + 2ndG(EMG+off) + per-iso beta + seed shapes + lucas bridge",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={
                   "spectral_fit": {
                       **_J128_SPEC,
                       "seed_shape_from_prelim": "seed",
                   },
                   "radon_inference": {
                       "enabled": True,
                       "external_rn": {
                           "mode": "constant",
                           "constant_bq_per_m3": 120.0,
                       },
                   },
                   "lucas_bridge": {"enabled": True},
               })

# ── J139: Full pipeline + lucas bridge + ambient rates from Pico XLS
# Same as J138 but ensures external_rn reads from the Pico40L spreadsheet.
_define_j_scan("J139",
               "bin1 full-Fourier DNL + 2ndG(EMG+off) + seed shapes + bridge + ambient Pico",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={
                   "spectral_fit": {
                       **_J128_SPEC,
                       "seed_shape_from_prelim": "seed",
                   },
                   "radon_inference": {
                       "enabled": True,
                       "external_rn": {
                           "mode": "file",
                           "file_path": "rad_4996_pico40l.xls",
                           "fallback_bq_per_m3": 120.0,
                           "time_columns": {
                               "year": "Year",
                               "month": "Month",
                               "day": "Day",
                               "hour": "Hour",
                               "minute": "Minute",
                               "year_format": "two_digit",
                           },
                           "value_column": "Radon Bq/m^3",
                           "tz": "America/Toronto",
                           "interpolation": "ffill",
                           "max_gap_seconds": 21600,
                       },
                   },
                   "lucas_bridge": {"enabled": True},
               })


# ── J140: Rebin x10 with Unknown1 auto-fix ────────────────────────
# Same model as J128, but post_dnl_rebin=10 to test the auto-fix of
# extra-peak (Unknown1) shape params at coarse binning.  Compare to
# J134 (rebin x10 WITHOUT auto-fix) to verify improvement.
_define_j_scan("J140",
               "full-Fourier DNL + rebin x10 + auto-fix Unknown1 + seed shapes",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={
                   "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
                   "post_dnl_rebin": 10,
               },
               config_overrides={"spectral_fit": {
                   **_J128_SPEC,
                   "seed_shape_from_prelim": "seed",
               }})

# ── J141: Rebin x32 with Unknown1 auto-fix ────────────────────────
# Extreme rebin to compare against J136 (rebin x32, chi2/ndf=15.6).
_define_j_scan("J141",
               "full-Fourier DNL + rebin x32 + auto-fix Unknown1 + seed shapes",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={
                   "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
                   "post_dnl_rebin": 32,
               },
               config_overrides={"spectral_fit": {
                   **_J128_SPEC,
                   "seed_shape_from_prelim": "seed",
               }})


# ── J142: sigma_gauss2 ratio + dropped f_beta_Po212 ───────────────
# Tests the multiplicative sigma_gauss2_ratio reparameterization
# AND f_beta_Po212 dropped. Full pipeline with bridge + ambient.
_define_j_scan("J142",
               "bin1 sigma_gauss2_ratio + no f_beta_Po212 + bridge + ambient",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={
                   "spectral_fit": {
                       **_J128_SPEC,
                       "seed_shape_from_prelim": "seed",
                   },
                   "radon_inference": {"enabled": True},
                   "lucas_bridge": {"enabled": True},
               })

# ── J143: sigma_gauss2 ratio + KEEP f_beta_Po212 ─────────────────
# Same sigma_gauss2 fix but KEEPS f_beta_Po212 to see if the improved
# peak shapes change the beta coincidence significance.
_define_j_scan("J143",
               "bin1 sigma_gauss2_ratio + WITH f_beta_Po212 + bridge + ambient",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={
                   "spectral_fit": {
                       **_J128_SPEC,
                       "seed_shape_from_prelim": "seed",
                       # Re-enable f_beta_Po212 to test if sigma fix changes it
                       "f_beta_Po212_prior": [0.30, 0.20],
                       "lambda_beta_Po212_prior": [0.25, 0.15],
                   },
                   "radon_inference": {"enabled": True},
                   "lucas_bridge": {"enabled": True},
               })

# ── J144: sigma_gauss2 ratio + rebin x10 ─────────────────────────
# Tests the Unknown1 auto-fix + sigma_gauss2 ratio at rebin x10.
_define_j_scan("J144",
               "sigma_gauss2_ratio + rebin x10 + auto-fix Unknown1",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides={
                   "fourier_periods_codes": [4, 8, 16, 32, 64, 128, 256, 512],
                   "post_dnl_rebin": 10,
               },
               config_overrides={"spectral_fit": {
                   **_J128_SPEC,
                   "seed_shape_from_prelim": "seed",
               }})

# ── J145: Template fitting (per-bin spectral extraction) ──────────
# Full pipeline with template fitting instead of ROI counting.
# Uses sigma_gauss2 ratio + bridge + ambient + radon inference.
_define_j_scan("J145",
               "template fitting + bridge + ambient",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={
                   "spectral_fit": {
                       **_J128_SPEC,
                       "seed_shape_from_prelim": "seed",
                   },
                   "time_fit": {
                       "extraction_method": "template",
                       "template_rebin": 20,
                       "template_min_counts": 30,
                       "float_centroids": True,
                       "fix_weak_isotopes": True,
                   },
                   "radon_inference": {"enabled": True},
                   "lucas_bridge": {"enabled": True},
               })

# ── J146: Template fitting (ROI baseline for comparison) ──────────
# Same config as J145 but with ROI counting, for direct comparison.
_define_j_scan("J146",
               "ROI baseline for template comparison",
               adc_bin_width=1, dnl_mode="full_res_fourier", crossval=False,
               dnl_overrides=_J128_DNL,
               config_overrides={
                   "spectral_fit": {
                       **_J128_SPEC,
                       "seed_shape_from_prelim": "seed",
                   },
                   "time_fit": {
                       "extraction_method": "roi",
                   },
                   "radon_inference": {"enabled": True},
                   "lucas_bridge": {"enabled": True},
               })


def _load_base_config(config_path):
    """Load and return base config as dict."""
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def _deep_merge(base: dict, override: dict) -> dict:
    """Recursively merge *override* into *base* (in-place).

    For every key in *override*:
      - If both base[key] and override[key] are dicts, recurse.
      - Otherwise, override[key] replaces base[key].

    Returns *base* for convenience.
    """
    for k, v in override.items():
        if k in base and isinstance(base[k], dict) and isinstance(v, dict):
            _deep_merge(base[k], v)
        else:
            base[k] = v
    return base


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
    if not scan_def.get("preserve_shared", False):
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

    # Disable expensive extra-fit diagnostics unless scan requests them
    if not scan_def.get("enable_diagnostics", False):
        sp["split_half_validation"] = False
        sp["skip_minos"] = True
    else:
        sp["split_half_validation"] = True
        # MINOS is too slow for 30-param scans (hours per run).
        # Always skip unless explicitly requested via enable_minos flag.
        sp["skip_minos"] = not scan_def.get("enable_minos", False)

    # Apply arbitrary top-level config overrides (calibration, analysis, etc.)
    # Uses deep merge so nested dicts (e.g. flags, penalty_priors) are merged
    # rather than replaced.
    for section_key, overrides in scan_def.get("config_overrides", {}).items():
        section = cfg.setdefault(section_key, {})
        if isinstance(overrides, dict):
            _deep_merge(section, overrides)
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
    # Clean previous results to avoid accumulating stale timestamped folders
    if output_dir.exists():
        import shutil
        for child in list(output_dir.iterdir()):
            if child.is_dir():
                try:
                    shutil.rmtree(child)
                    logger.info("Cleaned old result folder: %s", child.name)
                except OSError as _e:
                    logger.warning("Could not remove %s: %s", child.name, _e)
    cmd = [
        sys.executable, str(BASE_DIR / "analyze.py"),
        "--config", str(config_path),
        "--output-dir", str(output_dir),
        "--job-id", scan_id,
        "--overwrite",
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


def _prepare_scan(scan_id, base_cfg):
    """Prepare config for a scan, return (scan_id, config_path, scan_def)."""
    scan_def = SCANS[scan_id]
    cfg = _apply_scan_overrides(base_cfg, scan_def)
    config_path = _write_scan_config(cfg, scan_id)
    return scan_id, config_path, scan_def


def _run_and_collect(scan_id, config_path, scan_def):
    """Run a single scan and collect results dict."""
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

    return result


def main():
    parser = argparse.ArgumentParser(description="Run shape parameter scans")
    parser.add_argument("--config", default=str(BASE_DIR / "config.yaml"),
                        help="Base config file")
    parser.add_argument("--scans", default=None,
                        help="Comma-separated scan IDs (default: all)")
    parser.add_argument("--parallel", "-j", type=int, default=1,
                        help="Number of parallel scan processes (default: 1)")
    args = parser.parse_args()

    base_cfg = _load_base_config(args.config)

    # Determine which scans to run
    if args.scans:
        scan_ids = [s.strip() for s in args.scans.split(",")]
    else:
        scan_ids = list(SCANS.keys())

    # Filter valid scan IDs and prepare configs
    valid_scans = []
    for scan_id in scan_ids:
        if scan_id not in SCANS:
            logger.warning("Unknown scan ID: %s, skipping", scan_id)
            continue
        sid, cpath, sdef = _prepare_scan(scan_id, base_cfg)
        valid_scans.append((sid, cpath, sdef))

    n_parallel = max(1, args.parallel)
    logger.info("Running %d scans with %d parallel workers", len(valid_scans), n_parallel)

    # Run scans (parallel or serial)
    all_results = []
    if n_parallel == 1:
        # Serial mode (original behavior)
        for scan_id, config_path, scan_def in valid_scans:
            logger.info("=" * 60)
            logger.info("SCAN %s: %s", scan_id, scan_def["description"])
            logger.info("=" * 60)
            result = _run_and_collect(scan_id, config_path, scan_def)
            all_results.append(result)
    else:
        # Parallel mode using concurrent.futures
        from concurrent.futures import ProcessPoolExecutor, as_completed
        logger.info("=" * 60)
        logger.info("PARALLEL MODE: %d workers", n_parallel)
        logger.info("=" * 60)
        futures = {}
        with ProcessPoolExecutor(max_workers=n_parallel) as executor:
            for scan_id, config_path, scan_def in valid_scans:
                logger.info("Submitting scan %s: %s", scan_id, scan_def["description"])
                future = executor.submit(
                    _run_and_collect, scan_id, config_path, scan_def,
                )
                futures[future] = scan_id
            for future in as_completed(futures):
                scan_id = futures[future]
                try:
                    result = future.result()
                    all_results.append(result)
                    logger.info(
                        "Scan %s finished: chi2/ndf=%.2f, elapsed=%.1f min",
                        scan_id,
                        result.get("chi2_ndf", float("nan")),
                        result.get("elapsed_min", 0),
                    )
                except Exception as e:
                    logger.error("Scan %s raised exception: %s", scan_id, e)
                    all_results.append({
                        "scan_id": scan_id,
                        "description": SCANS[scan_id]["description"],
                        "returncode": -1,
                        "elapsed_min": 0,
                    })
        # Sort results by scan ID for consistent output
        all_results.sort(key=lambda r: r["scan_id"])

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
