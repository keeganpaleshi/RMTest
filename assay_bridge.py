"""Lucas-cell assay bridge — external-reference calibration stage for RMTest.

Compares Lucas-cell assay reference activity against RMTest's measured
radon activity to derive a bridge factor (calibration ratio).

This is a post-fit stage, separate from ``radon_inference``.
"""

from __future__ import annotations

import hashlib
import logging
import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from assay_reader import LucasCellAssay, filter_assays, parse_assay_csv
from utils.time_utils import to_epoch_seconds

logger = logging.getLogger(__name__)

# Rn-222 half-life in seconds (3.8240 days)
_RN222_HALFLIFE_S = 3.8240 * 86400.0
_RN222_LAMBDA_S = math.log(2) / _RN222_HALFLIFE_S
_RN222_LAMBDA_DAY = math.log(2) / 3.8240

# ── Canonical Lucas-cell constants (from Excel formula chain) ──
# Rn-222 alpha chain: Rn-222→Po-218, Po-218→Pb-214, Po-214→Pb-210
_ALPHA_MULTIPLICITY = 3
# Lucas cell single-alpha scintillation detection efficiency
_LC_SINGLE_ALPHA_EFF = 0.74
# Effective counting efficiency = multiplicity × single-alpha eff
_EFF_COUNTING = _ALPHA_MULTIPLICITY * _LC_SINGLE_ALPHA_EFF  # 2.22
# Dual-path factor: assay samples from one of two parallel paths
_DUAL_PATH_FACTOR = 2
# Sample volume in liters (gas extracted into the 30L assay volume)
_SAMPLE_VOLUME_L = 30.0
# Default monitor volume (from Excel formula: =BD37*600)
_MONITOR_VOLUME_L_DEFAULT = 600.0
# Default trap-to-LC transfer efficiency (from UGboardcalibration!G56)
_TRANSFER_EFF_DEFAULT = 0.3873
# Fractional uncertainty on transfer efficiency (~10%)
_TRANSFER_EFF_REL_UNC = 0.10
# Fractional uncertainty on LC single-alpha efficiency (~5%)
_LC_EFF_REL_UNC = 0.05


def _epoch_to_iso_utc(ts: float) -> str:
    """Return ``ts`` as an ISO-8601 UTC string with ``Z`` suffix."""

    return datetime.fromtimestamp(ts, tz=timezone.utc).isoformat().replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Reference activity from Lucas-cell assay
# ---------------------------------------------------------------------------

def compute_reference_activity_bq(
    assay: LucasCellAssay,
    *,
    recompute: bool = False,
    monitor_volume_l: float = _MONITOR_VOLUME_L_DEFAULT,
) -> tuple[float, float]:
    """Convert a Lucas-cell assay to a reference activity in Bq.

    Parameters
    ----------
    assay : LucasCellAssay
        Parsed assay record.
    recompute : bool
        If True, recompute from raw inputs using the canonical Lucas
        formula chain.  When both recompute and pre-computed spreadsheet
        values are available, the spreadsheet uncertainty is used as a
        floor (it includes volume/flow uncertainty that our partial-
        derivative propagation does not capture).
    monitor_volume_l : float
        Monitor volume in liters for scaling (default 600 L per Excel).

    Returns
    -------
    (reference_activity_bq, uncertainty_bq)
        The Rn-222 activity in the monitor volume implied by the assay.
    """
    lambda_s = math.log(2) / (assay.rn_half_life_days * 86400.0)

    # ── Primary path when recompute=True: canonical formula chain ──
    # The recompute chain (Excel formula replication) gives the correct
    # central value for atoms_in_monitor.  The spreadsheet's
    # atoms_per_liter uses a subtly different formula chain (e.g.
    # different volume scaling) and must NOT be used for the central
    # value when recompute is requested.
    #
    # However, the spreadsheet uncertainty captures volume/flow
    # uncertainties that our partial-derivative propagation does not,
    # so we use max(recompute_rel_unc, spreadsheet_rel_unc) for the
    # uncertainty.
    if recompute:
        result = _recompute_from_raw(assay, monitor_volume_l)
        n_atoms, n_unc = result[0], result[1]
        if not math.isnan(n_atoms) and n_atoms > 0:
            # Use spreadsheet relative uncertainty as floor (it includes
            # volume/flow uncertainty our recompute doesn't capture)
            if not math.isnan(assay.atoms_per_liter_unc) and not math.isnan(assay.atoms_per_liter):
                ss_rel = assay.atoms_per_liter_unc / assay.atoms_per_liter
                recomp_rel = n_unc / n_atoms if n_atoms > 0 else 0.0
                if ss_rel > recomp_rel:
                    logger.debug(
                        "Using spreadsheet rel unc (%.1f%%) > recompute "
                        "(%.1f%%) for %s",
                        ss_rel * 100, recomp_rel * 100, assay.label,
                    )
                    n_unc = n_atoms * ss_rel
            return (n_atoms * lambda_s, n_unc * lambda_s)

    # ── Fallback: pre-computed concentration from spreadsheet ──
    if not math.isnan(assay.atoms_per_liter):
        n = assay.atoms_per_liter * monitor_volume_l
        u = assay.atoms_per_liter_unc * monitor_volume_l if not math.isnan(assay.atoms_per_liter_unc) else math.nan
        return (n * lambda_s, u * lambda_s if not math.isnan(u) else math.nan)

    # ── Fallback: pre-computed atoms_in_monitor (already volume-scaled) ──
    if not math.isnan(assay.atoms_in_monitor):
        n = assay.atoms_in_monitor
        u = assay.atoms_in_monitor_unc if not math.isnan(assay.atoms_in_monitor_unc) else math.nan
        return (n * lambda_s, u * lambda_s if not math.isnan(u) else math.nan)

    # ── Last resort: manual volume scaling from total_atoms_in_assay ──
    n_atoms = assay.total_atoms_in_assay
    n_unc = assay.total_atoms_in_assay_unc
    if math.isnan(n_atoms):
        return (math.nan, math.nan)

    n_atoms, n_unc = _scale_to_monitor(
        assay, n_atoms, n_unc, monitor_volume_l
    )
    if math.isnan(n_atoms):
        return (math.nan, math.nan)

    return (n_atoms * lambda_s, n_unc * lambda_s if not math.isnan(n_unc) else math.nan)


def _scale_to_monitor(
    assay: LucasCellAssay,
    n_atoms: float,
    n_unc: float,
    monitor_volume_l: float,
) -> tuple[float, float]:
    """Scale assay atoms from sampled volume to monitor volume.

    assay_volume = flow_rate_lpm * assay_duration_min.
    concentration = n_atoms / assay_volume.
    monitor_atoms = concentration * monitor_volume_l.
    """
    flow = assay.flow_rate_lpm
    dur = assay.assay_duration_min

    # Default flow rate: 1.0 L/min for most assays
    if math.isnan(flow):
        flow = 1.0
        logger.debug("Using default flow rate 1.0 L/min for %s", assay.label)

    if math.isnan(dur) or dur <= 0 or flow <= 0:
        logger.warning(
            "Cannot compute assay volume for %s (flow=%.2f, dur=%.1f)",
            assay.label, flow, dur,
        )
        return (math.nan, math.nan)

    assay_volume_l = flow * dur
    scale = monitor_volume_l / assay_volume_l
    return (n_atoms * scale, n_unc * scale if not math.isnan(n_unc) else math.nan)


def _recompute_from_raw(
    assay: LucasCellAssay,
    monitor_volume_l: float = _MONITOR_VOLUME_L_DEFAULT,
) -> tuple[float, float]:
    """Canonical Lucas-cell formula chain (replicates Excel exactly).

    Implements the full calculation from raw counts to atoms-in-monitor
    with partial-derivative uncertainty propagation matching the
    "UI Uncertainty" sheet.

    Excel formula chain
    -------------------
    1. net_counts = raw_counts - lc_bkg_cpd × counting_time_days
    2. decayed_in_cell = net_counts / (3 × 0.74)
    3. atoms_at_extraction = decayed / (exp(-λ·t_delay) - exp(-λ·(t_delay+t_c)))
    4. atoms_in_trap = atoms_at_extraction / transfer_eff
    5. atoms_in_assay = atoms_in_trap
    6. atoms_per_liter = atoms_in_assay × 2 / 30
    7. atoms_in_monitor = atoms_per_liter × 600

    Uncertainty sources (partial derivatives)
    -----------------------------------------
    - Poisson counts: σ_N = √N_raw
    - LC background rate: σ_bkg
    - LC single-alpha efficiency: 5% relative
    - Transfer efficiency: 10% relative
    - Delay time: from spreadsheet
    - Counting time: from spreadsheet
    - Decay constant: from half-life uncertainty

    Returns
    -------
    (atoms_in_monitor, atoms_in_monitor_unc)
    """
    N_raw = assay.n_counts
    t_c = assay.counting_time_days
    t_delay = assay.delay_time_days
    transfer_eff = assay.overall_efficiency
    bkg_cpd = assay.lc_background_cpd
    lam = _RN222_LAMBDA_DAY

    # Validate required inputs
    for name, val in [("n_counts", N_raw), ("counting_time", t_c)]:
        if math.isnan(val):
            logger.warning("Cannot recompute: %s is NaN for assay %s", name, assay.label)
            return (math.nan, math.nan)

    if math.isnan(t_delay):
        t_delay = 0.0
        logger.debug("No delay time for %s, assuming 0", assay.label)

    # Use the assay's own transfer efficiency when available.
    # Fall back to the canonical 0.3873 (UGboardcalibration!$G$56) when NaN.
    # Note: the Aug 8 background assay stores 1.42 (composite efficiency)
    # which is correct for its specific calculation chain.
    if math.isnan(transfer_eff) or transfer_eff <= 0:
        transfer_eff = _TRANSFER_EFF_DEFAULT
        logger.debug("Using default transfer efficiency %.4f for %s",
                      transfer_eff, assay.label)

    # ── Step 1: Background-subtracted counts ──
    bkg_counts = bkg_cpd * t_c if not math.isnan(bkg_cpd) else 0.0
    N_net = N_raw - bkg_counts

    if N_net <= 0:
        logger.warning("Negative net counts (%.1f) for assay %s", N_net, assay.label)
        return (0.0, math.nan)

    # ── Step 2: Decayed atoms in cell ──
    # Excel row 25: =(BD23-BD24)/(3*0.74)
    decayed = N_net / _EFF_COUNTING

    # ── Step 3: Atoms at extraction time ──
    # Excel row 32: =BD25/(EXP(-1*BD19*BD11)-EXP(-1*BD19*(BD12+BD11)))
    # where BD19=λ(1/day), BD11=delay(days), BD12=counting_time(days)
    exp_delay = math.exp(-lam * t_delay)
    exp_delay_plus_tc = math.exp(-lam * (t_delay + t_c))
    denom = exp_delay - exp_delay_plus_tc

    if denom <= 0:
        logger.warning("Decay denominator <= 0 for %s", assay.label)
        return (math.nan, math.nan)

    atoms_extraction = decayed / denom

    # ── Step 4: Atoms in trap ──
    # Excel row 33: =BD32/BD20  (transfer_eff = 0.3873)
    atoms_trap = atoms_extraction / transfer_eff

    # ── Step 5: Atoms in assay = atoms in trap (for RM assays) ──
    atoms_assay = atoms_trap

    # ── Step 6: Volume scaling ──
    # Excel row 36: =BD35*2  (dual path factor)
    # Excel row 37: =BD36/30  (per liter)
    atoms_per_liter = atoms_assay * _DUAL_PATH_FACTOR / _SAMPLE_VOLUME_L

    # ── Step 7: Scale to monitor ──
    # Excel row 40: =BD37*600
    atoms_monitor = atoms_per_liter * monitor_volume_l

    # ═══════════════════════════════════════════════════════════════
    # Uncertainty propagation (partial derivatives)
    # ═══════════════════════════════════════════════════════════════
    # The total scaling from decayed → atoms_monitor is:
    #   S = (dual_path / sample_vol) * monitor_vol / transfer_eff / denom
    # So atoms_monitor = (N_net / eff_counting) * S
    # We propagate uncertainties on each input variable.

    # Full chain factor from net_counts to atoms_in_monitor
    chain_factor = (1.0 / _EFF_COUNTING) * (1.0 / denom) * (1.0 / transfer_eff) \
                   * (_DUAL_PATH_FACTOR / _SAMPLE_VOLUME_L) * monitor_volume_l

    # --- σ from Poisson counting statistics ---
    sigma_N_raw = math.sqrt(N_raw)  # Poisson
    # ∂(atoms_monitor)/∂(N_raw) = chain_factor (N_net = N_raw - bkg, linear)
    var_counts = (sigma_N_raw * chain_factor) ** 2

    # --- σ from LC background rate ---
    bkg_unc_cpd = assay.lc_background_unc_cpd if not math.isnan(assay.lc_background_unc_cpd) else 0.0
    sigma_bkg_counts = bkg_unc_cpd * t_c
    # ∂(atoms_monitor)/∂(bkg_counts) = -chain_factor (subtracted)
    var_bkg = (sigma_bkg_counts * chain_factor) ** 2

    # --- σ from LC single-alpha efficiency ---
    # decayed = N_net / (3 * ε_lc), so ∂decayed/∂ε_lc = -N_net / (3 * ε_lc²)
    # Relative: σ_atoms/atoms = σ_ε_lc / ε_lc
    var_lc_eff = (atoms_monitor * _LC_EFF_REL_UNC) ** 2

    # --- σ from transfer efficiency ---
    # atoms_trap = atoms_extraction / η_transfer
    # Relative: σ_atoms/atoms = σ_η / η
    var_transfer = (atoms_monitor * _TRANSFER_EFF_REL_UNC) ** 2

    # --- σ from delay time ---
    # ∂(atoms_extraction)/∂(t_delay) via the denominator:
    # denom = exp(-λ·td) - exp(-λ·(td+tc))
    # ∂denom/∂td = -λ·exp(-λ·td) + λ·exp(-λ·(td+tc)) = -λ·denom... wait
    # Actually: ∂denom/∂td = λ·(-exp(-λ·td) + exp(-λ·(td+tc)))
    #                       = -λ·(exp(-λ·td) - exp(-λ·(td+tc))) = -λ·denom
    # And atoms_extraction = decayed/denom, so
    # ∂(atoms_extraction)/∂td = -decayed·(-λ·denom)/denom² = decayed·λ/denom²·(wrong sign)
    # Let's be more careful:
    #   f(td) = exp(-λ·td) - exp(-λ·(td+tc))
    #   f'(td) = -λ·exp(-λ·td) + λ·exp(-λ·(td+tc)) = -λ·f(td)
    #   ∂(decayed/f)/∂td = decayed · (-f'/f²) = decayed · (λ·f)/(f²) = decayed·λ/f
    #   = atoms_extraction · λ
    d_extraction_d_td = atoms_extraction * lam
    # Scale through the rest of the chain
    scale_extraction_to_monitor = (1.0 / transfer_eff) * (_DUAL_PATH_FACTOR / _SAMPLE_VOLUME_L) * monitor_volume_l
    d_monitor_d_td = d_extraction_d_td * scale_extraction_to_monitor
    sigma_td = assay.delay_time_unc_days if not math.isnan(assay.delay_time_unc_days) else 0.0
    var_delay = (d_monitor_d_td * sigma_td) ** 2

    # --- σ from counting time ---
    # f(tc) = exp(-λ·td) - exp(-λ·(td+tc))
    # f'(tc) = λ·exp(-λ·(td+tc))
    # ∂(decayed/f)/∂tc = -decayed·f'(tc)/f² = -decayed·λ·exp(-λ·(td+tc))/f²
    d_f_d_tc = lam * exp_delay_plus_tc
    d_extraction_d_tc = -decayed * d_f_d_tc / (denom ** 2)
    d_monitor_d_tc = d_extraction_d_tc * scale_extraction_to_monitor
    sigma_tc = assay.counting_time_unc_days if not math.isnan(assay.counting_time_unc_days) else 0.0
    var_tc = (d_monitor_d_tc * sigma_tc) ** 2

    # --- Total variance ---
    # Separate UNCORRELATED (per-assay) from CORRELATED (common systematic).
    # Correlated: LC single-alpha efficiency and transfer efficiency are the
    # same physical constants for every assay — they shift all results
    # coherently and must NOT average down in a weighted mean.
    var_uncorrelated = var_counts + var_bkg + var_delay + var_tc
    var_correlated = var_lc_eff + var_transfer
    total_var = var_uncorrelated + var_correlated
    atoms_monitor_unc = math.sqrt(total_var)

    logger.info(
        "Canonical recompute %s: N_raw=%d, N_net=%.1f, decayed=%.2f, "
        "atoms_extraction=%.2f, atoms_trap=%.2f, atoms/L=%.3f, "
        "atoms_monitor=%.1f ± %.1f",
        assay.label, int(N_raw), N_net, decayed,
        atoms_extraction, atoms_trap, atoms_per_liter,
        atoms_monitor, atoms_monitor_unc,
    )
    # Log uncertainty breakdown
    logger.debug(
        "  Uncertainty breakdown: counts=%.1f, bkg=%.1f, lc_eff=%.1f, "
        "transfer=%.1f, delay=%.1f, counting=%.1f",
        math.sqrt(var_counts), math.sqrt(var_bkg), math.sqrt(var_lc_eff),
        math.sqrt(var_transfer), math.sqrt(var_delay), math.sqrt(var_tc),
    )
    logger.debug(
        "  Uncorrelated: %.1f (%.1f%%)  Correlated: %.1f (%.1f%%)",
        math.sqrt(var_uncorrelated), math.sqrt(var_uncorrelated) / atoms_monitor * 100,
        math.sqrt(var_correlated), math.sqrt(var_correlated) / atoms_monitor * 100,
    )

    return (atoms_monitor, atoms_monitor_unc, math.sqrt(var_uncorrelated), math.sqrt(var_correlated))


# ---------------------------------------------------------------------------
# Extract measured activity from RMTest summary
# ---------------------------------------------------------------------------

_COMPARISON_TARGETS = {
    "baseline_corrected_combined",
    "radon_activity",
    "radon_combined",
    "po214",
    "po218",
}


def extract_measured_activity(
    summary: Mapping[str, Any],
    cfg: dict[str, Any],
) -> tuple[float, float]:
    """Extract the measured radon activity from an RMTest summary.

    Parameters
    ----------
    summary : Mapping
        The RMTest summary dict (or Summary object).
    cfg : dict
        The ``lucas_bridge`` config section.

    Returns
    -------
    (measured_activity_bq, uncertainty_bq)
    """
    target = cfg.get("comparison_target", "baseline_corrected_combined")

    if target == "baseline_corrected_combined":
        return _extract_baseline_corrected_combined(summary, cfg)
    elif target == "radon_combined":
        return _extract_radon_combined(summary)
    elif target == "radon_activity":
        return _extract_radon_activity(summary)
    elif target in ("po214", "po218"):
        return _extract_isotope_activity(summary, target)
    else:
        logger.warning("Unknown comparison_target: %s", target)
        return (math.nan, math.nan)


def _extract_baseline_corrected_combined(
    summary: Mapping[str, Any],
    cfg: dict[str, Any],
) -> tuple[float, float]:
    """Extract baseline-corrected combined radon activity.

    Uses the combined radon result from the baseline-corrected isotope
    activities.  If ``use_unclipped_baseline`` is True (default), uses
    the raw values before negative clamping.
    """
    baseline = summary.get("baseline")
    if not isinstance(baseline, Mapping):
        logger.info("No baseline in summary, falling back to radon_combined")
        return _extract_radon_combined(summary)

    corrected = baseline.get("corrected_activity")
    if not isinstance(corrected, Mapping):
        logger.info("No corrected_activity in baseline, falling back to radon_combined")
        return _extract_radon_combined(summary)

    # Try to get combined radon from radon_combined (which is the
    # inverse-variance weighted Po-214 + Po-218 combination)
    radon_combined = summary.get("radon_combined")
    if isinstance(radon_combined, Mapping):
        val = radon_combined.get("activity_Bq")
        unc = radon_combined.get("unc_Bq")
        if val is not None and unc is not None:
            return (float(val), float(unc))

    # Fall back to single-isotope corrected activity
    # Prefer Po-214 as it typically has better statistics
    for iso in ("Po214", "Po218"):
        entry = corrected.get(iso)
        if isinstance(entry, Mapping):
            val = entry.get("value")
            unc = entry.get("uncertainty")
            if val is not None:
                return (float(val), float(unc) if unc is not None else math.nan)

    return (math.nan, math.nan)


def _extract_radon_combined(summary: Mapping[str, Any]) -> tuple[float, float]:
    """Extract from radon_combined."""
    rc = summary.get("radon_combined")
    if isinstance(rc, Mapping):
        val = rc.get("activity_Bq")
        unc = rc.get("unc_Bq")
        if val is not None:
            return (float(val), float(unc) if unc is not None else math.nan)
    return _extract_radon_activity(summary)


def _extract_radon_activity(summary: Mapping[str, Any]) -> tuple[float, float]:
    """Extract from radon_results."""
    rr = summary.get("radon_results")
    if isinstance(rr, Mapping):
        ra = rr.get("radon_activity_Bq") or rr.get("radon_activity_bq")
        if isinstance(ra, Mapping):
            return (float(ra.get("value", math.nan)),
                    float(ra.get("uncertainty", math.nan)))
    # Try radon dict
    radon = summary.get("radon")
    if isinstance(radon, Mapping):
        val = radon.get("Rn_activity_Bq")
        unc = radon.get("stat_unc_Bq")
        if val is not None:
            return (float(val), float(unc) if unc is not None else math.nan)
    return (math.nan, math.nan)


def _extract_isotope_activity(
    summary: Mapping[str, Any],
    isotope: str,
) -> tuple[float, float]:
    """Extract a single-isotope activity from baseline-corrected outputs."""
    iso_key = isotope.capitalize()  # po214 -> Po214
    if iso_key.startswith("Po"):
        iso_key = f"Po{iso_key[2:]}"

    baseline = summary.get("baseline")
    if isinstance(baseline, Mapping):
        corrected = baseline.get("corrected_activity", {})
        entry = corrected.get(iso_key)
        if isinstance(entry, Mapping):
            val = entry.get("value")
            unc = entry.get("uncertainty")
            if val is not None:
                return (float(val), float(unc) if unc is not None else math.nan)

    return (math.nan, math.nan)


# ---------------------------------------------------------------------------
# Bridge factor computation
# ---------------------------------------------------------------------------

def _compute_bridge_factor(
    ref_bq: float,
    ref_unc: float,
    meas_bq: float,
    meas_unc: float,
) -> tuple[float, float]:
    """Compute bridge_factor = reference / measured with propagated uncertainty.

    Returns (bridge_factor, bridge_factor_unc).
    """
    if math.isnan(ref_bq) or math.isnan(meas_bq):
        return (math.nan, math.nan)
    if meas_bq == 0:
        return (math.inf, math.nan)

    bf = ref_bq / meas_bq

    # Relative uncertainty: delta(bf)/bf = sqrt((dR/R)^2 + (dM/M)^2)
    rel_ref = (ref_unc / ref_bq) if (not math.isnan(ref_unc) and ref_bq != 0) else 0.0
    rel_meas = (meas_unc / meas_bq) if (not math.isnan(meas_unc) and meas_bq != 0) else 0.0
    bf_unc = abs(bf) * math.sqrt(rel_ref**2 + rel_meas**2)

    return (bf, bf_unc)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def _find_skid_background(
    all_assays: list[LucasCellAssay],
    cfg: dict[str, Any],
) -> LucasCellAssay | None:
    """Find the assay-system (skid) background assay.

    The skid background represents radon emanation from the titan trap,
    tubing, and other components of the assay system.  This must be
    subtracted from RM assay results to isolate the monitor's internal
    emanation.

    The background assay is identified by:
    1. ``skid_background.label`` config substring match, or
    2. ``skid_background.date`` config (YYYY-MM-DD), or
    3. ``skid_background.column_index`` config value.

    Parameters
    ----------
    all_assays : list[LucasCellAssay]
        All parsed assays (before selection filtering).
    cfg : dict
        The ``lucas_bridge`` config section.

    Returns
    -------
    LucasCellAssay or None
    """
    skid_cfg = cfg.get("skid_background")
    if not isinstance(skid_cfg, Mapping):
        return None

    label_match = skid_cfg.get("label", "").lower()
    date_match = skid_cfg.get("date")  # "YYYY-MM-DD"
    col_match = skid_cfg.get("column_index")

    for assay in all_assays:
        if col_match is not None and assay.column_index == col_match:
            return assay
        if date_match and assay.assay_date:
            if assay.assay_date.strftime("%Y-%m-%d") == date_match:
                if label_match and label_match in assay.label.lower():
                    return assay
                if not label_match:
                    return assay
        if label_match and label_match in assay.label.lower():
            if date_match is None:
                return assay

    if label_match or date_match or col_match is not None:
        logger.warning(
            "Skid background assay not found (label=%r, date=%r, col=%r)",
            label_match, date_match, col_match,
        )
    return None


def _subtract_skid_background(
    assay: LucasCellAssay,
    skid_bg: LucasCellAssay,
    monitor_volume_l: float,
    recompute: bool = False,
) -> tuple[float, float, float, float]:
    """Subtract skid background from assay and return net reference activity.

    Prefers spreadsheet pre-computed atoms_per_liter when available (these
    include full uncertainty propagation from the Excel sheet, including
    volume/flow uncertainty).  Falls back to _recompute_from_raw when
    the spreadsheet values are missing.

    Returns
    -------
    (net_atoms_in_monitor, net_atoms_unc, net_activity_bq, net_activity_unc_bq)
    """
    lambda_s = math.log(2) / (assay.rn_half_life_days * 86400.0)

    # Get assay atoms.
    # For UI-connected assays, the recompute formula chain (dual_path=2,
    # sample_vol=30 L) doesn't apply — use the spreadsheet directly.
    lbl_lower = assay.label.lower()
    is_ui_connected = "ui connected" in lbl_lower or "ui through" in lbl_lower

    if is_ui_connected and not math.isnan(assay.atoms_per_liter):
        assay_atoms = assay.atoms_per_liter * monitor_volume_l
        assay_unc = (assay.atoms_per_liter_unc * monitor_volume_l
                     if not math.isnan(assay.atoms_per_liter_unc) else 0.0)
    elif recompute:
        _r = _recompute_from_raw(assay, monitor_volume_l)
        assay_atoms, assay_unc = _r[0], _r[1]
        # Use spreadsheet relative uncertainty as floor
        if (not math.isnan(assay_atoms) and assay_atoms > 0
                and not math.isnan(assay.atoms_per_liter_unc)
                and not math.isnan(assay.atoms_per_liter)
                and assay.atoms_per_liter > 0):
            ss_rel = assay.atoms_per_liter_unc / assay.atoms_per_liter
            rc_rel = assay_unc / assay_atoms if assay_atoms > 0 else 0.0
            if ss_rel > rc_rel:
                assay_unc = assay_atoms * ss_rel
    elif not math.isnan(assay.atoms_per_liter):
        assay_atoms = assay.atoms_per_liter * monitor_volume_l
        assay_unc = (assay.atoms_per_liter_unc * monitor_volume_l
                     if not math.isnan(assay.atoms_per_liter_unc) else 0.0)
    else:
        return (math.nan, math.nan, math.nan, math.nan)

    # Get background atoms: prefer recompute, fall back to spreadsheet
    if recompute:
        _r = _recompute_from_raw(skid_bg, monitor_volume_l)
        bg_atoms, bg_unc = _r[0], _r[1]
        if (not math.isnan(bg_atoms) and bg_atoms > 0
                and not math.isnan(skid_bg.atoms_per_liter_unc)
                and not math.isnan(skid_bg.atoms_per_liter)
                and skid_bg.atoms_per_liter > 0):
            ss_rel = skid_bg.atoms_per_liter_unc / skid_bg.atoms_per_liter
            rc_rel = bg_unc / bg_atoms if bg_atoms > 0 else 0.0
            if ss_rel > rc_rel:
                bg_unc = bg_atoms * ss_rel
    elif not math.isnan(skid_bg.atoms_per_liter):
        bg_atoms = skid_bg.atoms_per_liter * monitor_volume_l
        bg_unc = (skid_bg.atoms_per_liter_unc * monitor_volume_l
                  if not math.isnan(skid_bg.atoms_per_liter_unc) else 0.0)
    else:
        return (math.nan, math.nan, math.nan, math.nan)

    if math.isnan(assay_atoms) or math.isnan(bg_atoms):
        return (math.nan, math.nan, math.nan, math.nan)

    net_atoms = assay_atoms - bg_atoms
    net_atoms_unc = math.sqrt(assay_unc**2 + bg_unc**2)

    net_activity = net_atoms * lambda_s
    net_activity_unc = net_atoms_unc * lambda_s

    return (net_atoms, net_atoms_unc, net_activity, net_activity_unc)


def _normalise_downtime_periods(
    downtime_periods: Sequence[Mapping[str, Any]] | Sequence[tuple[float, float]] | None,
) -> list[tuple[float, float]]:
    """Return sorted downtime periods as ``[(start_unix, end_unix), ...]``."""

    periods: list[tuple[float, float]] = []
    if not downtime_periods:
        return periods

    for period in downtime_periods:
        if isinstance(period, Mapping):
            start_raw = period.get("start_unix", period.get("start"))
            end_raw = period.get("end_unix", period.get("end"))
        elif isinstance(period, Sequence) and len(period) == 2:
            start_raw, end_raw = period
        else:
            continue
        try:
            start_f = float(start_raw)
            end_f = float(end_raw)
        except (TypeError, ValueError):
            continue
        if np.isfinite(start_f) and np.isfinite(end_f) and end_f > start_f:
            periods.append((start_f, end_f))

    periods.sort()
    return periods


def _interval_overlaps_periods(
    start_unix: float,
    end_unix: float,
    periods: Sequence[tuple[float, float]],
) -> bool:
    """Return True when ``[start_unix, end_unix)`` overlaps any downtime period."""

    for period_start, period_end in periods:
        if start_unix < period_end and end_unix > period_start:
            return True
    return False


def _target_inside_period(
    target_unix: float,
    periods: Sequence[tuple[float, float]],
) -> bool:
    """Return True when ``target_unix`` lies inside a downtime period."""

    for period_start, period_end in periods:
        if period_start <= target_unix <= period_end:
            return True
    return False


def _combine_isotope_activity_bins(
    isotope_series: dict[str, list[dict[str, float]]],
    isotopes: Sequence[str],
    cal_window_rel_unc: Mapping[str, float] | None,
) -> list[dict[str, float]]:
    """Combine isotope time slices into shared per-bin activity measurements."""

    if not isotope_series:
        return []

    if cal_window_rel_unc is None:
        cal_window_rel_unc = {}

    combined: dict[tuple[float, float], dict[str, float]] = {}
    for iso in isotopes:
        entries = isotope_series.get(iso, [])
        for entry in entries:
            try:
                t = float(entry.get("t"))
                dt = float(entry.get("dt"))
                counts = float(entry.get("counts", 0.0))
            except (TypeError, ValueError):
                continue
            if not np.isfinite(t) or not np.isfinite(dt) or dt <= 0.0:
                continue
            if not np.isfinite(counts):
                continue

            key = (t, dt)
            item = combined.setdefault(
                key,
                {
                    "t": t,
                    "dt": dt,
                    "start": t - 0.5 * dt,
                    "end": t + 0.5 * dt,
                    "counts_total": 0.0,
                    "var_counts_total": 0.0,
                },
            )

            counts_var = max(counts, 0.0)
            counts_unc = entry.get("counts_unc")
            if counts_unc is not None:
                try:
                    counts_unc_f = float(counts_unc)
                except (TypeError, ValueError):
                    counts_unc_f = float("nan")
                if np.isfinite(counts_unc_f) and counts_unc_f >= 0.0:
                    counts_var = counts_unc_f ** 2

            cal_rel = float(cal_window_rel_unc.get(iso, 0.0))
            if np.isfinite(cal_rel) and cal_rel > 0.0:
                rate = counts / dt
                counts_var += (rate * cal_rel * dt) ** 2

            item["counts_total"] += counts
            item["var_counts_total"] += counts_var

    merged: list[dict[str, float]] = []
    for item in combined.values():
        dt = float(item["dt"])
        counts_total = float(item["counts_total"])
        var_counts_total = float(item["var_counts_total"])
        rate = counts_total / dt if dt > 0 else math.nan
        rate_unc = math.sqrt(max(var_counts_total, 0.0)) / dt if dt > 0 else math.nan
        item["rate"] = rate
        item["rate_unc"] = rate_unc
        merged.append(item)

    merged.sort(key=lambda rec: rec["t"])
    return merged


def _build_activity_segments(
    merged_bins: Sequence[Mapping[str, float]],
    downtime_periods: Sequence[tuple[float, float]],
) -> list[list[dict[str, float]]]:
    """Split merged activity bins into contiguous segments."""

    if not merged_bins:
        return []

    valid_bins: list[dict[str, float]] = []
    for entry in merged_bins:
        start = float(entry["start"])
        end = float(entry["end"])
        if _interval_overlaps_periods(start, end, downtime_periods):
            continue
        valid_bins.append(dict(entry))
    if not valid_bins:
        return []

    median_dt = float(
        np.median([float(entry["dt"]) for entry in valid_bins if entry.get("dt", 0.0) > 0.0])
    ) if valid_bins else 0.0
    if not np.isfinite(median_dt) or median_dt <= 0.0:
        median_dt = 3600.0
    gap_split_s = max(1.5 * median_dt, 300.0)

    segments: list[list[dict[str, float]]] = [[valid_bins[0]]]
    for entry in valid_bins[1:]:
        prev = segments[-1][-1]
        gap_s = float(entry["start"]) - float(prev["end"])
        if gap_s > gap_split_s or _interval_overlaps_periods(prev["end"], entry["start"], downtime_periods):
            segments.append([entry])
        else:
            segments[-1].append(entry)
    return segments


def _extract_activity_fixed_window(
    isotope_series: dict[str, list[dict[str, float]]],
    target_unix: float,
    window_days: float,
    isotopes: Sequence[str],
    cal_window_rel_unc: Mapping[str, float] | None,
) -> tuple[float, float, dict[str, Any]]:
    """Legacy fixed-window measured-activity extraction."""

    half_window_s = float(window_days) * 86400.0
    t_lo = target_unix - half_window_s
    t_hi = target_unix + half_window_s

    merged_bins = _combine_isotope_activity_bins(
        isotope_series,
        isotopes,
        cal_window_rel_unc,
    )
    selected = [
        entry for entry in merged_bins
        if t_lo <= float(entry["t"]) <= t_hi
    ]
    if not selected:
        return math.nan, math.nan, {
            "status": "no_data_in_window",
            "mode": "fixed",
            "target_unix": float(target_unix),
            "window_start_unix": float(t_lo),
            "window_end_unix": float(t_hi),
            "n_bins": 0,
        }

    total_counts = sum(float(entry["counts_total"]) for entry in selected)
    total_var = sum(float(entry["var_counts_total"]) for entry in selected)
    total_dt = sum(float(entry["dt"]) for entry in selected)
    if total_dt <= 0.0:
        return math.nan, math.nan, {
            "status": "invalid_live_time",
            "mode": "fixed",
            "target_unix": float(target_unix),
            "window_start_unix": float(t_lo),
            "window_end_unix": float(t_hi),
            "n_bins": len(selected),
        }

    rate = total_counts / total_dt
    rate_unc = math.sqrt(max(total_var, 0.0)) / total_dt
    details = {
        "status": "ok",
        "mode": "fixed",
        "target_unix": float(target_unix),
        "window_start_unix": float(min(float(entry["start"]) for entry in selected)),
        "window_end_unix": float(max(float(entry["end"]) for entry in selected)),
        "requested_half_window_hours": half_window_s / 3600.0,
        "n_bins": len(selected),
        "rate_bq": float(rate),
        "rate_unc_bq": float(rate_unc),
    }
    return rate, rate_unc, details


def _extract_activity_adaptive(
    isotope_series: dict[str, list[dict[str, float]]],
    target_unix: float,
    isotopes: Sequence[str],
    cal_window_rel_unc: Mapping[str, float] | None,
    assay_rel_unc: float | None,
    match_cfg: Mapping[str, Any] | None,
    downtime_periods: Sequence[tuple[float, float]],
) -> tuple[float, float, dict[str, Any]]:
    """Adaptively accumulate bins until the measured uncertainty is adequate."""

    cfg = match_cfg if isinstance(match_cfg, Mapping) else {}
    rel_unc_factor = float(cfg.get("adaptive_target_rel_unc_factor", 1.0))
    max_window_hours = float(
        cfg.get(
            "adaptive_max_window_hours",
            cfg.get("match_window_hours", 24.0),
        )
    )
    min_window_hours = float(cfg.get("adaptive_min_window_hours", 0.0))
    min_bins = max(1, int(cfg.get("adaptive_min_bins", 1)))
    default_target_rel_unc = float(cfg.get("adaptive_default_target_rel_unc", 0.25))

    if not np.isfinite(max_window_hours) or max_window_hours <= 0.0:
        max_window_hours = 24.0
    if not np.isfinite(min_window_hours) or min_window_hours < 0.0:
        min_window_hours = 0.0

    assay_rel_unc_val = float(assay_rel_unc) if assay_rel_unc is not None else float("nan")
    if np.isfinite(assay_rel_unc_val) and assay_rel_unc_val > 0.0:
        target_rel_unc = assay_rel_unc_val * rel_unc_factor
    else:
        target_rel_unc = default_target_rel_unc
    if not np.isfinite(target_rel_unc) or target_rel_unc <= 0.0:
        target_rel_unc = default_target_rel_unc

    merged_bins = _combine_isotope_activity_bins(
        isotope_series,
        isotopes,
        cal_window_rel_unc,
    )
    if not merged_bins:
        return math.nan, math.nan, {
            "status": "no_series",
            "mode": "adaptive",
            "target_unix": float(target_unix),
        }

    if _target_inside_period(target_unix, downtime_periods):
        return math.nan, math.nan, {
            "status": "target_in_downtime",
            "mode": "adaptive",
            "target_unix": float(target_unix),
            "target_iso": _epoch_to_iso_utc(target_unix),
        }

    segments = _build_activity_segments(merged_bins, downtime_periods)
    if not segments:
        return math.nan, math.nan, {
            "status": "no_usable_bins",
            "mode": "adaptive",
            "target_unix": float(target_unix),
        }

    segment: list[dict[str, float]] | None = None
    for candidate in segments:
        seg_start = float(candidate[0]["start"])
        seg_end = float(candidate[-1]["end"])
        if seg_start <= target_unix <= seg_end:
            segment = candidate
            break
    if segment is None:
        return math.nan, math.nan, {
            "status": "target_outside_coverage",
            "mode": "adaptive",
            "target_unix": float(target_unix),
            "target_iso": _epoch_to_iso_utc(target_unix),
            "coverage_start_unix": float(segments[0][0]["start"]),
            "coverage_end_unix": float(segments[-1][-1]["end"]),
        }

    max_window_s = max_window_hours * 3600.0
    min_window_s = min_window_hours * 3600.0
    candidates = [
        entry for entry in segment
        if abs(float(entry["t"]) - target_unix) <= max_window_s
    ]
    if not candidates:
        return math.nan, math.nan, {
            "status": "no_bins_within_max_window",
            "mode": "adaptive",
            "target_unix": float(target_unix),
            "target_iso": _epoch_to_iso_utc(target_unix),
            "max_window_hours": float(max_window_hours),
        }

    ordered = sorted(
        candidates,
        key=lambda entry: (
            abs(float(entry["t"]) - target_unix),
            abs(float(entry["start"]) - target_unix),
            float(entry["t"]),
        ),
    )

    used: list[dict[str, float]] = []
    total_counts = 0.0
    total_var = 0.0
    total_dt = 0.0
    best_details: dict[str, Any] | None = None

    for entry in ordered:
        used.append(entry)
        total_counts += float(entry["counts_total"])
        total_var += float(entry["var_counts_total"])
        total_dt += float(entry["dt"])
        if total_dt <= 0.0:
            continue

        rate = total_counts / total_dt
        rate_unc = math.sqrt(max(total_var, 0.0)) / total_dt
        rel_unc = (
            abs(rate_unc / rate)
            if np.isfinite(rate) and rate != 0.0 and np.isfinite(rate_unc)
            else float("inf")
        )
        window_start = min(float(item["start"]) for item in used)
        window_end = max(float(item["end"]) for item in used)
        window_hours = (window_end - window_start) / 3600.0
        best_details = {
            "status": "insufficient_precision",
            "mode": "adaptive",
            "target_unix": float(target_unix),
            "target_iso": _epoch_to_iso_utc(target_unix),
            "window_start_unix": float(window_start),
            "window_end_unix": float(window_end),
            "window_hours": float(window_hours),
            "n_bins": len(used),
            "rate_bq": float(rate),
            "rate_unc_bq": float(rate_unc),
            "achieved_rel_unc": float(rel_unc),
            "target_rel_unc": float(target_rel_unc),
            "max_window_hours": float(max_window_hours),
        }
        if len(used) < min_bins or window_hours < min_window_hours:
            continue
        if np.isfinite(rel_unc) and rel_unc <= target_rel_unc:
            best_details["status"] = "ok"
            return rate, rate_unc, best_details

    if best_details is None:
        return math.nan, math.nan, {
            "status": "no_accumulation",
            "mode": "adaptive",
            "target_unix": float(target_unix),
            "target_iso": _epoch_to_iso_utc(target_unix),
        }
    return math.nan, math.nan, best_details


def _extract_activity_at_time(
    isotope_series: dict[str, list[dict[str, float]]],
    target_unix: float,
    window_days: float = 1.0,
    isotopes: Sequence[str] = ("Po214", "Po218"),
    cal_window_rel_unc: dict[str, float] | None = None,
    *,
    assay_rel_unc: float | None = None,
    match_cfg: Mapping[str, Any] | None = None,
    downtime_periods: Sequence[Mapping[str, Any]] | Sequence[tuple[float, float]] | None = None,
    return_details: bool = False,
) -> tuple[float, float] | tuple[float, float, dict[str, Any]]:
    """Extract time-matched measured activity from per-bin time slices.

    In ``fixed`` mode, combine every bin whose center lies inside the
    requested half-window. In ``adaptive`` mode, grow outward from the assay
    time until the measured relative uncertainty is comparable to the assay
    uncertainty, while respecting downtime gaps and time-coverage limits.
    """
    if not isotope_series:
        details = {"status": "no_series", "mode": "adaptive"}
        return (math.nan, math.nan, details) if return_details else (math.nan, math.nan)

    cfg = match_cfg if isinstance(match_cfg, Mapping) else {}
    mode = str(cfg.get("time_match_mode", "adaptive")).strip().lower()
    downtime = _normalise_downtime_periods(downtime_periods)
    if mode == "fixed":
        rate, rate_unc, details = _extract_activity_fixed_window(
            isotope_series,
            target_unix,
            window_days,
            isotopes,
            cal_window_rel_unc,
        )
    else:
        rate, rate_unc, details = _extract_activity_adaptive(
            isotope_series,
            target_unix,
            isotopes,
            cal_window_rel_unc,
            assay_rel_unc,
            cfg,
            downtime,
        )

    return (rate, rate_unc, details) if return_details else (rate, rate_unc)


def _compute_emanation_baseline(
    isotope_series: dict[str, list[dict[str, float]]],
    cfg: dict[str, Any],
    isotopes: Sequence[str] = ("Po214", "Po218"),
) -> tuple[float, float] | None:
    """Compute the detector emanation baseline rate from a quiescent period.

    The radon monitor emanates radon from its own internal surfaces.
    During a period with no external radon source (between spike
    injections), the measured count rate equals this emanation rate.

    Parameters
    ----------
    isotope_series : dict
        Per-bin time series data.
    cfg : dict
        The ``lucas_bridge`` config section containing
        ``emanation_baseline.start`` and ``emanation_baseline.end``.
    isotopes : sequence of str
        Isotopes to sum.

    Returns
    -------
    (emanation_rate_bq, emanation_rate_unc_bq) or None
        The average count rate during the quiescent period, which
        represents the detector emanation contribution.
    """
    from datetime import datetime

    eman_cfg = cfg.get("emanation_baseline")
    if not isinstance(eman_cfg, Mapping):
        return None
    t_start_str = eman_cfg.get("start")
    t_end_str = eman_cfg.get("end")
    if not t_start_str or not t_end_str:
        return None

    try:
        t0 = datetime.fromisoformat(t_start_str).timestamp()
        t1 = datetime.fromisoformat(t_end_str).timestamp()
    except (ValueError, TypeError):
        logger.warning("Invalid emanation baseline dates: %s → %s",
                        t_start_str, t_end_str)
        return None

    # Sum counts across isotopes in the baseline window
    total_counts = 0.0
    total_dt = 0.0
    total_var_fitted = 0.0
    has_fitted_unc = False
    for iso in isotopes:
        for entry in isotope_series.get(iso, []):
            t = entry.get("t")
            c = entry.get("counts", 0.0)
            dt = entry.get("dt", 0.0)
            if t is None or not np.isfinite(t) or dt <= 0:
                continue
            if t0 <= t <= t1:
                if np.isfinite(c):
                    total_counts += c
                    total_dt += dt
                    _cu = entry.get("counts_unc")
                    if _cu is not None:
                        try:
                            _cu_f = float(_cu)
                            if math.isfinite(_cu_f) and _cu_f > 0:
                                total_var_fitted += _cu_f ** 2
                                has_fitted_unc = True
                        except (TypeError, ValueError):
                            pass

    if total_dt <= 0:
        logger.warning("No data in emanation baseline period [%s, %s]",
                        t_start_str, t_end_str)
        return None

    rate = total_counts / total_dt
    if has_fitted_unc and total_var_fitted > 0:
        rate_unc = math.sqrt(total_var_fitted) / total_dt
    else:
        rate_unc = math.sqrt(max(total_counts, 1.0)) / total_dt

    n_days = (t1 - t0) / 86400.0
    logger.info(
        "Emanation baseline [%s → %s] (%.1f days): "
        "%.6f ± %.6f Bq (%.0f counts in %.0f s)",
        t_start_str, t_end_str, n_days,
        rate, rate_unc, total_counts, total_dt,
    )
    return (rate, rate_unc)


def compute_bridge(
    assay_files: list[str],
    summary: Mapping[str, Any],
    cfg: dict[str, Any],
    isotope_series: dict[str, list[dict[str, float]]] | None = None,
    cal_window_rel_unc: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Run the Lucas-cell assay bridge computation.

    Parameters
    ----------
    assay_files : list[str]
        Paths to assay CSV/XLS files.
    summary : Mapping
        The RMTest summary dict.
    cfg : dict
        The full pipeline config (the ``lucas_bridge`` section is
        extracted internally).
    isotope_series : dict, optional
        Per-bin time series data from the pipeline
        (``{"Po214": [{"t": ..., "counts": ..., "dt": ...}, ...], ...}``).
        When provided, each assay is compared against the measured
        activity in the time window around the assay date rather than
        the dataset-averaged activity.
    cal_window_rel_unc : dict, optional
        Per-isotope fractional calibration window efficiency
        uncertainty, passed through to measured-activity extraction.

    Returns
    -------
    dict
        Bridge results for inclusion in ``summary.json``.
    """
    bridge_cfg = cfg.get("lucas_bridge", {})
    if not isinstance(bridge_cfg, Mapping):
        bridge_cfg = {}

    recompute = bridge_cfg.get("recompute_from_raw", False)
    comparison_target = bridge_cfg.get("comparison_target", "baseline_corrected_combined")
    volume_convention = bridge_cfg.get("volume_convention", "monitor")
    selection = bridge_cfg.get("selection")

    # Parse all assay files
    all_assays: list[LucasCellAssay] = []
    parsed_files: list[str] = []
    for fpath in assay_files:
        try:
            assays = parse_assay_csv(fpath)
            all_assays.extend(assays)
            parsed_files.append(str(fpath))
        except Exception as exc:
            logger.warning("Failed to parse assay file %s: %s", fpath, exc)

    if not all_assays:
        logger.info("No assays parsed from %d files", len(assay_files))
        return {}

    # Filter
    filtered = filter_assays(all_assays, selection)
    if not filtered:
        logger.info("No assays matched selection criteria")
        return {}

    logger.info("Bridge: %d assays after filtering (from %d total)", len(filtered), len(all_assays))

    # Find skid background assay (if configured)
    skid_bg = _find_skid_background(all_assays, bridge_cfg)
    if skid_bg is not None:
        logger.info(
            "Skid background: %s (%s), conc=%.3f +/- %.3f atoms/L",
            skid_bg.label,
            skid_bg.assay_date.date() if skid_bg.assay_date else "?",
            skid_bg.atoms_per_liter if not math.isnan(skid_bg.atoms_per_liter) else 0,
            skid_bg.atoms_per_liter_unc if not math.isnan(skid_bg.atoms_per_liter_unc) else 0,
        )

    # Extract fallback (dataset-averaged) measured activity from summary
    meas_bq_avg, meas_unc_avg = extract_measured_activity(summary, bridge_cfg)
    use_time_match = isotope_series is not None and len(isotope_series) > 0
    match_mode = str(
        bridge_cfg.get(
            "time_match_mode",
            "adaptive" if use_time_match else "fixed",
        )
    ).strip().lower()
    fallback_to_dataset_average = bool(
        bridge_cfg.get(
            "fallback_to_dataset_average",
            False if match_mode == "adaptive" else True,
        )
    )
    downtime_periods = _normalise_downtime_periods(
        (summary.get("downtime") or {}).get("periods")
        if isinstance(summary, Mapping)
        else None
    )
    # Match window: prefer hours (tighter), fall back to days for compat
    if "match_window_hours" in bridge_cfg:
        match_window_days = float(bridge_cfg["match_window_hours"]) / 24.0
    else:
        match_window_days = float(bridge_cfg.get("match_window_days", 1.0))

    # Auto-widen match window for coarse time bins (e.g. daily bins).
    # If the bin width exceeds the match window, the bin center may land
    # outside ±match_window of the assay time and nothing ever matches.
    if use_time_match and isotope_series and match_mode == "fixed":
        _sample_dts: list[float] = []
        for _iso_entries in isotope_series.values():
            for _e in _iso_entries[:5]:   # sample first few entries
                _dt_val = _e.get("dt")
                if _dt_val is not None and np.isfinite(_dt_val) and _dt_val > 0:
                    _sample_dts.append(float(_dt_val))
                    break
        if _sample_dts:
            _median_dt_s = float(np.median(_sample_dts))
            # Need half-window ≥ bin_width / 2 + small margin (1 hour)
            _min_window_days = (_median_dt_s / 2.0 + 3600.0) / 86400.0
            if match_window_days < _min_window_days:
                logger.info(
                    "Auto-widening match window from ±%.1f h to ±%.1f h "
                    "(bin_width=%.0f s)",
                    match_window_days * 24, _min_window_days * 24, _median_dt_s,
                )
                match_window_days = _min_window_days

    # Resolve volumes
    # The reference activity must be scaled to the volume the radon occupies
    # The bridge factor compares reference vs measured activity.  The
    # monitor's measured activity reflects radon in its own sensitive
    # volume (605 L), so the reference must also use the monitor volume.
    # When the UI is connected, the gas is well-mixed across the full
    # system, so atoms_per_liter is the same everywhere — but the
    # monitor only "sees" its own 605 L.  Using the total system volume
    # would inflate the bridge factor by the volume ratio.
    monitor_volume_l = float(bridge_cfg.get("monitor_volume_l",
                                            bridge_cfg.get("volume_l", 605.0)))
    ui_volume_l = float(bridge_cfg.get("ui_volume_l", 4196.0))

    # Compute detector emanation baseline (radon from detector surfaces)
    # This must be subtracted from measured activity before computing BF.
    eman_baseline = None
    if use_time_match and isotope_series:
        eman_baseline = _compute_emanation_baseline(isotope_series, bridge_cfg)
    eman_cfg = bridge_cfg.get("emanation_baseline", {})
    dilute_for_ui = eman_cfg.get("dilute_for_ui", True) if isinstance(eman_cfg, Mapping) else True
    total_system_volume_l = monitor_volume_l + ui_volume_l

    # Compute per-assay bridge factors
    assay_results: list[dict[str, Any]] = []
    valid_bfs: list[tuple[float, float, float, float]] = []

    for assay in filtered:
        # Per-assay overrides from config
        _assay_date_key = assay.assay_date.strftime("%Y-%m-%d") if assay.assay_date else ""
        _override = bridge_cfg.get("per_assay_overrides", {}).get(_assay_date_key, {})

        # Bridge always uses monitor volume (measured activity is per monitor)
        volume_l = monitor_volume_l

        # Compute reference activity — with or without skid background subtraction
        skid_subtracted = False

        # Detect UI-connected assays: the recompute formula (dual_path=2,
        # sample_vol=30 L) only applies to standalone RM assays.  For
        # UI-connected assays, the sampling geometry is different and the
        # spreadsheet formula chain handles this correctly.
        lbl_lower = assay.label.lower()
        is_ui_connected = "ui connected" in lbl_lower or "ui through" in lbl_lower

        # Gross atoms in monitor volume
        if is_ui_connected and not math.isnan(assay.atoms_per_liter):
            # UI-connected: use spreadsheet atoms_per_liter directly
            # (the recompute formula uses standalone sampling constants
            # that don't apply to UI-connected assays)
            gross_atoms_monitor = assay.atoms_per_liter * volume_l
            gross_atoms_monitor_unc = (
                assay.atoms_per_liter_unc * volume_l
                if not math.isnan(assay.atoms_per_liter_unc) else math.nan
            )
            logger.info(
                "UI-connected %s: using spreadsheet atoms/L=%.3f → "
                "atoms_monitor=%.1f (605 L)",
                assay.label[:40], assay.atoms_per_liter, gross_atoms_monitor,
            )
        elif recompute:
            _gross_result = _recompute_from_raw(assay, volume_l)
            gross_atoms_monitor = _gross_result[0]
            gross_atoms_monitor_unc = _gross_result[1]
            # Use spreadsheet relative uncertainty as floor
            if (not math.isnan(gross_atoms_monitor) and gross_atoms_monitor > 0
                    and not math.isnan(assay.atoms_per_liter_unc)
                    and not math.isnan(assay.atoms_per_liter)
                    and assay.atoms_per_liter > 0):
                ss_rel = assay.atoms_per_liter_unc / assay.atoms_per_liter
                rc_rel = gross_atoms_monitor_unc / gross_atoms_monitor
                if ss_rel > rc_rel:
                    gross_atoms_monitor_unc = gross_atoms_monitor * ss_rel
        elif not math.isnan(assay.atoms_per_liter):
            gross_atoms_monitor = assay.atoms_per_liter * volume_l
            gross_atoms_monitor_unc = (
                assay.atoms_per_liter_unc * volume_l
                if not math.isnan(assay.atoms_per_liter_unc) else math.nan
            )
        else:
            gross_atoms_monitor = math.nan
            gross_atoms_monitor_unc = math.nan

        # Apply per-assay background override (shallow copy to avoid
        # mutating the original assay object)
        if "lc_background_cpd" in _override:
            import copy as _copy_mod
            assay = _copy_mod.copy(assay)
            assay.lc_background_cpd = float(_override["lc_background_cpd"])
            logger.info(
                "Per-assay override %s: lc_background_cpd=%.1f",
                _assay_date_key, assay.lc_background_cpd,
            )

        if skid_bg is not None:
            net_atoms, net_atoms_unc, ref_bq, ref_unc = _subtract_skid_background(
                assay, skid_bg, volume_l, recompute=recompute,
            )
            if not math.isnan(ref_bq) and ref_bq > 0:
                skid_subtracted = True
            else:
                if not math.isnan(ref_bq) and ref_bq <= 0:
                    logger.warning(
                        "Skid background subtraction gave non-positive "
                        "activity (%.4f Bq) for %s — using gross value",
                        ref_bq, assay.label,
                    )
                # Fall back to no subtraction
                ref_bq, ref_unc = compute_reference_activity_bq(
                    assay, recompute=recompute, monitor_volume_l=volume_l,
                )
                net_atoms = math.nan
                net_atoms_unc = math.nan
        else:
            ref_bq, ref_unc = compute_reference_activity_bq(
                assay, recompute=recompute, monitor_volume_l=volume_l,
            )
            net_atoms = math.nan
            net_atoms_unc = math.nan

        # Time-matched or fallback measured activity.
        time_matched = False
        target_unix = math.nan
        match_details: dict[str, Any] = {}
        _sens_results: list[tuple[float, float, float]] = []
        if use_time_match and assay.assay_date is not None:
            dt_assay = assay.assay_date
            if dt_assay.hour == 0 and dt_assay.minute == 0:
                _def_time = bridge_cfg.get("default_match_time", "18:00")
                _def_hh, _def_mm = map(int, str(_def_time).split(":"))
                dt_assay = dt_assay.replace(hour=_def_hh, minute=_def_mm)
            if "match_time" in _override:
                _hh, _mm = map(int, _override["match_time"].split(":"))
                dt_assay = dt_assay.replace(hour=_hh, minute=_mm)
                logger.info(
                    "Per-assay override %s: match_time=%s",
                    _assay_date_key, _override["match_time"],
                )
            target_unix = dt_assay.timestamp()

            if "match_window_hours" in _override:
                _per_assay_mwd = float(_override["match_window_hours"]) / 24.0
            else:
                _per_assay_mwd = match_window_days

            assay_rel_unc = (
                ref_unc / ref_bq
                if np.isfinite(ref_unc) and np.isfinite(ref_bq) and ref_bq > 0.0
                else math.nan
            )

            if match_mode == "fixed":
                _sensitivity_hours = [0.5, 1.0, 2.0, 6.0, 24.0]
                for _sh in _sensitivity_hours:
                    _sw_d = _sh / 24.0
                    _sm, _su = _extract_activity_at_time(
                        isotope_series,
                        target_unix,
                        _sw_d,
                        cal_window_rel_unc=cal_window_rel_unc,
                        match_cfg={"time_match_mode": "fixed"},
                    )
                    if not math.isnan(_sm):
                        _sens_results.append((_sh, _sm, _su))
                if _sens_results:
                    _sens_str = ", ".join(
                        f"{h:.1f}h: {m:.4f}±{u:.4f} Bq"
                        for h, m, u in _sens_results
                    )
                    logger.info(
                        "Time-match sensitivity %s (%s): %s",
                        assay.label[:40], dt_assay.strftime("%Y-%m-%d %H:%M"),
                        _sens_str,
                    )

            meas_bq, meas_unc, match_details = _extract_activity_at_time(
                isotope_series,
                target_unix,
                _per_assay_mwd,
                cal_window_rel_unc=cal_window_rel_unc,
                assay_rel_unc=assay_rel_unc,
                match_cfg=bridge_cfg,
                downtime_periods=downtime_periods,
                return_details=True,
            )
            time_matched = not math.isnan(meas_bq)
            if not time_matched:
                status = match_details.get("status", "no_match")
                if fallback_to_dataset_average:
                    logger.warning(
                        "No usable time match for %s (%s; status=%s) — "
                        "falling back to dataset average",
                        assay.label[:40],
                        dt_assay.isoformat(),
                        status,
                    )
                    meas_bq, meas_unc = meas_bq_avg, meas_unc_avg
                    match_details["fallback_used"] = True
                    match_details["fallback_reason"] = status
                    if math.isnan(meas_bq):
                        logger.warning(
                            "Dataset average also unavailable for %s — SKIPPING",
                            assay.label[:40],
                        )
                else:
                    logger.warning(
                        "No usable time match for %s (%s; status=%s) — not using dataset average",
                        assay.label[:40],
                        dt_assay.isoformat(),
                        status,
                    )
        else:
            meas_bq, meas_unc = meas_bq_avg, meas_unc_avg
            match_details = {
                "status": "dataset_average",
                "mode": "none",
            }
            if math.isnan(meas_bq):
                logger.warning(
                    "No time-series and no dataset average for %s — SKIPPING",
                    assay.label[:40],
                )
            else:
                logger.info(
                    "Using dataset-averaged activity for %s: %.4f ± %.4f Bq",
                    assay.label[:40], meas_bq, meas_unc,
                )

        # ── Detector emanation baseline (for diagnostics only) ──
        # The monitor detects radon from its own surfaces.  For efficiency
        # measurements this is real signal (the detector IS detecting it),
        # so we do NOT subtract it from measured activity for the bridge.
        # We record it for informational / concentration-correction use.
        eman_sub = 0.0
        if eman_baseline is not None and not math.isnan(meas_bq):
            eman_rate, eman_unc = eman_baseline
            lbl = assay.label.lower()
            is_ui_connected = "ui connected" in lbl or "ui through" in lbl
            if is_ui_connected and dilute_for_ui:
                dilution = monitor_volume_l / total_system_volume_l
                eman_sub = eman_rate * dilution
            else:
                eman_sub = eman_rate
            logger.info(
                "Emanation baseline for %s: %.6f Bq%s (NOT subtracted for efficiency)",
                assay.label[:40], eman_sub,
                " (diluted)" if is_ui_connected and dilute_for_ui else "",
            )

        if math.isnan(meas_bq) or math.isnan(meas_unc):
            entry = {
                "assay_label": assay.label,
                "assay_date": assay.assay_date.isoformat() if assay.assay_date else None,
                "source_file": assay.source_file,
                "column_index": assay.column_index,
                "volume_l": volume_l,
                "gross_atoms_in_monitor": _safe_float(gross_atoms_monitor),
                "gross_atoms_in_monitor_unc": _safe_float(gross_atoms_monitor_unc),
                "skid_background_subtracted": skid_subtracted,
                "net_atoms_in_monitor": _safe_float(net_atoms) if skid_subtracted else _safe_float(gross_atoms_monitor),
                "net_atoms_in_monitor_unc": _safe_float(net_atoms_unc) if skid_subtracted else _safe_float(gross_atoms_monitor_unc),
                "reference_activity_bq": _safe_float(ref_bq),
                "reference_activity_unc_bq": _safe_float(ref_unc),
                "measured_activity_bq": None,
                "measured_activity_unc_bq": None,
                "emanation_subtracted_bq": _safe_float(eman_sub),
                "bridge_factor": None,
                "bridge_factor_unc": None,
                "detection_efficiency": None,
                "detection_efficiency_unc": None,
                "time_matched": False,
                "time_match_mode": match_mode if use_time_match else "none",
                "time_match_details": match_details or None,
                "time_match_sensitivity": (
                    {f"{h:.1f}h": {"activity_bq": _safe_float(m), "unc_bq": _safe_float(u)}
                     for h, m, u in _sens_results}
                    if _sens_results else None
                ),
                "raw_fields": {
                    "n_counts": _safe_float(assay.n_counts),
                    "transfer_efficiency": _safe_float(assay.overall_efficiency),
                    "counting_time_days": _safe_float(assay.counting_time_days),
                    "delay_time_days": _safe_float(assay.delay_time_days),
                    "lc_background_cpd": _safe_float(assay.lc_background_cpd),
                    "eff_counting": _EFF_COUNTING,
                    "alpha_multiplicity": _ALPHA_MULTIPLICITY,
                    "lc_single_alpha_eff": _LC_SINGLE_ALPHA_EFF,
                },
                "is_background_assay": "background" in assay.label.lower(),
                "in_aggregate": False,
            }
            assay_results.append(entry)
            continue

        bf, bf_unc = _compute_bridge_factor(ref_bq, ref_unc, meas_bq, meas_unc)

        # Split reference uncertainty into CORRELATED and UNCORRELATED.
        # Correlated: LC single-alpha efficiency (5%) and transfer eff (10%)
        # are the same physical constants for every assay.  These shift all
        # results coherently and must NOT average down in a weighted mean.
        # Everything else (Poisson, background, volume/flow, decay timing)
        # is per-assay = uncorrelated.
        _corr_rel = math.sqrt(_LC_EFF_REL_UNC**2 + _TRANSFER_EFF_REL_UNC**2)
        ref_unc_corr = abs(ref_bq) * _corr_rel if ref_bq != 0 else 0.0
        # Uncorrelated = remainder: sqrt(total^2 - correlated^2)
        if not math.isnan(ref_unc) and ref_unc > ref_unc_corr:
            ref_unc_uncorr = math.sqrt(ref_unc**2 - ref_unc_corr**2)
        else:
            ref_unc_uncorr = ref_unc if not math.isnan(ref_unc) else 0.0
        _, bf_unc_uncorr = _compute_bridge_factor(ref_bq, ref_unc_uncorr, meas_bq, meas_unc)

        # Derived: detection efficiency = 1/BF with uncertainty
        eff_det = math.nan
        eff_det_unc = math.nan
        if not math.isnan(bf) and bf > 0 and not math.isnan(bf_unc):
            eff_det = 1.0 / bf
            eff_det_unc = bf_unc / bf**2  # δ(1/x) = δx / x²

        entry: dict[str, Any] = {
            "assay_label": assay.label,
            "assay_date": assay.assay_date.isoformat() if assay.assay_date else None,
            "source_file": assay.source_file,
            "column_index": assay.column_index,
            "volume_l": volume_l,
            "gross_atoms_in_monitor": _safe_float(gross_atoms_monitor),
            "gross_atoms_in_monitor_unc": _safe_float(gross_atoms_monitor_unc),
            "skid_background_subtracted": skid_subtracted,
            "net_atoms_in_monitor": _safe_float(net_atoms) if skid_subtracted else _safe_float(gross_atoms_monitor),
            "net_atoms_in_monitor_unc": _safe_float(net_atoms_unc) if skid_subtracted else _safe_float(gross_atoms_monitor_unc),
            "reference_activity_bq": _safe_float(ref_bq),
            "reference_activity_unc_bq": _safe_float(ref_unc),
            "measured_activity_bq": _safe_float(meas_bq),
            "measured_activity_unc_bq": _safe_float(meas_unc),
            "emanation_subtracted_bq": _safe_float(eman_sub),
            "bridge_factor": _safe_float(bf),
            "bridge_factor_unc": _safe_float(bf_unc),
            "detection_efficiency": _safe_float(eff_det),
            "detection_efficiency_unc": _safe_float(eff_det_unc),
            "time_matched": time_matched,
            "time_match_mode": match_mode if use_time_match else "none",
            "time_match_details": match_details or None,
            "time_match_sensitivity": (
                {f"{h:.1f}h": {"activity_bq": _safe_float(m), "unc_bq": _safe_float(u)}
                 for h, m, u in _sens_results}
                if _sens_results else None
            ),
            "raw_fields": {
                "n_counts": _safe_float(assay.n_counts),
                "transfer_efficiency": _safe_float(assay.overall_efficiency),
                "counting_time_days": _safe_float(assay.counting_time_days),
                "delay_time_days": _safe_float(assay.delay_time_days),
                "lc_background_cpd": _safe_float(assay.lc_background_cpd),
                "eff_counting": _EFF_COUNTING,
                "alpha_multiplicity": _ALPHA_MULTIPLICITY,
                "lc_single_alpha_eff": _LC_SINGLE_ALPHA_EFF,
            },
        }

        if recompute:
            entry["precomputed_atoms_in_assay"] = _safe_float(assay.total_atoms_in_assay)
            entry["precomputed_atoms_per_liter"] = _safe_float(assay.atoms_per_liter)
            entry["precomputed_atoms_in_monitor"] = _safe_float(assay.atoms_in_monitor)

        # Flag assays excluded from aggregate:
        #  - background assays (huge uncertainties)
        #  - dates listed in exclude_from_aggregate config
        is_background_assay = "background" in assay.label.lower()
        _excl_dates = bridge_cfg.get("exclude_from_aggregate", [])
        _assay_date_str = (assay.assay_date.strftime("%Y-%m-%d")
                           if assay.assay_date else "")
        is_excluded = is_background_assay or _assay_date_str in _excl_dates
        entry["is_background_assay"] = is_background_assay
        entry["in_aggregate"] = not is_excluded
        if is_excluded and not is_background_assay:
            logger.info("Excluding %s (%s) from aggregate per config",
                        assay.label[:40], _assay_date_str)

        assay_results.append(entry)

        if (not math.isnan(bf) and not math.isinf(bf) and not math.isnan(bf_unc)
                and not is_excluded):
            valid_bfs.append((bf, bf_unc, bf_unc_uncorr, ref_unc_corr / ref_bq if ref_bq > 0 else 0.0))

    # Aggregate — weight by uncorrelated uncertainty, then add correlated back
    aggregate = _compute_aggregate(valid_bfs)

    # Build provenance
    provenance: dict[str, Any] = {
        "assay_files": parsed_files,
        "selection_criteria": selection if selection else {},
        "comparison_target": comparison_target,
        "volume_convention": volume_convention,
        "monitor_volume_l": monitor_volume_l,
        "ui_volume_l": ui_volume_l,
        "recomputed": recompute,
        "time_matched": use_time_match,
        "time_match_mode": match_mode if use_time_match else None,
        "match_window_days": match_window_days if use_time_match else None,
        "fallback_to_dataset_average": fallback_to_dataset_average,
        "downtime_period_count": len(downtime_periods),
        "skid_background": {
            "label": skid_bg.label if skid_bg else None,
            "date": skid_bg.assay_date.isoformat() if skid_bg and skid_bg.assay_date else None,
            "atoms_per_liter": _safe_float(skid_bg.atoms_per_liter) if skid_bg else None,
            "atoms_per_liter_unc": _safe_float(skid_bg.atoms_per_liter_unc) if skid_bg else None,
        } if skid_bg else None,
        "emanation_baseline": {
            "rate_bq": _safe_float(eman_baseline[0]) if eman_baseline else None,
            "rate_unc_bq": _safe_float(eman_baseline[1]) if eman_baseline else None,
            "period_start": eman_cfg.get("start") if isinstance(eman_cfg, Mapping) else None,
            "period_end": eman_cfg.get("end") if isinstance(eman_cfg, Mapping) else None,
            "dilute_for_ui": dilute_for_ui,
            "total_system_volume_l": total_system_volume_l,
        } if eman_baseline else None,
    }
    # Add config hash if available
    config_sha = cfg.get("config_sha256")
    if config_sha:
        provenance["config_sha256"] = config_sha

    return {
        "enabled": True,
        "n_assays": len(assay_results),
        "comparison_target": comparison_target,
        "volume_convention": volume_convention,
        "recomputed": recompute,
        "assays": assay_results,
        "aggregate": aggregate,
        "provenance": provenance,
    }


def _compute_aggregate(
    bfs: list[tuple[float, float, float, float]],
) -> dict[str, Any]:
    """Inverse-variance weighted mean of bridge factors.

    Each entry is ``(bf, bf_unc_total, bf_unc_uncorr, ref_corr_rel_frac)``.

    Weighting uses the UNCORRELATED uncertainty so that correlated
    systematics (LC efficiency, transfer efficiency) do not get
    averaged down.  After computing the weighted mean the correlated
    systematic is added back in quadrature.
    """
    if not bfs:
        return {}

    vals = np.array([b[0] for b in bfs])
    uncs_total = np.array([b[1] for b in bfs])
    uncs_uncorr = np.array([b[2] for b in bfs])
    corr_rel_fracs = np.array([b[3] for b in bfs])

    result: dict[str, Any] = {
        "n_assays": len(bfs),
        "mean_bridge_factor": float(np.mean(vals)),
    }

    # Weight by uncorrelated uncertainty only
    good = uncs_uncorr > 0
    if good.sum() >= 1:
        w = 1.0 / uncs_uncorr[good] ** 2
        w_sum = w.sum()
        wm = (w * vals[good]).sum() / w_sum
        wm_unc_stat_internal = 1.0 / math.sqrt(w_sum)
        result["weighted_mean_bridge_factor"] = float(wm)

        # Chi-squared consistency (computed with uncorrelated uncertainties)
        if good.sum() > 1:
            chi2 = ((vals[good] - wm) ** 2 * w).sum()
            ndf = int(good.sum()) - 1
            result["chi2"] = float(chi2)
            result["ndf"] = ndf
            chi2_ndf = chi2 / ndf if ndf > 0 else 1.0
            result["chi2_ndf"] = float(chi2_ndf)

            # Inflate for excess scatter (PDG prescription)
            scale = math.sqrt(chi2_ndf) if chi2_ndf > 1.0 else 1.0
            wm_unc_stat = wm_unc_stat_internal * scale
        else:
            wm_unc_stat = wm_unc_stat_internal

        # Add correlated systematic back in quadrature.
        # The correlated fraction (σ_ref_corr / ref) is the same physical
        # constants (LC eff 5%, transfer eff 10%) = ~11.2% relative.
        # On the bridge factor BF = ref/meas, this becomes BF * corr_rel.
        # Use the mean correlated fraction across assays (they should be the
        # same, but average in case of small differences).
        mean_corr_rel = float(np.mean(corr_rel_fracs[good]))
        wm_unc_syst = abs(wm) * mean_corr_rel
        wm_unc = math.sqrt(wm_unc_stat**2 + wm_unc_syst**2)

        result["weighted_mean_unc"] = float(wm_unc)
        result["weighted_mean_unc_stat"] = float(wm_unc_stat)
        result["weighted_mean_unc_syst"] = float(wm_unc_syst)
        result["weighted_mean_unc_internal"] = float(wm_unc_stat_internal)

    return result


def get_bridge_detection_efficiency(
    bridge_results: dict[str, Any],
) -> dict[str, Any] | None:
    """Extract per-isotope detection efficiency from bridge aggregate.

    The bridge factor measures the overall system response, so the
    derived detection efficiency applies equally to all isotopes
    (Po214 and Po218 pass through the same detector volume).

    Returns dict with keys:
        ``"Po214"``, ``"Po218"`` — efficiency values (float)
        ``"rel_unc"`` — fractional (relative) uncertainty σ_ε/ε = σ_BF/BF
    or ``None`` if unavailable.
    """
    agg = bridge_results.get("aggregate", {})
    bf = agg.get("weighted_mean_bridge_factor")
    bf_unc = agg.get("weighted_mean_unc")

    if bf is None or bf <= 0:
        return None

    eff = 1.0 / bf
    eff_unc = bf_unc / bf**2 if bf_unc is not None else None
    # Relative uncertainty: σ_ε/ε = σ_BF/BF (same for both isotopes)
    rel_unc = bf_unc / bf if bf_unc is not None and bf_unc > 0 else 0.0

    logger.info(
        "Bridge-derived detection efficiency: %.4f +/- %.4f (rel %.1f%%) "
        "(BF=%.3f +/- %.3f)",
        eff, eff_unc or 0, rel_unc * 100, bf, bf_unc or 0,
    )
    return {"Po214": eff, "Po218": eff, "rel_unc": rel_unc}


# ---------------------------------------------------------------------------
# Spike-based independent efficiency measurement
# ---------------------------------------------------------------------------

def fit_spike_decay(
    isotope_series: dict[str, list[dict[str, float]]],
    t_start_iso: str,
    t_end_iso: str,
    *,
    half_life_days: float = 3.8235,
    isotopes: Sequence[str] = ("Po214", "Po218"),
    skip_initial_days: float = 0.5,
    fit_start_iso: str | None = None,
    time_zone: str = "UTC",
) -> dict[str, Any]:
    """Fit exponential decay to a radon spike period with fixed half-life.

    Model: R(t) = R0 * exp(-λ*(t - t0)) + B
    where λ = ln(2)/half_life (fixed, textbook Rn-222) and t0 = t_start.

    The fit extracts R0 (count rate at injection) and B (background)
    from the binned time-series data.

    Parameters
    ----------
    isotope_series : dict
        ``{"Po214": [{"t": unix_s, "counts": N, "dt": width_s}, ...], ...}``
    t_start_iso, t_end_iso : str
        ISO-format date/time strings defining the spike-fit interval.
    half_life_days : float
        Rn-222 half-life in days (fixed, textbook value).
    isotopes : sequence of str
        Isotopes to sum.
    skip_initial_days : float
        Skip this many days from the start of the spike period before
        fitting when ``fit_start_iso`` is not provided.
    fit_start_iso : str, optional
        Explicit fit-start timestamp. When provided, bins are included
        only when their full width lies inside the fit window.

    Returns
    -------
    dict with keys:
        t_start, t_end, half_life_days, lambda_per_s,
        R0, R0_unc, B, B_unc, chi2, ndf, chi2_ndf,
        n_bins, initial_activity_bq (= R0, the inferred count rate at t0)
    """
    try:
        from scipy.optimize import curve_fit
    except ImportError:
        logger.warning("scipy not available for spike fitting")
        return {"error": "scipy not available"}

    t0 = to_epoch_seconds(t_start_iso, tz=time_zone)
    t1 = to_epoch_seconds(t_end_iso, tz=time_zone)
    if fit_start_iso is not None:
        t_fit_start = to_epoch_seconds(fit_start_iso, tz=time_zone)
        skip_initial_days = max(0.0, (t_fit_start - t0) / 86400.0)
    else:
        t_fit_start = t0 + skip_initial_days * 86400.0
        fit_start_iso = _epoch_to_iso_utc(t_fit_start)
    lam = math.log(2) / (half_life_days * 86400.0)

    # Collect binned data: combine isotopes
    bin_data: list[tuple[float, float, float]] = []  # (t_mid, rate, rate_unc)
    # First, collect all bins across isotopes into a time-indexed dict
    time_bins: dict[float, list[tuple[float, float]]] = {}  # t -> [(counts, dt)]
    for iso in isotopes:
        for entry in isotope_series.get(iso, []):
            t = entry.get("t")
            c = entry.get("counts", 0.0)
            dt = entry.get("dt", 0.0)
            if t is None or not np.isfinite(t) or dt <= 0:
                continue
            bin_start = t - dt / 2.0
            bin_end = t + dt / 2.0
            if bin_start >= t_fit_start and bin_end <= t1:
                if t not in time_bins:
                    time_bins[t] = []
                time_bins[t].append((c, dt))

    # Sum counts per time bin, compute rate
    for t_mid in sorted(time_bins.keys()):
        entries = time_bins[t_mid]
        total_counts = sum(c for c, _ in entries)
        # dt should be the same for all isotopes in the same bin
        dt = entries[0][1]
        rate = total_counts / dt
        rate_unc = math.sqrt(max(total_counts, 1.0)) / dt
        bin_data.append((t_mid, rate, rate_unc))

    logger.info(
        "Spike fit %s → %s: %d bins (fit starts %s; skipped %.4f days)",
        t_start_iso, t_end_iso, len(bin_data), fit_start_iso, skip_initial_days,
    )
    if len(bin_data) < 3:
        logger.warning("Too few bins (%d) for spike fit in [%s, %s]",
                        len(bin_data), t_start_iso, t_end_iso)
        return {"error": f"too few bins ({len(bin_data)})"}

    t_arr = np.array([b[0] for b in bin_data])
    r_arr = np.array([b[1] for b in bin_data])
    u_arr = np.array([b[2] for b in bin_data])

    # Model: R(t) = R0 * exp(-lam*(t-t0)) + B
    def _decay_model(t, R0, B):
        return R0 * np.exp(-lam * (t - t0)) + B

    # Initial guesses
    r_max = float(np.max(r_arr))
    r_min = float(np.min(r_arr))
    p0 = [r_max - r_min, r_min]

    try:
        popt, pcov = curve_fit(
            _decay_model, t_arr, r_arr, p0=p0,
            sigma=u_arr, absolute_sigma=True,
            maxfev=10000,
        )
    except RuntimeError as e:
        logger.warning("Spike fit failed: %s", e)
        return {"error": str(e)}

    R0_fit, B_fit = popt
    R0_unc = math.sqrt(pcov[0, 0]) if pcov[0, 0] > 0 else 0.0
    B_unc = math.sqrt(pcov[1, 1]) if pcov[1, 1] > 0 else 0.0

    # Chi-squared
    residuals = r_arr - _decay_model(t_arr, *popt)
    chi2 = float(np.sum((residuals / u_arr) ** 2))
    ndf = len(bin_data) - 2
    chi2_ndf = chi2 / ndf if ndf > 0 else float("nan")

    result = {
        "t_start": t_start_iso,
        "t_end": t_end_iso,
        "fit_start": fit_start_iso,
        "skip_initial_days": skip_initial_days,
        "fit_delay_minutes": _safe_float(skip_initial_days * 1440.0),
        "half_life_days": half_life_days,
        "lambda_per_s": lam,
        "R0": _safe_float(R0_fit),
        "R0_unc": _safe_float(R0_unc),
        "B": _safe_float(B_fit),
        "B_unc": _safe_float(B_unc),
        "chi2": _safe_float(chi2),
        "ndf": ndf,
        "chi2_ndf": _safe_float(chi2_ndf),
        "n_bins": len(bin_data),
        "initial_count_rate_bq": _safe_float(R0_fit),
        "initial_count_rate_unc_bq": _safe_float(R0_unc),
    }

    logger.info(
        "Spike decay fit [%s → %s]: R0=%.5f±%.5f Bq, B=%.5f±%.5f Bq, "
        "χ²/ndf=%.2f/%d=%.2f, %d bins",
        t_start_iso, t_end_iso, R0_fit, R0_unc, B_fit, B_unc,
        chi2, ndf, chi2_ndf, len(bin_data),
    )
    return result


def fit_spike_periods(
    isotope_series: dict[str, list[dict[str, float]]],
    cfg: dict[str, Any],
    assay_results: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Fit radon spike injection periods and derive detection efficiency.

    For each spike period, fits R(t) = R0·exp(-λ·t) + B to extract R0
    (measured count rate at injection).  If an assay was performed on the
    spike injection date, the detection efficiency is:

        eff_spike = R0 / ref_activity_bq

    where ``ref_activity_bq`` is the assay's reference activity in the
    monitor volume.  This is an independent efficiency estimate from the
    exponential fit.

    Parameters
    ----------
    isotope_series : dict
        Per-isotope time series data.
    cfg : dict
        Full pipeline config (looks for lucas_bridge.spike_periods).
    assay_results : list of dict, optional
        Per-assay bridge results (from ``compute_bridge``), used to look
        up the reference activity for spike-derived efficiency.

    Returns
    -------
    dict with per-spike results including spike-derived efficiency.
    """
    bridge_cfg = cfg.get("lucas_bridge", {})
    spike_periods = bridge_cfg.get("spike_periods", [])
    default_skip_days = float(bridge_cfg.get("spike_skip_initial_days", 0.5))
    spike_tz = str(
        ((cfg.get("radon_inference") or {}).get("external_rn") or {}).get("tz", "UTC")
    )

    if not spike_periods:
        # Default periods from user specification
        spike_periods = [
            {
                "start": "2024-09-12T14:30:00-04:00",
                "end": "2024-09-28T00:00:00-04:00",
                "fit_delay_minutes": 30,
                "label": "Spike 1 (Sep 2024)",
                "assay_date": "2024-09-16",
            },
            {
                "start": "2024-11-01T19:00:00-04:00",
                "end": "2024-11-24T00:00:00-05:00",
                "fit_delay_minutes": 30,
                "label": "Spike 2 (Nov 2024)",
                "assay_date": "2024-11-05",
            },
        ]

    half_life = bridge_cfg.get("rn222_half_life_days", 3.8235)
    results: list[dict[str, Any]] = []

    for period in spike_periods:
        t_start = period.get("start")
        t_end = period.get("end")
        label = period.get("label", f"{t_start} → {t_end}")

        if not t_start or not t_end:
            logger.warning("Spike period missing start/end: %s", period)
            continue

        logger.info("Fitting spike period: %s [%s → %s]", label, t_start, t_end)
        fit_start_iso = period.get("fit_start")
        if fit_start_iso is not None:
            skip_initial_days = max(
                0.0,
                (
                    to_epoch_seconds(fit_start_iso, tz=spike_tz)
                    - to_epoch_seconds(t_start, tz=spike_tz)
                ) / 86400.0,
            )
        elif period.get("fit_delay_minutes") is not None:
            skip_initial_days = float(period["fit_delay_minutes"]) / 1440.0
            fit_start_iso = _epoch_to_iso_utc(
                to_epoch_seconds(t_start, tz=spike_tz) + skip_initial_days * 86400.0
            )
        elif period.get("skip_initial_days") is not None:
            skip_initial_days = float(period["skip_initial_days"])
            fit_start_iso = _epoch_to_iso_utc(
                to_epoch_seconds(t_start, tz=spike_tz) + skip_initial_days * 86400.0
            )
        else:
            skip_initial_days = default_skip_days
            fit_start_iso = _epoch_to_iso_utc(
                to_epoch_seconds(t_start, tz=spike_tz) + skip_initial_days * 86400.0
            )
        fit_result = fit_spike_decay(
            isotope_series, t_start, t_end,
            half_life_days=half_life,
            skip_initial_days=skip_initial_days,
            fit_start_iso=fit_start_iso,
            time_zone=spike_tz,
        )
        fit_result["label"] = label

        # ── Spike-derived efficiency ──
        # Match the spike to the assay done shortly AFTER injection.
        # The fit gives R(t) = R0·exp(-λ·(t-t0)) + B where R0 is the
        # excess above baseline at t0 (spike start).  The assay measures
        # the TRUE total radon at t_assay (days later, with skid bg
        # already subtracted on the assay side).
        #
        # To compare like-for-like we evaluate the spike's measured
        # contribution at the assay time:
        #   R_spike(t_assay) = R0 · exp(-λ · (t_assay - t0))
        # and compare to the assay reference:
        #   eff = R_spike(t_assay) / ref_bq
        R0 = fit_result.get("R0")
        R0_unc = fit_result.get("R0_unc", 0)
        lam_s = fit_result.get("lambda_per_s", 0)
        match_date = period.get("assay_date", t_start)
        if R0 is not None and R0 > 0 and lam_s > 0 and assay_results:
            for a in assay_results:
                a_date = (a.get("assay_date") or "")[:10]
                if a_date == match_date:
                    ref_bq = a.get("reference_activity_bq", 0)
                    ref_unc = a.get("reference_activity_unc_bq", 0)
                    if ref_bq and ref_bq > 0:
                        # Time offset: spike start → assay date
                        try:
                            delta_s = (
                                to_epoch_seconds(a_date, tz=spike_tz)
                                - to_epoch_seconds(t_start, tz=spike_tz)
                            )
                        except (ValueError, TypeError):
                            delta_s = 0.0

                        # Decay R0 to the assay time
                        decay_factor = math.exp(-lam_s * delta_s)
                        R_at_assay = R0 * decay_factor
                        R_at_assay_unc = R0_unc * decay_factor

                        spike_eff = R_at_assay / ref_bq
                        spike_eff_unc = spike_eff * math.sqrt(
                            (R_at_assay_unc / R_at_assay) ** 2 +
                            (ref_unc / ref_bq) ** 2
                        ) if R_at_assay_unc and ref_unc else 0.0

                        fit_result["spike_efficiency"] = _safe_float(spike_eff)
                        fit_result["spike_efficiency_unc"] = _safe_float(spike_eff_unc)
                        fit_result["matched_assay_date"] = a_date
                        fit_result["matched_assay_ref_bq"] = _safe_float(ref_bq)
                        fit_result["R_at_assay"] = _safe_float(R_at_assay)
                        fit_result["decay_factor"] = _safe_float(decay_factor)
                        fit_result["delta_days"] = _safe_float(delta_s / 86400.0)
                        logger.info(
                            "Spike efficiency [%s]: R0=%.5f × exp(-λ·%.1fd)="
                            "%.5f / ref=%.5f → eff=%.3f ± %.3f",
                            label, R0, delta_s / 86400.0, R_at_assay,
                            ref_bq, spike_eff, spike_eff_unc,
                        )
                    break

        results.append(fit_result)

    return {
        "n_periods": len(results),
        "half_life_days_fixed": half_life,
        "periods": results,
    }


def _safe_float(v: Any) -> Any:
    """Convert to float for JSON, returning None for NaN/Inf."""
    if v is None:
        return None
    try:
        f = float(v)
        if math.isnan(f) or math.isinf(f):
            return None
        return f
    except (TypeError, ValueError):
        return None
