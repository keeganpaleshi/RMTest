"""Plotting helpers for radon inference outputs."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable, Mapping

import matplotlib.pyplot as plt
from matplotlib.axes import Axes
import numpy as np

from plot_utils._time_utils import guard_mpl_times, setup_time_axis

logger = logging.getLogger(__name__)
_DEFAULT_MARKER_SIZE = 0.5
_DEFAULT_CAPSIZE = 0
_DEFAULT_ELINEWIDTH = 0.5
_DEFAULT_ALPHA = 0.4


def _extract_series(
    data: Iterable[Mapping[str, object]] | None,
    value_key: str,
    error_key: str | None = None,
) -> tuple[list[float], list[float], list[float] | None]:
    """Extract (times, values, errors) from a series of dicts.

    If *error_key* is given and the entries contain that key, the third
    element is a list of error values; otherwise it is ``None``.
    """
    times: list[float] = []
    values: list[float] = []
    errors: list[float] | None = [] if error_key else None
    if not data:
        return times, values, errors
    for entry in data:
        try:
            t = float(entry.get("t"))
            val = float(entry.get(value_key))
        except (AttributeError, TypeError, ValueError):
            continue
        if not np.isfinite(t) or not np.isfinite(val):
            continue
        times.append(t)
        values.append(val)
        if error_key is not None and errors is not None:
            try:
                err = float(entry.get(error_key, 0.0))
            except (TypeError, ValueError):
                err = 0.0
            errors.append(err if np.isfinite(err) else 0.0)
    # If all errors are zero, treat as "no errors available"
    if errors is not None and all(e == 0.0 for e in errors):
        errors = None
    return times, values, errors


def _format_meta(meta: Mapping[str, object] | None) -> str:
    if not isinstance(meta, Mapping):
        return ""

    parts: list[str] = []

    # Source isotopes
    isotopes = meta.get("source_isotopes")
    if isotopes:
        iso_list = ", ".join(str(iso) for iso in isotopes)
        parts.append(f"Source isotopes: {iso_list}")

    # Detection efficiencies
    detection = meta.get("detection_efficiency") or {}
    if isinstance(detection, Mapping):
        det_text = ", ".join(f"{iso}: {val:.3g}" for iso, val in detection.items())
        if det_text:
            parts.append(f"Detection eff: {det_text}")

    # Transport and retention
    transport = meta.get("transport_efficiency")
    retention = meta.get("retention_efficiency")
    if transport is not None:
        parts.append(f"Transport eff: {transport}")
    if retention is not None:
        parts.append(f"Retention eff: {retention}")

    # Chain correction
    chain = meta.get("chain_correction")
    if chain:
        parts.append(f"Chain correction: {chain}")

    return "\n".join(parts)


def _legend_label(meta: Mapping[str, object] | None) -> str:
    if not isinstance(meta, Mapping):
        return "Inferred Rn222"
    isotopes = meta.get("source_isotopes") or []
    weights = meta.get("source_weights") or {}
    if not isotopes:
        return "Inferred Rn222"

    labels = []
    for iso in isotopes:
        if isinstance(weights, Mapping) and iso in weights:
            labels.append(f"{iso} (w={weights[iso]:.2f})")
        else:
            labels.append(str(iso))
    return "Weighted by " + ", ".join(labels)


def plot_rn_inferred_vs_time(radon_results: Mapping[str, object], out_dir: Path) -> None:
    if not isinstance(radon_results, Mapping):
        logger.info("radon_inference missing, skipping plot.")
        return

    series = radon_results.get("rn_inferred")
    if not series:
        logger.info("radon_inference missing, skipping plot.")
        return

    times, values, errors = _extract_series(series, "rn_bq", "rn_bq_err")
    if not times or not values:
        logger.info("No inferred radon data available for plotting")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    meta = radon_results.get("meta")
    det_eff = meta.get("detection_efficiency", {}) if isinstance(meta, Mapping) else {}
    source_isos = meta.get("source_isotopes", []) if isinstance(meta, Mapping) else []
    weights = meta.get("source_weights", {}) if isinstance(meta, Mapping) else {}
    overall_eff = 1.0
    if isinstance(meta, Mapping):
        overall_eff = float(meta.get("transport_efficiency", 1.0) or 1.0) * \
                      float(meta.get("retention_efficiency", 1.0) or 1.0)

    # --- Rn-222 inferred activity plot (combined, no per-isotope breakdown) ---
    fig, ax = plt.subplots(figsize=(10, 5))
    times_mpl = guard_mpl_times(times=times)

    label = "Rn-222"
    if errors is not None:
        ax.errorbar(times_mpl, values, yerr=errors, fmt="o",
                    markersize=_DEFAULT_MARKER_SIZE, label=label,
                    capsize=_DEFAULT_CAPSIZE, elinewidth=_DEFAULT_ELINEWIDTH,
                    alpha=_DEFAULT_ALPHA, color="#1f77b4", zorder=5)
    else:
        ax.plot(times_mpl, values, marker="o", linestyle="None",
                markersize=_DEFAULT_MARKER_SIZE, label=label,
                alpha=_DEFAULT_ALPHA, color="#1f77b4", zorder=5)
    ax.set_ylabel("Inferred Rn-222 Activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("Inferred Radon Activity", fontsize=11)
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)

    fig.tight_layout()
    fig.savefig(out_dir / "radon_inferred.png", dpi=300)
    plt.close(fig)


def plot_ambient_rn_vs_time(
    radon_results: Mapping[str, object],
    out_dir: Path,
    fallback_bq_per_m3: float | None = None,
) -> None:
    """Plot external ambient radon time series.

    Parameters
    ----------
    radon_results : dict
        Radon inference results containing ``ambient_rn`` series.
    out_dir : Path
        Output directory for the plot.
    fallback_bq_per_m3 : float or None
        If provided, draw a horizontal reference line at this constant
        value (Bq/m³) for comparison with the file-based ambient data.
    """
    if not isinstance(radon_results, Mapping):
        logger.info("No radon results available for ambient radon plotting")
        return

    ambient_series = radon_results.get("ambient_rn")
    times, values, _ = _extract_series(ambient_series, "rn_bq_per_m3")
    if not times or not values:
        logger.info("No ambient radon data available for plotting")
        return

    # Check if we need to trim to match rn_inferred length
    rn_inferred_series = radon_results.get("rn_inferred")
    rn_times, _, _ = _extract_series(rn_inferred_series, "rn_bq")
    if rn_times and len(times) != len(rn_times):
        min_len = min(len(times), len(rn_times))
        logger.info(
            "Trimming ambient_rn from %d to %d points to match rn_inferred length",
            len(times),
            min_len,
        )
        times = times[:min_len]
        values = values[:min_len]

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Check if values are constant
    is_constant = False
    if len(set(values)) == 1:
        is_constant = True
        logger.info("Ambient radon is constant at %.3g Bq/m^3", values[0])

    fig, ax = plt.subplots(figsize=(8, 5))
    times_mpl = guard_mpl_times(times=times)
    ax.plot(times_mpl, values, marker="o", linestyle="None", markersize=_DEFAULT_MARKER_SIZE, color="#1f77b4", alpha=_DEFAULT_ALPHA, label="Pico40L ambient")
    ax.set_ylabel("Ambient Radon [Bq/m$^3$]")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("External Ambient Radon", fontsize=11)
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)

    # Optional constant reference line for comparison
    if fallback_bq_per_m3 is not None and fallback_bq_per_m3 > 0:
        ax.axhline(
            fallback_bq_per_m3,
            color="red", lw=1.0, ls="--", alpha=0.7,
            label=f"Constant fallback ({fallback_bq_per_m3:.0f} Bq/m³)",
        )
        ax.legend(fontsize=8, loc="upper right")

    # Annotate if constant
    if is_constant:
        ax.text(
            0.98,
            0.98,
            f"Constant: {values[0]:.3g} Bq/m^3",
            transform=ax.transAxes,
            ha="right",
            va="top",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.8),
            fontsize="small",
        )

    fig.tight_layout()
    fig.savefig(out_dir / "ambient_radon.png", dpi=300)
    plt.close(fig)


def plot_volume_equiv_vs_time(
    radon_results: Mapping[str, object],
    out_dir: Path,
    volume_units: str | None = None,
    *,
    suffix: str = "",
) -> None:
    if not isinstance(radon_results, Mapping):
        logger.info("No radon results available for volume plotting")
        return

    volume_series = radon_results.get("volume_equiv")
    times, volumes_m3, vol_errs = _extract_series(volume_series, "v_m3", "v_m3_err")
    _, volumes_lpm, lpm_errs = _extract_series(volume_series, "v_lpm", "v_lpm_err")
    if not times or not volumes_m3:
        logger.info("No leak-volume data available for plotting")
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Determine units from config or use default
    if volume_units is None:
        meta = radon_results.get("meta")
        if isinstance(meta, Mapping):
            volume_units = meta.get("volume_units", "m^3 leaked per interval")
        else:
            volume_units = "m^3 leaked per interval"
    volume_units = str(volume_units)
    cumulative_units = volume_units.replace(" per interval", "")

    times_mpl = guard_mpl_times(times=times)

    _eb = dict(fmt="o", markersize=_DEFAULT_MARKER_SIZE, capsize=_DEFAULT_CAPSIZE,
               elinewidth=_DEFAULT_ELINEWIDTH, alpha=_DEFAULT_ALPHA)
    _mk = dict(marker="o", linestyle="None", markersize=_DEFAULT_MARKER_SIZE,
               alpha=_DEFAULT_ALPHA)

    def _fname(base: str) -> str:
        """Insert suffix before the .png extension."""
        if suffix:
            return base.replace(".png", f"{suffix}.png")
        return base

    # --- per-interval volume (m^3) ---
    fig, ax1 = plt.subplots(figsize=(10, 5))
    if vol_errs is not None:
        ax1.errorbar(times_mpl, volumes_m3, yerr=vol_errs, color="#2ca02c", **_eb)
    else:
        ax1.plot(times_mpl, volumes_m3, color="#2ca02c", **_mk)
    ax1.set_ylabel(f"Leak Volume [{volume_units}]")
    ax1.set_xlabel("Time (UTC)")
    ax1.set_title("Per-Interval Leaked Volume", fontsize=11)
    ax1.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax1, times_mpl)
    ax1.yaxis.get_offset_text().set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / _fname("equivalent_volume.png"), dpi=300)
    plt.close(fig)

    # --- per-interval volume (litres) ---
    volumes_liters = [v * 1000.0 for v in volumes_m3]
    vol_errs_liters = [e * 1000.0 for e in vol_errs] if vol_errs is not None else None
    fig_l, ax_l = plt.subplots(figsize=(10, 5))
    if vol_errs_liters is not None:
        ax_l.errorbar(times_mpl, volumes_liters, yerr=vol_errs_liters, color="#17becf", **_eb)
    else:
        ax_l.plot(times_mpl, volumes_liters, color="#17becf", **_mk)
    ax_l.set_ylabel("Leak Volume [L per interval]")
    ax_l.set_xlabel("Time (UTC)")
    ax_l.set_title("Per-Interval Leaked Volume (litres)", fontsize=11)
    ax_l.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax_l, times_mpl)
    ax_l.yaxis.get_offset_text().set_visible(False)
    fig_l.autofmt_xdate()
    fig_l.tight_layout()
    fig_l.savefig(out_dir / _fname("equivalent_volume_liters.png"), dpi=300)
    plt.close(fig_l)

    # --- flow rate (mL/min) — linear and log ---
    if volumes_lpm:
        paired_len = min(len(times_mpl), len(volumes_lpm))
        if paired_len == 0:
            logger.info("Leak-rate data present but no overlapping timestamps; skipping flow plot")
        else:
            times_flow = times_mpl[:paired_len]
            # Convert L/min to mL/min
            volumes_mlpm = [v * 1000.0 for v in volumes_lpm[:paired_len]]
            flow_errs_ml = [e * 1000.0 for e in lpm_errs[:paired_len]] if lpm_errs is not None else None

            # Linear scale
            fig_flow, ax_flow = plt.subplots(figsize=(10, 5))
            if flow_errs_ml is not None:
                ax_flow.errorbar(times_flow, volumes_mlpm, yerr=flow_errs_ml, color="#ff7f0e", **_eb)
            else:
                ax_flow.plot(times_flow, volumes_mlpm, color="#ff7f0e", **_mk)
            ax_flow.set_ylabel("Leak Rate [mL/min]")
            ax_flow.set_xlabel("Time (UTC)")
            ax_flow.set_title("Equivalent Leak Rate", fontsize=11)
            ax_flow.ticklabel_format(axis="y", style="plain")
            setup_time_axis(ax_flow, times_flow)
            ax_flow.yaxis.get_offset_text().set_visible(False)
            fig_flow.autofmt_xdate()
            fig_flow.tight_layout()
            fig_flow.savefig(out_dir / _fname("equivalent_flow.png"), dpi=300)
            plt.close(fig_flow)

            # Log scale
            fig_flow_log, ax_flow_log = plt.subplots(figsize=(10, 5))
            if flow_errs_ml is not None:
                ax_flow_log.errorbar(times_flow, volumes_mlpm, yerr=flow_errs_ml, color="#ff7f0e", **_eb)
            else:
                ax_flow_log.plot(times_flow, volumes_mlpm, color="#ff7f0e", **_mk)
            ax_flow_log.set_yscale("log")
            ax_flow_log.set_ylabel("Leak Rate [mL/min]")
            ax_flow_log.set_xlabel("Time (UTC)")
            ax_flow_log.set_title("Equivalent Leak Rate (log)", fontsize=11)
            setup_time_axis(ax_flow_log, times_flow)
            fig_flow_log.autofmt_xdate()
            fig_flow_log.tight_layout()
            fig_flow_log.savefig(out_dir / _fname("equivalent_flow_log.png"), dpi=300)
            plt.close(fig_flow_log)

    # --- cumulative volume (m^3 and litres) ---
    cumulative_series = radon_results.get("volume_cumulative")
    if cumulative_series:
        cum_times, cum_values, cum_errs = _extract_series(
            cumulative_series, "v_m3_cum", "v_m3_cum_err"
        )
        if cum_times and cum_values:
            times_mpl_cum = guard_mpl_times(times=cum_times)
            for unit_label, vals, errs, color, fname in [
                (f"Cumulative Leaked Volume [{cumulative_units}]",
                 cum_values, cum_errs, "#9467bd",
                 "equivalent_volume_cumulative.png"),
                ("Cumulative Leaked Volume [L]",
                 [v * 1000.0 for v in cum_values],
                 [e * 1000.0 for e in cum_errs] if cum_errs else None,
                 "#8c564b",
                 "equivalent_volume_cumulative_liters.png"),
            ]:
                fig_c, ax_c = plt.subplots(figsize=(10, 5))
                if errs is not None:
                    ax_c.errorbar(times_mpl_cum, vals, yerr=errs, color=color, **_eb)
                else:
                    ax_c.plot(times_mpl_cum, vals, color=color, **_mk)
                ax_c.set_ylabel(unit_label)
                ax_c.set_xlabel("Time (UTC)")
                ax_c.set_title("Cumulative Leaked Volume", fontsize=11)
                ax_c.ticklabel_format(axis="y", style="plain")
                setup_time_axis(ax_c, times_mpl_cum)
                ax_c.yaxis.get_offset_text().set_visible(False)
                fig_c.autofmt_xdate()
                fig_c.tight_layout()
                fig_c.savefig(out_dir / _fname(fname), dpi=300)
                plt.close(fig_c)
        else:
            logger.info("No valid cumulative volume data for plotting")
    else:
        logger.info("No cumulative volume series available, skipping cumulative plot")


def plot_radon_in_liquid(
    radon_results: Mapping[str, object],
    out_dir: Path,
    cfg: Mapping[str, object] | None = None,
) -> None:
    """Plot radon concentration implied in the liquid phase via Henry's law.

    Uses the inferred Rn-222 activity in the detector gas volume and the
    Henry's law partition coefficient to estimate what the radon
    concentration in the liquid must be:

        C_liquid = activity_Bq / (sample_volume_L * K_H)

    where K_H is the dimensionless air/liquid partition coefficient
    (C_air / C_liquid).
    """
    if not isinstance(radon_results, Mapping):
        return

    rn_series = radon_results.get("rn_inferred")
    if not rn_series:
        logger.info("No Rn-222 inference data for liquid concentration plot")
        return

    gas_cfg = (cfg or {}).get("gas_dissolution", {})
    K_H = float(gas_cfg.get("radon_henry_constant", 11.0))
    sample_vol = float((cfg or {}).get("analysis", {}).get("sample_volume_l", 1.0))

    times, activity, errors = _extract_series(rn_series, "rn_bq", "rn_bq_err")
    if not times or not activity:
        return

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # C_liquid = activity / (volume * K_H)
    scale = 1.0 / (sample_vol * K_H) if (sample_vol * K_H) > 0 else 1.0
    conc = [a * scale for a in activity]
    conc_err = [e * scale for e in errors] if errors else None

    times_mpl = guard_mpl_times(times=times)
    _eb = dict(fmt="o", markersize=_DEFAULT_MARKER_SIZE, capsize=_DEFAULT_CAPSIZE,
               elinewidth=_DEFAULT_ELINEWIDTH, alpha=_DEFAULT_ALPHA)

    fig, ax = plt.subplots(figsize=(10, 5))
    if conc_err is not None:
        ax.errorbar(times_mpl, conc, yerr=conc_err, color="#d62728", **_eb)
    else:
        ax.plot(times_mpl, conc, "o", color="#d62728",
                markersize=_DEFAULT_MARKER_SIZE, alpha=_DEFAULT_ALPHA)
    ax.set_ylabel("Rn-222 Concentration [Bq/L]")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(
        f"Implied Radon in Liquid (K_H = {K_H})",
        fontsize=11,
    )
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    ax.yaxis.get_offset_text().set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "radon_in_liquid.png", dpi=300)
    plt.close(fig)

    # --- Total dissolved Rn inventory: activity_Bq / K_H  (cumulative) ---
    # Each interval's dissolved Rn contribution to the liquid (Bq).
    total_scale = 1.0 / K_H if K_H > 0 else 1.0
    total_rn = [a * total_scale for a in activity]
    total_rn_err = [e * total_scale for e in errors] if errors else None

    # Cumulative sum
    cum_rn = list(np.cumsum(total_rn))
    if total_rn_err is not None:
        cum_rn_err = list(np.sqrt(np.cumsum([e**2 for e in total_rn_err])))
    else:
        cum_rn_err = None

    fig_c, ax_c = plt.subplots(figsize=(10, 5))
    if cum_rn_err is not None:
        ax_c.errorbar(times_mpl, cum_rn, yerr=cum_rn_err, color="#d62728", **_eb)
    else:
        ax_c.plot(times_mpl, cum_rn, "o", color="#d62728",
                  markersize=_DEFAULT_MARKER_SIZE, alpha=_DEFAULT_ALPHA)
    ax_c.set_ylabel("Cumulative Dissolved Rn-222 [Bq]")
    ax_c.set_xlabel("Time (UTC)")
    ax_c.set_title(
        f"Cumulative Dissolved Radon Inventory (K_H = {K_H})",
        fontsize=11,
    )
    ax_c.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax_c, times_mpl)
    ax_c.yaxis.get_offset_text().set_visible(False)
    fig_c.autofmt_xdate()
    fig_c.tight_layout()
    fig_c.savefig(out_dir / "radon_in_liquid_cumulative.png", dpi=300)
    plt.close(fig_c)

    logger.info("Saved radon_in_liquid.png, radon_in_liquid_cumulative.png")


def plot_argon_from_leak(
    radon_results: Mapping[str, object],
    out_dir: Path,
    cfg: Mapping[str, object] | None = None,
) -> None:
    """Plot argon entering the system from the equivalent air leak.

    Atmospheric air is ~0.934% argon by volume. The equivalent leaked
    air volume (from the radon mass balance) tells us how much argon
    entered the gas phase. Optionally applies a Henry's law constant
    to estimate the dissolved argon concentration in the liquid.

    Produces four plots:
      - argon_leaked.png:            per-interval argon volume (L)
      - argon_leaked_cumulative.png: cumulative argon volume (L)
      - argon_dissolved.png:         per-interval dissolved concentration (L Ar / L liquid)
      - argon_dissolved_cumulative.png: cumulative dissolved (L Ar / L liquid)
    """
    if not isinstance(radon_results, Mapping):
        return

    gas_cfg = (cfg or {}).get("gas_dissolution", {})
    ar_frac = float(gas_cfg.get("argon_atmospheric_fraction", 0.00934))
    K_H_ar = float(gas_cfg.get("argon_henry_constant", 1.0))
    sample_vol = float((cfg or {}).get("analysis", {}).get("sample_volume_l", 1.0))

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _eb = dict(fmt="o", markersize=_DEFAULT_MARKER_SIZE, capsize=_DEFAULT_CAPSIZE,
               elinewidth=_DEFAULT_ELINEWIDTH, alpha=_DEFAULT_ALPHA)
    _mk = dict(marker="o", linestyle="None", markersize=_DEFAULT_MARKER_SIZE,
               alpha=_DEFAULT_ALPHA)

    # -- Per-interval argon volume --
    volume_series = radon_results.get("volume_equiv")
    times, volumes_m3, vol_errs = _extract_series(volume_series, "v_m3", "v_m3_err")
    if not times or not volumes_m3:
        logger.info("No leak-volume data for argon plots")
        return

    # Convert m^3 air -> L argon
    ar_L = [v * 1000.0 * ar_frac for v in volumes_m3]
    ar_L_err = [e * 1000.0 * ar_frac for e in vol_errs] if vol_errs else None

    times_mpl = guard_mpl_times(times=times)

    # Per-interval argon leaked (liters)
    fig, ax = plt.subplots(figsize=(10, 5))
    if ar_L_err is not None:
        ax.errorbar(times_mpl, ar_L, yerr=ar_L_err, color="#1f77b4", **_eb)
    else:
        ax.plot(times_mpl, ar_L, color="#1f77b4", **_mk)
    ax.set_ylabel("Argon Leaked [L per interval]")
    ax.set_xlabel("Time (UTC)")
    ax.set_title(f"Argon from Air Leak (Ar fraction = {ar_frac*100:.2f}%)", fontsize=11)
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    ax.yaxis.get_offset_text().set_visible(False)
    fig.autofmt_xdate()
    fig.tight_layout()
    fig.savefig(out_dir / "argon_leaked.png", dpi=300)
    plt.close(fig)

    # Cumulative argon leaked (liters)
    cumulative_series = radon_results.get("volume_cumulative")
    if cumulative_series:
        cum_times, cum_m3, cum_errs = _extract_series(
            cumulative_series, "v_m3_cum", "v_m3_cum_err"
        )
        if cum_times and cum_m3:
            cum_ar_L = [v * 1000.0 * ar_frac for v in cum_m3]
            cum_ar_err = [e * 1000.0 * ar_frac for e in cum_errs] if cum_errs else None
            times_mpl_cum = guard_mpl_times(times=cum_times)

            fig_c, ax_c = plt.subplots(figsize=(10, 5))
            if cum_ar_err is not None:
                ax_c.errorbar(times_mpl_cum, cum_ar_L, yerr=cum_ar_err,
                              color="#1f77b4", **_eb)
            else:
                ax_c.plot(times_mpl_cum, cum_ar_L, color="#1f77b4", **_mk)
            ax_c.set_ylabel("Cumulative Argon Leaked [L]")
            ax_c.set_xlabel("Time (UTC)")
            ax_c.set_title("Cumulative Argon from Air Leak", fontsize=11)
            ax_c.ticklabel_format(axis="y", style="plain")
            setup_time_axis(ax_c, times_mpl_cum)
            ax_c.yaxis.get_offset_text().set_visible(False)
            fig_c.autofmt_xdate()
            fig_c.tight_layout()
            fig_c.savefig(out_dir / "argon_leaked_cumulative.png", dpi=300)
            plt.close(fig_c)

    # -- Dissolved argon concentration (Henry's law) --
    if K_H_ar > 0 and sample_vol > 0:
        # Dissolved Ar per interval: argon_L / (K_H * sample_vol_L)
        # Units: L_Ar / L_liquid (dimensionless volume ratio)
        diss_scale = 1.0 / (K_H_ar * sample_vol)
        diss = [a * diss_scale for a in ar_L]
        diss_err = [e * diss_scale for e in ar_L_err] if ar_L_err else None

        fig_d, ax_d = plt.subplots(figsize=(10, 5))
        if diss_err is not None:
            ax_d.errorbar(times_mpl, diss, yerr=diss_err, color="#2ca02c", **_eb)
        else:
            ax_d.plot(times_mpl, diss, color="#2ca02c", **_mk)
        ax_d.set_ylabel("Dissolved Ar [L$_{Ar}$ / L$_{liquid}$]")
        ax_d.set_xlabel("Time (UTC)")
        ax_d.set_title(
            f"Dissolved Argon (K_H = {K_H_ar}, V = {sample_vol:.0f} L)",
            fontsize=11,
        )
        ax_d.ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))
        setup_time_axis(ax_d, times_mpl)
        fig_d.autofmt_xdate()
        fig_d.tight_layout()
        fig_d.savefig(out_dir / "argon_dissolved.png", dpi=300)
        plt.close(fig_d)

        # Cumulative dissolved argon (concentration)
        if cumulative_series and cum_times and cum_m3:
            cum_diss = [a * diss_scale for a in cum_ar_L]
            cum_diss_err = [e * diss_scale for e in cum_ar_err] if cum_ar_err else None

            fig_dc, ax_dc = plt.subplots(figsize=(10, 5))
            if cum_diss_err is not None:
                ax_dc.errorbar(times_mpl_cum, cum_diss, yerr=cum_diss_err,
                               color="#2ca02c", **_eb)
            else:
                ax_dc.plot(times_mpl_cum, cum_diss, color="#2ca02c", **_mk)
            ax_dc.set_ylabel("Cumulative Dissolved Ar [L$_{Ar}$ / L$_{liquid}$]")
            ax_dc.set_xlabel("Time (UTC)")
            ax_dc.set_title("Cumulative Dissolved Argon", fontsize=11)
            ax_dc.ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))
            setup_time_axis(ax_dc, times_mpl_cum)
            fig_dc.autofmt_xdate()
            fig_dc.tight_layout()
            fig_dc.savefig(out_dir / "argon_dissolved_cumulative.png", dpi=300)
            plt.close(fig_dc)

        # -- Total dissolved argon volume (L_Ar total in liquid) --
        # Per-interval: dissolved_conc * sample_vol = argon_L / K_H
        total_diss = [a / K_H_ar for a in ar_L]
        total_diss_err = [e / K_H_ar for e in ar_L_err] if ar_L_err else None

        fig_t, ax_t = plt.subplots(figsize=(10, 5))
        if total_diss_err is not None:
            ax_t.errorbar(times_mpl, total_diss, yerr=total_diss_err,
                          color="#9467bd", **_eb)
        else:
            ax_t.plot(times_mpl, total_diss, color="#9467bd", **_mk)
        ax_t.set_ylabel("Dissolved Argon [L$_{Ar}$]")
        ax_t.set_xlabel("Time (UTC)")
        ax_t.set_title(
            f"Total Dissolved Argon per Interval (K_H = {K_H_ar})",
            fontsize=11,
        )
        ax_t.ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))
        setup_time_axis(ax_t, times_mpl)
        fig_t.autofmt_xdate()
        fig_t.tight_layout()
        fig_t.savefig(out_dir / "argon_dissolved_total.png", dpi=300)
        plt.close(fig_t)

        # Cumulative total dissolved argon volume
        cum_total_diss = list(np.cumsum(total_diss))
        if total_diss_err is not None:
            cum_total_err = list(np.sqrt(np.cumsum([e**2 for e in total_diss_err])))
        else:
            cum_total_err = None

        fig_tc, ax_tc = plt.subplots(figsize=(10, 5))
        if cum_total_err is not None:
            ax_tc.errorbar(times_mpl, cum_total_diss, yerr=cum_total_err,
                           color="#9467bd", **_eb)
        else:
            ax_tc.plot(times_mpl, cum_total_diss, color="#9467bd", **_mk)
        ax_tc.set_ylabel("Cumulative Dissolved Argon [L$_{Ar}$]")
        ax_tc.set_xlabel("Time (UTC)")
        ax_tc.set_title("Cumulative Dissolved Argon Volume", fontsize=11)
        ax_tc.ticklabel_format(axis="y", style="scientific", scilimits=(-3, 3))
        setup_time_axis(ax_tc, times_mpl)
        fig_tc.autofmt_xdate()
        fig_tc.tight_layout()
        fig_tc.savefig(out_dir / "argon_dissolved_total_cumulative.png", dpi=300)
        plt.close(fig_tc)

    logger.info("Saved argon leak/dissolution plots")


__all__ = [
    "plot_rn_inferred_vs_time",
    "plot_ambient_rn_vs_time",
    "plot_volume_equiv_vs_time",
    "plot_radon_in_liquid",
    "plot_argon_from_leak",
]

