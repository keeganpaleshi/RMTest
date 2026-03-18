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


def plot_ambient_rn_vs_time(radon_results: Mapping[str, object], out_dir: Path) -> None:
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
    ax.plot(times_mpl, values, marker="o", linestyle="None", markersize=_DEFAULT_MARKER_SIZE, color="#1f77b4", alpha=_DEFAULT_ALPHA)
    ax.set_ylabel("Ambient Radon [Bq/m$^3$]")
    ax.set_xlabel("Time (UTC)")
    ax.set_title("External Ambient Radon", fontsize=11)
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)

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

    # --- flow rate (L/min) ---
    if volumes_lpm:
        paired_len = min(len(times_mpl), len(volumes_lpm))
        if paired_len == 0:
            logger.info("Leak-rate data present but no overlapping timestamps; skipping flow plot")
        else:
            times_flow = times_mpl[:paired_len]
            volumes_flow = volumes_lpm[:paired_len]
            flow_errs = lpm_errs[:paired_len] if lpm_errs is not None else None
            fig_flow, ax_flow = plt.subplots(figsize=(10, 5))
            if flow_errs is not None:
                ax_flow.errorbar(times_flow, volumes_flow, yerr=flow_errs, color="#ff7f0e", **_eb)
            else:
                ax_flow.plot(times_flow, volumes_flow, color="#ff7f0e", **_mk)
            ax_flow.set_ylabel("Leak Rate [L/min]")
            ax_flow.set_xlabel("Time (UTC)")
            ax_flow.set_title("Equivalent Leak Rate", fontsize=11)
            ax_flow.ticklabel_format(axis="y", style="plain")
            setup_time_axis(ax_flow, times_flow)
            ax_flow.yaxis.get_offset_text().set_visible(False)
            fig_flow.autofmt_xdate()
            fig_flow.tight_layout()
            fig_flow.savefig(out_dir / _fname("equivalent_flow.png"), dpi=300)
            plt.close(fig_flow)

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


__all__ = [
    "plot_rn_inferred_vs_time",
    "plot_ambient_rn_vs_time",
    "plot_volume_equiv_vs_time",
]

