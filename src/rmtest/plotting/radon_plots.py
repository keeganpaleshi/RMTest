"""Plotting helpers for radon inference results."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Iterable, Mapping

import matplotlib as _mpl

_mpl.use("Agg")
import matplotlib.pyplot as plt

from plot_utils._time_utils import guard_mpl_times, setup_time_axis


logger = logging.getLogger(__name__)


def _extract_times(series: Iterable[Mapping[str, Any]]):
    times = [float(entry.get("t")) for entry in series if entry.get("t") is not None]
    values = [entry for entry in series if entry.get("t") is not None]
    return times, values


def _format_weight_legend(meta: Mapping[str, Any]) -> str:
    weights = meta.get("source_weights") if isinstance(meta, Mapping) else None
    if not isinstance(weights, Mapping):
        return ""
    parts = [f"{iso}: {weights[iso]:.3g}" for iso in weights]
    return ", ".join(parts)


def _meta_text(meta: Mapping[str, Any] | None) -> str:
    if not isinstance(meta, Mapping):
        return ""
    effs = meta.get("detection_efficiency")
    if isinstance(effs, Mapping) and effs:
        eff_str = ", ".join(f"{iso}: {val}" for iso, val in effs.items())
    else:
        eff_str = "n/a"
    transport = meta.get("transport_efficiency")
    retention = meta.get("retention_efficiency")
    chain = meta.get("chain_correction", "none")
    return (
        "Detection eff: "
        f"{eff_str}\nTransport eff: {transport}\nRetention eff: {retention}\nChain: {chain}"
    )


def plot_rn_inferred_vs_time(
    radon_inference: Mapping[str, Any] | None,
    outdir: str | Path,
    *,
    filename: str = "rn_inferred.png",
) -> None:
    """Plot inferred Rn-222 activity over time."""

    if not isinstance(radon_inference, Mapping):
        logger.debug("radon inference data missing; skipping Rn plot")
        return

    series = radon_inference.get("rn_inferred")
    if not isinstance(series, Iterable):
        logger.debug("rn_inferred series missing; skipping plot")
        return

    times_raw, entries = _extract_times(series)
    if not entries:
        logger.debug("rn_inferred series empty; skipping plot")
        return

    activity = [float(entry.get("rn_bq", 0.0)) for entry in entries]
    mpl_times = guard_mpl_times(times=times_raw)

    legend_weights = _format_weight_legend(radon_inference.get("meta", {}))
    label = "Rn-222 inferred"
    if legend_weights:
        label += f" ({legend_weights})"

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(mpl_times, activity, marker="o", linestyle="-", label=label)
    ax.set_ylabel("Activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    if legend_weights:
        ax.legend(loc="best")
    setup_time_axis(ax, mpl_times)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    meta_text = _meta_text(radon_inference.get("meta"))
    if meta_text:
        ax.text(
            0.01,
            0.99,
            meta_text,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize="small",
            bbox={"facecolor": "white", "alpha": 0.7, "edgecolor": "none"},
        )
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=300)
    plt.close(fig)


def plot_ambient_rn_vs_time(
    radon_inference: Mapping[str, Any] | None,
    outdir: str | Path,
    *,
    filename: str = "ambient_rn.png",
) -> None:
    """Plot ambient radon concentration used for inference."""

    if not isinstance(radon_inference, Mapping):
        return
    series = radon_inference.get("ambient_rn")
    if not isinstance(series, Iterable):
        return

    filtered = [entry for entry in series if entry.get("rn_bq_per_m3") is not None]
    if not filtered:
        logger.debug("ambient radon series empty; skipping plot")
        return

    times, entries = _extract_times(filtered)
    values = [float(entry.get("rn_bq_per_m3", 0.0)) for entry in entries]
    mpl_times = guard_mpl_times(times=times)

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.step(mpl_times, values, where="mid")
    ax.set_ylabel("Ambient Rn [Bq/m³]")
    ax.set_xlabel("Time (UTC)")
    setup_time_axis(ax, mpl_times)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=300)
    plt.close(fig)


def plot_volume_equiv_vs_time(
    radon_inference: Mapping[str, Any] | None,
    outdir: str | Path,
    *,
    filename: str = "rn_volume.png",
) -> None:
    """Plot equivalent sampled volume and its cumulative integral."""

    if not isinstance(radon_inference, Mapping):
        return

    vol_series = radon_inference.get("volume_equiv")
    if not isinstance(vol_series, Iterable):
        return
    vol_entries = [entry for entry in vol_series if entry.get("v_m3") is not None]
    if not vol_entries:
        logger.debug("volume_equiv series empty; skipping plot")
        return

    times, entries = _extract_times(vol_entries)
    v_m3 = [float(entry.get("v_m3", 0.0)) for entry in entries]
    v_lpm = [float(entry.get("v_lpm", 0.0)) for entry in entries]
    mpl_times = guard_mpl_times(times=times)

    cumulative = radon_inference.get("volume_cumulative")
    cumulative_entries = []
    if isinstance(cumulative, Iterable):
        cumulative_entries = [
            entry for entry in cumulative if entry.get("v_m3_cum") is not None
        ]

    outdir = Path(outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    if cumulative_entries:
        fig, (ax, ax_cum) = plt.subplots(2, 1, sharex=True, figsize=(8, 6))
    else:
        fig, ax = plt.subplots()
        ax_cum = None

    ax.plot(mpl_times, v_m3, marker="o", linestyle="-", label="Volume [m³]")
    ax2 = ax.twinx()
    ax2.plot(mpl_times, v_lpm, color="tab:orange", linestyle="--", label="Flow [L/min]")
    ax.set_ylabel("Equivalent volume [m³]")
    ax2.set_ylabel("Equivalent flow [L/min]")
    ax.set_xlabel("Time (UTC)")
    setup_time_axis(ax, mpl_times)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    ax2.yaxis.get_offset_text().set_visible(False)
    lines, labels = ax.get_legend_handles_labels()
    lines2, labels2 = ax2.get_legend_handles_labels()
    ax.legend(lines + lines2, labels + labels2, loc="upper left")

    if ax_cum is not None:
        cum_times, cum_entries = _extract_times(cumulative_entries)
        cum_vals = [float(entry.get("v_m3_cum", 0.0)) for entry in cum_entries]
        cum_mpl = guard_mpl_times(times=cum_times)
        ax_cum.plot(cum_mpl, cum_vals, marker="o", linestyle="-", color="tab:green")
        ax_cum.set_ylabel("Cumulative volume [m³]")
        ax_cum.set_xlabel("Time (UTC)")
        setup_time_axis(ax_cum, cum_mpl)
        ax_cum.yaxis.get_offset_text().set_visible(False)

    fig.tight_layout()
    fig.savefig(outdir / filename, dpi=300)
    plt.close(fig)


__all__ = [
    "plot_rn_inferred_vs_time",
    "plot_ambient_rn_vs_time",
    "plot_volume_equiv_vs_time",
]

