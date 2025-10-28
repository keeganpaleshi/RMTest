# -----------------------------------------------------
# plot_utils.py
# -----------------------------------------------------

import os
from collections import OrderedDict
from collections.abc import Mapping

import numpy as np
import matplotlib as _mpl

_mpl.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from datetime import datetime, timezone
from pathlib import Path
from color_schemes import COLOR_SCHEMES
from constants import PO214, PO218, PO210, RN222
from calibration import gaussian, emg_left
from fitting import make_linear_bkg
from .paths import get_targets
from ._time_utils import guard_mpl_times, setup_time_axis

# Half-life constants used for the time-series overlay [seconds]
PO214_HALF_LIFE_S = PO214.half_life_s
PO218_HALF_LIFE_S = PO218.half_life_s


def _errorbar_kwargs(color=None, *, label=None):
    """Return styling options for visible error bars."""

    kwargs = {
        "fmt": "o",
        "capsize": 3,
        "capthick": 1,
        "elinewidth": 1,
        "barsabove": True,
    }
    if color is not None:
        kwargs["color"] = color
        kwargs["markerfacecolor"] = color
        kwargs["markeredgecolor"] = color
    if label is not None:
        kwargs["label"] = label
    return kwargs

__all__ = [
    "extract_time_series",
    "plot_time_series",
    "plot_spectrum",
    "plot_equivalent_air",
    "plot_modeled_radon_activity",
    "plot_radon_activity",
    "plot_total_radon",
    "plot_radon_trend",
    "plot_radon_activity_full",
    "plot_total_radon_full",
    "plot_radon_trend_full",
]


def _counts_per_bin(density, widths):
    """Convert a spectral density into expected counts per bin."""

    density_arr = np.asarray(density, dtype=float)
    widths_arr = np.asarray(widths, dtype=float)
    counts = np.multiply(density_arr, widths_arr)
    if counts.shape != density_arr.shape:
        raise ValueError("density and widths must share the same shape")
    return counts


def _coerce_timestamp_seconds(value):
    """Return ``value`` as UNIX seconds or ``None`` if conversion fails."""

    if value is None:
        return None

    if isinstance(value, (int, float)):
        return float(value)

    if isinstance(value, np.generic):
        if np.issubdtype(type(value), np.integer) or np.issubdtype(
            type(value), np.floating
        ):
            return float(value)
        if np.issubdtype(type(value), np.datetime64):
            return float(value.astype("int64") / 1e9)

    if isinstance(value, np.datetime64):
        return float(value.astype("int64") / 1e9)

    if isinstance(value, datetime):
        return float(value.timestamp())

    if hasattr(value, "to_pydatetime"):
        try:
            return float(value.to_pydatetime().timestamp())
        except Exception:
            pass

    if isinstance(value, str):
        try:
            parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
            return float(parsed.timestamp())
        except ValueError:
            try:
                parsed_np = np.datetime64(value)
            except ValueError:
                try:
                    return float(value)
                except ValueError:
                    return None
            return float(parsed_np.astype("int64") / 1e9)

    return None


def _resolve_run_periods(config, default_start, default_end):
    """Return sorted run periods clipped to ``[default_start, default_end]``."""

    periods = None
    if isinstance(config, Mapping):
        periods = config.get("run_periods")
        if not periods:
            analysis_cfg = config.get("analysis")
            if isinstance(analysis_cfg, Mapping):
                periods = analysis_cfg.get("run_periods")

    if not periods:
        return [(float(default_start), float(default_end))]

    resolved = []
    for entry in periods:
        if not isinstance(entry, (list, tuple)) or len(entry) != 2:
            continue
        start_val = _coerce_timestamp_seconds(entry[0])
        end_val = _coerce_timestamp_seconds(entry[1])
        if start_val is None or end_val is None:
            continue
        start = max(float(start_val), float(default_start))
        end = min(float(end_val), float(default_end))
        if not np.isfinite(start) or not np.isfinite(end):
            continue
        if end <= start:
            continue
        resolved.append((start, end))

    if not resolved:
        return [(float(default_start), float(default_end))]

    resolved.sort(key=lambda pair: pair[0])
    return resolved


def _build_time_segments(
    timestamps,
    *,
    periods,
    bin_mode,
    bin_width_s,
    time_bins_fallback,
    t_start,
):
    """Return per-run binning metadata for the time-series plots."""

    segments = []
    timestamps = np.asarray(timestamps, dtype=float)
    fallback_bins = max(1, int(time_bins_fallback))

    for start, end in periods:
        duration = float(end) - float(start)
        if duration <= 0:
            continue

        if bin_mode in ("fd", "auto"):
            mask = (timestamps >= float(start)) & (timestamps <= float(end))
            data = timestamps[mask]
            if data.size < 2:
                n_bins = 1
            else:
                rel = data - float(start)
                q25, q75 = np.percentile(rel, [25, 75])
                iqr = q75 - q25
                if iqr <= 0:
                    n_bins = fallback_bins
                else:
                    bin_width = 2 * iqr / (rel.size ** (1.0 / 3.0))
                    bin_width = float(bin_width)
                    data_range = float(rel.max() - rel.min())
                    if not np.isfinite(bin_width) or bin_width <= 0:
                        n_bins = fallback_bins
                    else:
                        n_bins = max(1, int(np.ceil(data_range / bin_width)))
        else:
            width = float(bin_width_s)
            if not np.isfinite(width) or width <= 0:
                width = 1.0

            duration = float(end) - float(start)
            n_bins = max(1, int(np.floor(duration / width)))
            edges = float(start) + width * np.arange(n_bins + 1, dtype=float)
            if edges.size < 2:
                continue
            # Always terminate the final edge at the run end.  This preserves the
            # configured number of whole-width bins while allowing the trailing
            # bin to absorb any fractional remainder inside the run period.
            edges[-1] = float(end)
            edges = np.asarray(edges, dtype=float)
        if bin_mode in ("fd", "auto"):
            edges = np.linspace(float(start), float(end), int(n_bins) + 1)

        edges = np.asarray(edges, dtype=float)
        if edges.size < 2:
            continue
        bin_widths = np.diff(edges)
        if np.any(bin_widths <= 0):
            continue
        centers_abs = edges[:-1] + bin_widths / 2.0
        centers_rel_global = centers_abs - float(t_start)
        centers_rel_segment = centers_abs - float(edges[0])
        centers_mpl = guard_mpl_times(times=centers_abs)

        segments.append(
            {
                "start": float(edges[0]),
                "end": float(edges[-1]),
                "edges_abs": edges,
                "bin_widths": bin_widths,
                "centers_abs": centers_abs,
                "centers_rel_global": centers_rel_global,
                "centers_rel_segment": centers_rel_segment,
                "centers_mpl": centers_mpl,
                "centers_elapsed_hours": centers_rel_segment / 3600.0,
                "counts": {},
            }
        )

    return segments


def _isoformat_utc(seconds):
    """Return an ISO-8601 representation with a ``Z`` suffix."""

    dt = datetime.fromtimestamp(float(seconds), tz=timezone.utc)
    return dt.isoformat().replace("+00:00", "Z")


def extract_time_series(timestamps, energies, window, t_start, t_end, bin_width_s=1.0):
    """Return histogram counts for events within an energy window.

    Parameters
    ----------
    timestamps : array-like
        Absolute event times in seconds.
    energies : array-like
        Event energies in MeV.
    window : tuple(float, float) or None
        Inclusive energy range. If ``None`` an empty array is returned.
    t_start, t_end : float
        Start and end times for the histogram.
    bin_width_s : float, optional
        Width of each time bin in seconds.

    Returns
    -------
    counts : np.ndarray
        Counts in each time bin.
    edges : np.ndarray
        Bin edges relative to ``t_start``.
    """

    if window is None:
        return np.array([]), np.array([])

    lo, hi = window
    timestamps = np.asarray(timestamps, dtype=float)
    energies = np.asarray(energies, dtype=float)

    mask = (
        (energies >= lo)
        & (energies <= hi)
        & (timestamps >= float(t_start))
        & (timestamps <= float(t_end))
    )

    rel_times = timestamps[mask] - float(t_start)

    n_bins = int(np.floor((float(t_end) - float(t_start)) / float(bin_width_s)))
    if n_bins < 1:
        n_bins = 1
    edges = np.arange(0, (n_bins + 1) * float(bin_width_s), float(bin_width_s))
    counts, _ = np.histogram(rel_times, bins=edges)
    return counts, edges


def plot_time_series(
    all_timestamps,
    all_energies,
    fit_results,
    t_start,
    t_end,
    config,
    out_png,
    hl_po214=None,
    hl_po218=None,
    *,
    model_errors=None,
    **_legacy_kwargs,
):
    """
    all_timestamps: 1D np.ndarray of absolute UNIX times (s)
    all_energies:   1D np.ndarray of energies (MeV)
    fit_results:    dict from fit_time_series(...)
    t_start, t_end: floats (absolute UNIX times) for the fit window
    config:         JSON dict or nested configuration
    out_png:        output path for the PNG file
    hl_po214, hl_po218: optional half-life values in seconds for the
        Po-214 and Po-218 time-series overlays.  When not provided, the
        values are looked up using the configuration keys ``hl_po214``
        and ``hl_po218`` under ``time_fit`` and default to the Rn-222
        half-life.  This ensures the daughter activities are propagated
        using the parent radon decay constant.  When Po-210 is plotted
        the overlay uses the ``hl_po210`` configuration value.
    model_errors : dict[str, array-like], optional
        Mapping of isotope name to 1D arrays of uncertainties for the
        model curve. When provided, ``fill_between`` is used to draw
        +/-1 sigma bands around the corresponding model.
    """

    if fit_results is None:
        fit_results = {}

    # Convert timestamps to UNIX seconds when datetime64 or datetime objects
    ts_array = np.asarray(all_timestamps)
    if np.issubdtype(ts_array.dtype, "datetime64"):
        all_timestamps = ts_array.astype("int64") / 1e9
    elif np.issubdtype(ts_array.dtype, np.object_):
        if ts_array.size > 0 and isinstance(ts_array.flat[0], datetime):
            all_timestamps = np.array([dt.timestamp() for dt in ts_array], dtype=float)
        else:
            all_timestamps = ts_array.astype(float)
    else:
        all_timestamps = ts_array.astype(float)

    if isinstance(t_start, datetime):
        t_start = t_start.timestamp()
    elif isinstance(t_start, np.datetime64):
        t_start = float(t_start.astype("int64") / 1e9)
    if isinstance(t_end, datetime):
        t_end = t_end.timestamp()
    elif isinstance(t_end, np.datetime64):
        t_end = float(t_end.astype("int64") / 1e9)

    def _cfg_get(cfg, key, default=None):
        """Lookup ``key`` in ``cfg``.

        The search first checks the ``time_fit`` sub-dictionary, then the
        top level of ``cfg``.  Configuration keys should use lowercase
        names such as ``hl_po214`` and ``hl_po218``.
        """

        if not isinstance(cfg, dict):
            return default

        sub = cfg.get("time_fit", {})
        if isinstance(sub, dict) and key in sub:
            return sub[key]

        return cfg.get(key, default)

    default_const = config.get("nuclide_constants", {})
    default_rn = default_const.get("Rn222", RN222).half_life_s
    default210 = default_const.get("Po210", PO210).half_life_s

    if hl_po214 is None and "hl_po214" in _legacy_kwargs:
        hl_po214 = _legacy_kwargs.pop("hl_po214")
    if hl_po218 is None and "hl_po218" in _legacy_kwargs:
        hl_po218 = _legacy_kwargs.pop("hl_po218")

    def _hl_param(name, default):
        val = _cfg_get(config, name, default)
        if isinstance(val, list):
            return float(val[0]) if val else float(default)
        return float(default) if val is None else float(val)

    def _eff_param(name, default):
        val = _cfg_get(config, name, default)
        if isinstance(val, (list, tuple, np.ndarray)):
            if len(val) == 0:
                return float(default)
            return float(val[0])
        if val is None:
            return float(default)
        return float(val)

    po214_hl = float(hl_po214) if hl_po214 is not None else _hl_param("hl_po214", default_rn)
    po218_hl = float(hl_po218) if hl_po218 is not None else _hl_param("hl_po218", default_rn)

    if po214_hl <= 0:
        raise ValueError("hl_po214 must be positive")
    if po218_hl <= 0:
        raise ValueError("hl_po218 must be positive")

    iso_params = {
        "Po210": {
            # Energy window for Po-210 events (optional)
            "window": _cfg_get(config, "window_po210"),
            "eff": _eff_param("eff_po210", 1.0),
            "half_life": _hl_param("hl_po210", default210),
        },
        "Po218": {
            # Energy window for Po-218 events
            "window": _cfg_get(config, "window_po218"),
            "eff": _eff_param("eff_po218", 1.0),
            "half_life": po218_hl,
        },
        "Po214": {
            # Energy window for Po-214 events
            "window": _cfg_get(config, "window_po214"),
            "eff": _eff_param("eff_po214", 1.0),
            "half_life": po214_hl,
        },
    }
    iso_list = [iso for iso, p in iso_params.items() if p["window"] is not None]

    bin_mode = str(
        config.get(
            "plot_time_binning_mode",
            config.get("time_bin_mode", "fixed"),
        )
    ).lower()
    bin_width_s = float(
        config.get("plot_time_bin_width_s", config.get("time_bin_s", 3600.0))
    )
    time_bins_fallback = int(config.get("time_bins_fallback", 1))

    periods = _resolve_run_periods(config, t_start, t_end)
    segments = _build_time_segments(
        all_timestamps,
        periods=periods,
        bin_mode=bin_mode,
        bin_width_s=bin_width_s,
        time_bins_fallback=time_bins_fallback,
        t_start=t_start,
    )
    if not segments:
        segments = _build_time_segments(
            all_timestamps,
            periods=[(float(t_start), float(t_end))],
            bin_mode=bin_mode,
            bin_width_s=bin_width_s,
            time_bins_fallback=time_bins_fallback,
            t_start=t_start,
        )

    normalise_rate = bool(config.get("plot_time_normalise_rate", False))
    style = str(config.get("plot_time_style", "steps")).lower()

    plt.figure(figsize=(8, 6))
    palette_name = str(config.get("palette", "default"))
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    colors = {}
    for iso_key, default in (
        ("Po214", "#d62728"),
        ("Po218", "#1f77b4"),
        ("Po210", "#2ca02c"),
    ):
        if iso_key in iso_list:
            colors[iso_key] = palette.get(iso_key, default)

    width_lists = [seg["bin_widths"] for seg in segments if seg["bin_widths"].size]
    widths_all = (
        np.concatenate(width_lists) if width_lists else np.array([], dtype=float)
    )
    centers_rel_lists = [
        seg["centers_rel_global"] for seg in segments if seg["centers_rel_global"].size
    ]
    centers_rel_all = (
        np.concatenate(centers_rel_lists)
        if centers_rel_lists
        else np.array([], dtype=float)
    )
    centers_mpl_lists = [
        np.asarray(seg["centers_mpl"]) for seg in segments if np.asarray(seg["centers_mpl"]).size
    ]
    if centers_mpl_lists:
        centers_mpl_all = np.concatenate(centers_mpl_lists)
    else:
        if segments:
            centers_mpl_all = guard_mpl_times(
                times=[segments[0]["start"], segments[-1]["end"]]
            )
        else:
            centers_mpl_all = guard_mpl_times(times=[t_start, t_end])

    ts_counts_raw_arrays: dict[str, np.ndarray] = {}
    counts_for_json_lists: dict[str, list[int]] = {}

    for iso in iso_list:
        emin, emax = iso_params[iso]["window"]
        label_used = False
        model_label_used = False
        raw_segment_counts: list[np.ndarray] = []

        model_err_arr = None
        model_err_idx = 0
        if model_errors and iso in model_errors:
            model_err_arr = np.asarray(model_errors[iso], dtype=float)

        has_fit = any(k in fit_results for k in (f"E_{iso}", "E"))
        fit_ok = bool(
            fit_results.get("fit_valid", True)
            and fit_results.get(f"fit_valid_{iso}", True)
        )
        if has_fit and fit_ok:
            lam = np.log(2.0) / iso_params[iso]["half_life"]
            eff = iso_params[iso]["eff"]

            E_iso = fit_results.get(f"E_{iso}", fit_results.get("E", 0.0))
            B_iso = fit_results.get(f"B_{iso}", fit_results.get("B", 0.0))
            N0_iso = fit_results.get(f"N0_{iso}", fit_results.get("N0", 0.0))
        else:
            lam = eff = E_iso = B_iso = N0_iso = None

        for seg_idx, seg in enumerate(segments):
            centers_mpl = np.asarray(seg["centers_mpl"])
            bin_widths_seg = seg["bin_widths"]
            edges_abs = seg["edges_abs"]
            if centers_mpl.size == 0 or bin_widths_seg.size == 0:
                continue

            mask_seg = (
                (all_energies >= emin)
                & (all_energies <= emax)
                & (all_timestamps >= seg["start"])
                & (all_timestamps <= seg["end"])
            )
            times_seg = all_timestamps[mask_seg]
            counts_hist = np.histogram(times_seg, bins=edges_abs)[0].astype(float)
            seg["counts"].setdefault(iso, counts_hist.astype(int))
            raw_segment_counts.append(counts_hist.astype(float))

            errors_seg = np.sqrt(counts_hist)
            counts_plot = counts_hist.copy()
            errors_plot = errors_seg.copy()

            if normalise_rate:
                with np.errstate(divide="ignore", invalid="ignore"):
                    counts_plot = np.divide(
                        counts_plot,
                        bin_widths_seg,
                        out=np.zeros_like(counts_plot, dtype=float),
                        where=bin_widths_seg > 0,
                    )
                    errors_plot = np.divide(
                        errors_plot,
                        bin_widths_seg,
                        out=np.zeros_like(errors_plot, dtype=float),
                        where=bin_widths_seg > 0,
                    )
                counts_plot = np.nan_to_num(
                    counts_plot, nan=0.0, posinf=0.0, neginf=0.0
                )
                errors_plot = np.nan_to_num(
                    errors_plot, nan=0.0, posinf=0.0, neginf=0.0
                )

            if counts_plot.size == 0:
                if model_err_arr is not None:
                    model_err_idx += counts_hist.size
                continue

            label = f"Data {iso}" if not label_used else None
            color = colors.get(iso, palette.get("hist", "#1f77b4"))

            if style == "lines":
                err_kwargs = {
                    "fmt": "o-",
                    "color": color,
                    "capsize": 3,
                    "elinewidth": 1,
                    "capthick": 1,
                    "barsabove": True,
                }
                if label is not None:
                    err_kwargs["label"] = label
                plt.errorbar(
                    centers_mpl,
                    counts_plot,
                    yerr=errors_plot,
                    **err_kwargs,
                )
            else:
                step_kwargs = {"where": "mid", "color": color}
                if label is not None:
                    step_kwargs["label"] = label
                plt.step(centers_mpl, counts_plot, **step_kwargs)
                plt.errorbar(
                    centers_mpl,
                    counts_plot,
                    yerr=errors_plot,
                    fmt="none",
                    ecolor=color,
                    elinewidth=1.0,
                    capsize=3,
                )

            label_used = label_used or label is not None

            if has_fit and fit_ok:
                centers_rel = seg["centers_rel_global"]
                r_rel = (
                    eff
                    * (
                        E_iso * (1.0 - np.exp(-lam * centers_rel))
                        + lam * N0_iso * np.exp(-lam * centers_rel)
                    )
                    + B_iso
                )
                model_counts_seg = (
                    r_rel if normalise_rate else r_rel * bin_widths_seg
                )
                model_label = f"Model {iso}" if not model_label_used else None
                plt.plot(
                    centers_mpl,
                    model_counts_seg,
                    color=color,
                    lw=2,
                    ls="--",
                    label=model_label,
                )
                if model_label is not None:
                    model_label_used = True

                if model_err_arr is not None:
                    end_idx = model_err_idx + counts_hist.size
                    if end_idx <= model_err_arr.size:
                        err_seg = model_err_arr[model_err_idx:end_idx]
                        kw = {"step": "mid"} if style != "lines" else {}
                        plt.fill_between(
                            centers_mpl,
                            model_counts_seg - err_seg,
                            model_counts_seg + err_seg,
                            color=color,
                            alpha=0.3,
                            **kw,
                        )
                    else:
                        raise ValueError("model_errors array length mismatch")

            if model_err_arr is not None:
                model_err_idx += counts_hist.size

        if model_err_arr is not None and model_err_idx != model_err_arr.size:
            raise ValueError("model_errors array length mismatch")

        if raw_segment_counts:
            flat_counts = np.concatenate(raw_segment_counts).astype(int)
        else:
            flat_counts = np.array([], dtype=int)
        ts_counts_raw_arrays[iso] = flat_counts
        counts_for_json_lists[iso] = [int(v) for v in flat_counts.tolist()]

    plt.xlabel("Time (UTC)")
    plt.ylabel("Counts / s" if normalise_rate else "Counts per bin")
    title_isos = " & ".join(iso_list)
    plt.title(f"{title_isos} Time Series Fit")
    handles, labels = plt.gca().get_legend_handles_labels()
    if handles:
        plt.legend(fontsize="small")

    ax = plt.gca()
    setup_time_axis(ax, centers_mpl_all)
    plt.gcf().autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    ax.ticklabel_format(axis="y", style="plain")
    plt.tight_layout()
    targets = get_targets(config, out_png)
    for p in targets.values():
        plt.savefig(p, dpi=300)
    plt.close()

    segments_payload = []
    if config.get("dump_time_series_json", False):
        import json

        for seg in segments:
            payload = {
                "start_utc": _isoformat_utc(seg["start"]),
                "end_utc": _isoformat_utc(seg["end"]),
                "bin_edges_unix_s": seg["edges_abs"].tolist(),
                "bin_edges_utc": [
                    _isoformat_utc(val) for val in seg["edges_abs"]
                ],
                "bin_widths_s": seg["bin_widths"].tolist(),
                "bin_centers_unix_s": seg["centers_abs"].tolist(),
                "bin_centers_utc": [
                    _isoformat_utc(val) for val in seg["centers_abs"]
                ],
                "bin_centers_elapsed_hours": seg[
                    "centers_elapsed_hours"
                ].tolist(),
            }
            for iso in iso_list:
                counts_arr = seg["counts"].get(iso)
                if counts_arr is not None:
                    payload[f"counts_{iso}"] = [
                        int(v) for v in np.asarray(counts_arr, dtype=float)
                    ]
            segments_payload.append(payload)

        ts_summary = {
            "centers_s": centers_rel_all.tolist(),
            "widths_s": widths_all.tolist(),
            "segments": segments_payload,
            "ts_counts_raw": {
                iso: [int(v) for v in arr.tolist()]
                for iso, arr in ts_counts_raw_arrays.items()
            },
            "counts_for_json": counts_for_json_lists,
        }
        for iso in iso_list:
            counts_flat = counts_for_json_lists.get(iso, [])
            ts_summary[f"counts_{iso}"] = counts_flat
            ts_summary[f"live_time_{iso}_s"] = widths_all.tolist()
            ts_summary[f"eff_{iso}"] = [iso_params[iso]["eff"]] * len(widths_all)
        base = Path(out_png).with_suffix("")
        json_path = base.with_name(base.name + "_ts.json")
        with open(json_path, "w") as jf:
            json.dump(ts_summary, jf, indent=2)

    return {
        "segments": segments,
        "segments_serialized": segments_payload,
        "ts_counts_raw": ts_counts_raw_arrays,
        "counts_for_json": counts_for_json_lists,
    }


def plot_spectrum(
    energies,
    fit_vals=None,
    out_png="spectrum.png",
    bins=400,
    bin_edges=None,
    config=None,
    *,
    fit_flags=None,
):
    """Plot energy spectrum and optional fit overlay.

    Parameters
    ----------
    energies : array-like
        Energy values in MeV.
    fit_vals : Mapping or FitResult-like, optional
        Fit output used to overlay the model prediction. When provided a
        residual panel is shown beneath the spectrum.
    out_png : str, optional
        Output path (extension used if ``plot_save_formats`` not set).
    bins : int, optional
        Number of bins if ``bin_edges`` is not supplied.
    bin_edges : array-like, optional
        Explicit, strictly increasing bin edges in MeV.  Non-uniform widths
        are supported and override ``bins``.
    config : dict, optional
        Plotting configuration dictionary.
    fit_flags : Mapping, optional
        Flags passed to :func:`fit_spectrum`. These inform the background
        model reconstruction (e.g. ``{"background_model": "loglin_unit"}``).
    """

    energies = np.asarray(energies, dtype=float)

    def _coerce_params(result) -> dict:
        if result is None:
            return {}
        if isinstance(result, Mapping):
            return dict(result)
        for attr in ("best_values", "bestfit", "best_fit", "params"):
            candidate = getattr(result, attr, None)
            if isinstance(candidate, Mapping):
                return dict(candidate)
        return {}

    flags = {}
    if isinstance(fit_flags, Mapping):
        flags = dict(fit_flags)
    elif fit_flags is not None:
        flags = dict(getattr(fit_flags, "__dict__", {}))

    if (
        bin_edges is None
        and config is not None
        and "plot_spectrum_binsize_adc" in config
    ):
        step = float(config["plot_spectrum_binsize_adc"])
        e_min, e_max = energies.min(), energies.max()
        bin_edges = np.arange(e_min, e_max + step, step)

    if bin_edges is not None:
        hist, edges = np.histogram(energies, bins=bin_edges)
    else:
        hist, edges = np.histogram(energies, bins=bins)

    width = np.diff(edges)
    centers = edges[:-1] + width / 2.0

    fit_params = _coerce_params(fit_vals)

    def _build_model_components():
        if not fit_params:
            return OrderedDict(), None

        comps = OrderedDict()
        widths = np.asarray(width, dtype=float)
        if widths.size == 0:
            return comps, None
        centers_arr = np.asarray(centers, dtype=float)
        total = np.zeros_like(widths, dtype=float)

        # Peak contributions
        sigma0 = fit_params.get("sigma0")
        F = fit_params.get("F", 0.0)
        if sigma0 is None:
            sigma0 = fit_params.get("sigma_E", 0.0)
            F = 0.0 if "F" not in fit_params else fit_params.get("F", 0.0)
        sigma0 = float(sigma0 if sigma0 is not None else 0.0)
        F = float(F if F is not None else 0.0)
        sigma_sq = np.clip(sigma0**2 + F * centers_arr, 1e-12, np.inf)
        sigma_vals = np.sqrt(sigma_sq)

        for iso in ("Po210", "Po218", "Po214"):
            mu = fit_params.get(f"mu_{iso}")
            amp = fit_params.get(f"S_{iso}")
            if mu is None or amp is None:
                continue
            tau = fit_params.get(f"tau_{iso}")
            if tau is not None:
                density = emg_left(centers_arr, float(mu), sigma_vals, float(tau))
            else:
                density = gaussian(centers_arr, float(mu), sigma_vals)
            density = np.nan_to_num(density, nan=0.0, posinf=0.0, neginf=0.0)
            comps[iso] = _counts_per_bin(float(amp) * density, widths)
            total += comps[iso]

        # Background contribution
        b0 = fit_params.get("b0")
        b1 = fit_params.get("b1")
        background = None
        if b0 is not None or b1 is not None:
            b0 = float(0.0 if b0 is None else b0)
            b1 = float(0.0 if b1 is None else b1)
            if flags.get("background_model") == "loglin_unit":
                shape = make_linear_bkg(float(edges[0]), float(edges[-1]))
                amplitude = float(fit_params.get("S_bkg", 0.0))
                background_density = amplitude * shape(centers_arr, b0, b1)
            else:
                background_density = b0 + b1 * centers_arr
                if "S_bkg" in fit_params:
                    amplitude = float(fit_params["S_bkg"])
                    norm = b0 * (edges[-1] - edges[0]) + 0.5 * b1 * (
                        edges[-1] ** 2 - edges[0] ** 2
                    )
                    if norm > 0:
                        background_density = background_density * (amplitude / norm)
            background_density = np.nan_to_num(
                background_density, nan=0.0, posinf=0.0, neginf=0.0
            )
            background = _counts_per_bin(background_density, widths)
            total += background

        if background is not None:
            comps["Background"] = background

        if not comps:
            return comps, None

        return comps, total

    model_components, model_total = _build_model_components()
    show_res = model_total is not None

    if show_res:
        fig, (ax_main, ax_res) = plt.subplots(
            2, 1, sharex=True, figsize=(8, 6), gridspec_kw={"height_ratios": [3, 1]}
        )
    else:
        fig, ax_main = plt.subplots(figsize=(8, 6))
        ax_res = None

    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    hist_color = palette.get("hist", "#808080")
    ax_main.bar(centers, hist, width=width, color=hist_color, alpha=0.7, label="Data")

    # If an explicit Po-210 window is provided, focus the x-axis on that region
    win_p210 = None
    if config is not None:
        win_p210 = config.get("window_po210")
    if win_p210 is not None:
        lo, hi = win_p210
        ax_main.set_xlim(lo, hi)

    if model_total is not None:
        component_colors = {
            "Po210": palette.get("Po210", "#2ca02c"),
            "Po218": palette.get("Po218", "#1f77b4"),
            "Po214": palette.get("Po214", "#d62728"),
            "Background": palette.get("background", "#8c564b"),
            "Total": palette.get("fit", "#ff0000"),
        }

        for key in ("Po210", "Po218", "Po214", "Background"):
            if key in model_components:
                ax_main.plot(
                    centers,
                    model_components[key],
                    color=component_colors.get(key, "#000000"),
                    lw=1.5,
                    label=key,
                )

        ax_main.plot(
            centers,
            model_total,
            color=component_colors["Total"],
            lw=2,
            label="Total model",
        )

        if ax_res is not None:
            residuals = hist.astype(float) - model_total
            ax_res.bar(
                centers,
                residuals,
                width=width,
                color=hist_color,
                alpha=0.7,
            )
            ax_res.axhline(0.0, color="#000000", lw=1)
            ax_res.set_ylabel("Residuals [counts]")

    ax_main.set_ylabel("Counts per bin")
    ax_main.set_title("Energy Spectrum")
    if model_total is not None:
        ax_main.legend(fontsize="small")
    if ax_res is not None:
        ax_res.set_xlabel("Energy [MeV]")
    else:
        ax_main.set_xlabel("Energy [MeV]")
    fig.tight_layout()
    targets = get_targets(config, out_png)
    for p in targets.values():
        fig.savefig(p, dpi=300)
    plt.close(fig)

    return ax_main


def _elapsed_hours(times_mpl: np.ndarray) -> np.ndarray:
    epoch = mdates.date2num(datetime(1970, 1, 1))
    seconds = (times_mpl - epoch) * 86400.0
    if seconds.size == 0:
        return seconds
    return (seconds - seconds[0]) / 3600.0


def _apply_time_format(ax, times_mpl: np.ndarray) -> None:
    locator = mdates.AutoDateLocator()
    try:
        formatter = mdates.ConciseDateFormatter(locator)
    except AttributeError:  # pragma: no cover
        formatter = mdates.AutoDateFormatter(locator)
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(formatter)
    ax.xaxis.get_offset_text().set_visible(False)


def _compute_ylim(values: np.ndarray, errors: np.ndarray | None) -> tuple[float, float]:
    if values.size == 0:
        return -0.5, 0.5
    if errors is not None:
        lo_arr = values - errors
        hi_arr = values + errors
    else:
        lo_arr = hi_arr = values

    finite_lo = np.isfinite(lo_arr)
    finite_hi = np.isfinite(hi_arr)

    lower = np.nan
    upper = np.nan
    if finite_lo.any():
        lower = float(np.min(lo_arr[finite_lo]))
    if finite_hi.any():
        upper = float(np.max(hi_arr[finite_hi]))

    if not np.isfinite(lower) or not np.isfinite(upper):
        return -0.5, 0.5
    if lower == upper:
        delta = abs(lower) * 0.1 if lower != 0 else 0.5
        return lower - delta, upper + delta
    pad = 0.1 * (upper - lower)
    return lower - pad, upper + pad


def plot_radon_activity_full(
    times,
    activity,
    errors,
    out_png,
    config=None,
    *,
    po214_activity=None,
    sample_volume_l=None,
    background_mode=None,
):
    """Plot radon activity versus time with uncertainties."""

    times_mpl = guard_mpl_times(times=times)
    elapsed_hours = _elapsed_hours(times_mpl)

    activity = np.asarray(activity, dtype=float)
    errors_arr = None if errors is None else np.asarray(errors, dtype=float)

    volume = None
    if sample_volume_l is not None:
        try:
            volume = float(sample_volume_l)
        except (TypeError, ValueError):
            volume = None
    label_units = "Rn-222 Activity (Bq)"
    if volume is not None and np.isfinite(volume) and volume > 0:
        conc = activity / volume
        conc_err = None if errors_arr is None else errors_arr / volume
        label_units = "Rn-222 Concentration (Bq/L)"
    else:
        conc = activity
        conc_err = errors_arr

    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color = palette.get("radon_activity", "#9467bd")

    label = None if po214_activity is None else "Rn-222"
    ax_abs.errorbar(
        times_mpl,
        conc,
        yerr=conc_err,
        **_errorbar_kwargs(color, label=label),
    )
    _apply_time_format(ax_abs, times_mpl)
    ax_abs.set_xlabel("Time (UTC)")
    ax_abs.set_ylabel(label_units)
    ax_abs.set_title("Radon Concentration vs. Time")
    ax_abs.ticklabel_format(axis="y", style="plain")

    ax_rel.errorbar(
        elapsed_hours,
        conc,
        yerr=conc_err,
        **_errorbar_kwargs(color),
    )
    ax_rel.set_xlabel("Elapsed Time (h)")
    ax_rel.set_ylabel(label_units)
    ax_rel.set_title("Radon Concentration vs. Elapsed Hours")
    ax_rel.ticklabel_format(axis="y", style="plain")
    ax_rel.xaxis.get_offset_text().set_visible(False)

    if po214_activity is not None:
        po214_arr = np.asarray(po214_activity, dtype=float)
        if volume is not None and np.isfinite(volume) and volume > 0:
            po214_arr = po214_arr / volume
            po214_label = "Po-214 Concentration (QC)"
        else:
            po214_label = "Po-214 Activity (QC)"
        color214 = palette.get("Po214", "#d62728")
        ax2 = ax_abs.twinx()
        ax2.plot(times_mpl, po214_arr, "--", color=color214, label=po214_label)
        ax2.set_ylabel(po214_label)
        ax2.ticklabel_format(axis="y", style="plain")
        ax2.yaxis.get_offset_text().set_visible(False)
        lines1, labels1 = ax_abs.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax_abs.legend(lines1 + lines2, labels1 + labels2, loc="best")

    ylim_low, ylim_high = _compute_ylim(np.asarray(conc, dtype=float), conc_err)
    ax_abs.set_ylim(ylim_low, ylim_high)
    ax_rel.set_ylim(ylim_low, ylim_high)

    ax_abs.yaxis.get_offset_text().set_visible(False)
    ax_rel.yaxis.get_offset_text().set_visible(False)
    fig.autofmt_xdate()
    layout_rect = None
    if background_mode:
        subtitle = f"Background mode: {background_mode}"
        fig.suptitle(subtitle, fontsize=10)
        layout_rect = (0.0, 0.0, 1.0, 0.94)

    if layout_rect is not None:
        plt.tight_layout(rect=layout_rect)
    else:
        plt.tight_layout()
    targets = get_targets(config, out_png)
    for p in targets.values():
        fig.savefig(p, dpi=300)
    plt.close(fig)


def plot_total_radon_full(
    times,
    total_bq,
    errors,
    out_png,
    config=None,
    *,
    background_mode=None,
):
    """Plot total radon present in the sample versus time."""

    times_mpl = guard_mpl_times(times=times)
    elapsed_hours = _elapsed_hours(times_mpl)
    total_bq = np.asarray(total_bq, dtype=float)
    errors_arr = None if errors is None else np.asarray(errors, dtype=float)

    fig, (ax_abs, ax_rel) = plt.subplots(1, 2, figsize=(12, 4), sharey=True)
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color = palette.get("total_radon", palette.get("radon_activity", "#9467bd"))

    ax_abs.errorbar(
        times_mpl,
        total_bq,
        yerr=errors_arr,
        **_errorbar_kwargs(color),
    )
    _apply_time_format(ax_abs, times_mpl)
    ax_abs.set_xlabel("Time (UTC)")
    ax_abs.set_ylabel("Total Radon in Sample (Bq)")
    ax_abs.set_title("Total Radon vs. Time")
    ax_abs.ticklabel_format(axis="y", style="plain")

    ax_rel.errorbar(
        elapsed_hours,
        total_bq,
        yerr=errors_arr,
        **_errorbar_kwargs(color),
    )
    ax_rel.set_xlabel("Elapsed Time (h)")
    ax_rel.set_ylabel("Total Radon in Sample (Bq)")
    ax_rel.set_title("Total Radon vs. Elapsed Hours")
    ax_rel.ticklabel_format(axis="y", style="plain")
    ax_rel.xaxis.get_offset_text().set_visible(False)

    ylim_low, ylim_high = _compute_ylim(total_bq, errors_arr)
    ax_abs.set_ylim(ylim_low, ylim_high)
    ax_rel.set_ylim(ylim_low, ylim_high)

    ax_abs.yaxis.get_offset_text().set_visible(False)
    ax_rel.yaxis.get_offset_text().set_visible(False)
    fig.autofmt_xdate()
    layout_rect = None
    if background_mode:
        fig.suptitle(f"Background mode: {background_mode}", fontsize=10)
        layout_rect = (0.0, 0.0, 1.0, 0.94)

    if layout_rect is not None:
        plt.tight_layout(rect=layout_rect)
    else:
        plt.tight_layout()

    targets = get_targets(config, out_png)
    for p in targets.values():
        fig.savefig(p, dpi=300)
    plt.close(fig)


def plot_equivalent_air(times, volumes, errors, conc, out_png, config=None):
    """Plot equivalent air volume versus time.

    Parameters
    ----------
    conc : float or str or None
        Ambient concentration label to include in the plot title. When ``None``
        the concentration is omitted from the title.
    """
    times_mpl = guard_mpl_times(times=times)
    volumes = np.asarray(volumes, dtype=float)
    errors = np.asarray(errors, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color = palette.get("equivalent_air", "#2ca02c")
    ax.errorbar(
        times_mpl,
        volumes,
        yerr=errors,
        **_errorbar_kwargs(color),
    )
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Equivalent Air Volume")
    if conc is None:
        title = "Equivalent Air Volume vs. Time"
    else:
        title = f"Equivalent Air Volume vs. Time (ambient {conc} Bq/L)"
    ax.set_title(title)

    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    plt.tight_layout()
    targets = get_targets(config, out_png)
    for p in targets.values():
        fig.savefig(p, dpi=300)
    plt.close(fig)


def plot_modeled_radon_activity(
    times,
    E,
    dE,
    N0,
    dN0,
    out_png,
    config=None,
    *,
    overlay_po214=False,
    fit_valid=True,
):
    """Compute and plot modeled Rn-222 activity over time.

    Parameters
    ----------
    times : array-like
        Relative times in seconds.
    E, dE, N0, dN0 : float
        Fitted Po-214 parameters which are converted to Rn-222 activity.
    overlay_po214 : bool, optional
        When ``True`` overlay the Po-214 activity for QC on a secondary axis.
    """
    if not fit_valid:
        return

    from radon_activity import radon_activity_curve

    # The fitted Po-214 parameters are already decay rates in becquerels. The
    # radon curve should therefore use them directly without any additional
    # scaling factors.  Doing so preserves the physical units of the
    # steady-state and initial activities when evaluated with the Rn-222
    # half-life.
    activity, sigma = radon_activity_curve(times, E, dE, N0, dN0, RN222.half_life_s)

    po214_activity = None
    if overlay_po214:
        po214_activity, _ = radon_activity_curve(
            times, E, dE, N0, dN0, PO214_HALF_LIFE_S
        )

    plot_radon_activity_full(
        times,
        activity,
        sigma,
        out_png,
        config=config,
        po214_activity=po214_activity,
    )


def plot_radon_trend_full(times, activity, out_png, config=None, *, fit_valid=True):
    """Plot modeled radon activity trend without uncertainties."""
    if not fit_valid:
        return
    times_mpl = guard_mpl_times(times=times)
    activity = np.asarray(activity, dtype=float)

    fig, ax = plt.subplots(figsize=(8, 4))
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color = palette.get("radon_activity", "#9467bd")
    ax.plot(times_mpl, activity, "o-", color=color)
    ax.set_xlabel("Time (UTC)")
    ax.set_ylabel("Radon Concentration (Bq/L)")
    ax.set_title("Radon Concentration Trend")

    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    plt.tight_layout()

    targets = get_targets(config, out_png)
    for p in targets.values():
        fig.savefig(p, dpi=300)
    plt.close(fig)


def plot_radon_activity(ts_dict, outdir):
    """Simple wrapper to plot radon activity time series."""
    outdir = Path(outdir)
    sample_vol = ts_dict.get("sample_volume_l")
    plot_radon_activity_full(
        ts_dict["time"],
        ts_dict["activity"],
        ts_dict.get("error"),
        outdir / "radon_activity.png",
        sample_volume_l=sample_vol,
        background_mode=ts_dict.get("background_mode"),
    )


def plot_total_radon(ts_dict, outdir):
    """Simple wrapper to plot total radon present in the sample."""

    outdir = Path(outdir)
    plot_total_radon_full(
        ts_dict["time"],
        ts_dict["activity"],
        ts_dict.get("error"),
        outdir / "total_radon.png",
        background_mode=ts_dict.get("background_mode"),
    )


def plot_radon_trend(ts_dict, outdir):
    """Simple wrapper to plot a radon activity trend."""
    outdir = Path(outdir)
    times_mpl = guard_mpl_times(times=ts_dict["time"])
    y = np.asarray(ts_dict["activity"], dtype=float)
    coeff = np.polyfit(times_mpl, y, 1)

    fig, ax = plt.subplots()
    ax.plot(times_mpl, y, "o")
    ax.plot(times_mpl, np.polyval(coeff, times_mpl))
    ax.set_ylabel("Radon activity [Bq]")
    ax.set_xlabel("Time (UTC)")
    ax.ticklabel_format(axis="y", style="plain")
    setup_time_axis(ax, times_mpl)
    fig.autofmt_xdate()
    ax.yaxis.get_offset_text().set_visible(False)
    plt.tight_layout()
    fig.savefig(outdir / "radon_trend.png", dpi=300)
    plt.close(fig)


def plot_spectrum_comparison(
    pre_energies,
    post_energies,
    *,
    bins=400,
    bin_edges=None,
    out_png="spectrum_pre_post.png",
    config=None,
):
    """Overlay spectra before and after filtering and return ROI differences.

    When ``bin_edges`` is not supplied a fixed ``0 – 1 MeV`` range is used
    for the histogram binning.  Using deterministic bin edges avoids
    differences between runs that could arise from data-dependent bin
    calculations and makes generated plots reproducible.
    """

    pre = np.asarray(pre_energies, dtype=float)
    post = np.asarray(post_energies, dtype=float)
    if bin_edges is None:
        # Use a fixed binning scheme for reproducibility rather than deriving
        # edges from the data distribution which could vary run-to-run.
        bin_edges = np.linspace(0.0, 1.0, int(bins) + 1)

    hist_pre, edges = np.histogram(pre, bins=bin_edges)
    hist_post, _ = np.histogram(post, bins=edges)

    width = np.diff(edges)
    centers = edges[:-1] + width / 2.0

    fig, ax = plt.subplots(figsize=(8, 6))
    palette_name = str(config.get("palette", "default")) if config else "default"
    palette = COLOR_SCHEMES.get(palette_name, COLOR_SCHEMES["default"])
    color_pre = palette.get("hist", "#808080")
    color_post = palette.get("fit", "#ff0000")
    ax.bar(centers, hist_pre, width=width, color=color_pre, alpha=0.5, label="Pre")
    ax.bar(centers, hist_post, width=width, color=color_post, alpha=0.5, label="Post")
    ax.set_xlabel("Energy (MeV)")
    ax.set_ylabel("Counts per bin")
    ax.legend()
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

    roi_diff = {}
    if config is not None:
        for iso in ("Po210", "Po218", "Po214"):
            win = config.get(f"window_{iso.lower()}")
            if win is None:
                continue
            c_pre = int(((pre >= win[0]) & (pre <= win[1])).sum())
            c_post = int(((post >= win[0]) & (post <= win[1])).sum())
            roi_diff[iso] = c_post - c_pre

    return roi_diff


def plot_activity_grid(result_map, out_png="burst_scan.png", config=None):
    """Visualise radon activity on a parameter grid."""

    if not result_map:
        return

    mults = sorted({m for m, _ in result_map})
    wins = sorted({w for _, w in result_map})
    grid = np.empty((len(mults), len(wins)))
    for i, m in enumerate(mults):
        for j, w in enumerate(wins):
            grid[i, j] = result_map.get((m, w), np.nan)

    fig, ax = plt.subplots(figsize=(6, 5))
    im = ax.imshow(
        grid,
        origin="lower",
        aspect="auto",
        extent=[min(wins), max(wins), min(mults), max(mults)],
    )
    ax.set_xlabel("burst_window_size_s")
    ax.set_ylabel("burst_multiplier")
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label("Radon Activity (Bq)")
    plt.tight_layout()
    Path(out_png).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()


# -----------------------------------------------------
# End of plot_utils.py
# -----------------------------------------------------
