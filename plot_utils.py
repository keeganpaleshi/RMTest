# -----------------------------------------------------
# plot_utils.py
# -----------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

# Half-life constants used for the time-series overlay [seconds]
PO214_HALF_LIFE_S = 1.64e-4  # 164 µs
PO218_HALF_LIFE_S = 183.0    # ~3.05 minutes

__all__ = ["plot_time_series", "plot_spectrum"]


def plot_time_series(
    all_timestamps, all_energies, fit_results, t_start, t_end, config, out_png
):
    """
    all_timestamps: 1D np.ndarray of absolute UNIX times (s)
    all_energies:   1D np.ndarray of energies (MeV)
    fit_results:    dict from fit_time_series(...) or fit_decay(...)
    t_start, t_end: floats (absolute UNIX times) for the fit window
    config:         JSON dict
    out_png:        output path for the PNG file
    """

    if fit_results is None:
        fit_results = {}

    iso_params = {
        "Po214": {
            "window": config.get("window_Po214"),
            "eff": float(config.get("eff_Po214", [1.0])[0]),
            "half_life": PO214_HALF_LIFE_S,
        },
        "Po218": {
            "window": config.get("window_Po218"),
            "eff": float(config.get("eff_Po218", [1.0])[0]),
            "half_life": PO218_HALF_LIFE_S,
        },
    }
    iso_list = [iso for iso, p in iso_params.items() if p["window"] is not None]
    # Time since t_start:
    times_rel = all_timestamps - t_start

    # 1) Choose binning:
    # Newer config files may store plot options under keys prefixed with
    # "plot_".  Fall back to the old names for backwards compatibility.
    bin_mode = config.get(
        "plot_time_binning_mode",
        config.get("time_bin_mode", "fixed"),
    ).lower()
    if bin_mode == "FD":
        # Freedman Diaconis rule on the entire time range:
        data = times_rel[(times_rel >= 0) & (times_rel <= (t_end - t_start))]
        if len(data) < 2:
            n_bins = 1
        else:
            q25, q75 = np.percentile(data, [25, 75])
            iqr = q75 - q25
            if iqr <= 0:
                n_bins = int(config.get("time_bins_fallback", 1))
            else:
                bin_width = 2 * iqr / (len(data) ** (1.0 / 3.0))
                n_bins = max(
                    1, int(np.ceil((data.max() - data.min()) / bin_width)))
    else:
        # fixed width in seconds (e.g. 3600 s):
        dt = float(
            config.get(
                "plot_time_bin_width_s",
                config.get("time_bin_s", 3600.0),
            )
        )
        n_bins = int(np.ceil((t_end - t_start) / dt))
        if n_bins < 1:
            n_bins = 1

    # Now build histogram bins:
    edges = np.linspace(0, (t_end - t_start), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = np.diff(edges)

    # Select drawing style for the data histogram
    time_style = str(config.get("plot_time_style", "hist")).lower()

    # 2) Plot each isotope s histogram + overlay the model:
    plt.figure(figsize=(8, 6))
    colors = {"Po214": "tab:red", "Po218": "tab:blue"}
    legend_entries = []

    for iso in iso_list:
        emin, emax = iso_params[iso]["window"]
        mask_iso = (
            (all_energies >= emin)
            & (all_energies <= emax)
            & (all_timestamps >= t_start)
            & (all_timestamps <= t_end)
        )
        t_iso_rel = times_rel[mask_iso]

        # Histogram of observed counts:
        counts_iso, _ = np.histogram(t_iso_rel, bins=edges)
        if time_style == "lines":
            plt.plot(
                centers,
                counts_iso,
                color=colors[iso],
                label=f"Data {iso}",
            )
        else:
            plt.step(
                centers,
                counts_iso,
                where="mid",
                color=colors[iso],
                label=f"Data {iso}",
            )

        # Overlay the continuous model curve (scaled to counts/bin):
        lam = np.log(2.0) / iso_params[iso]["half_life"]
        eff = iso_params[iso]["eff"]

        E_iso = fit_results.get(f"E_{iso}", fit_results.get("E", 0.0))
        B_iso = fit_results.get(f"B_{iso}", fit_results.get("B", 0.0))
        N0_iso = fit_results.get(f"N0_{iso}", fit_results.get("N0", 0.0))

        # r_iso(t_rel) = eff * [E*(1 - exp(-lam*t_rel)) + lam*N0*exp(-lam*t_rel)] + B
        r_rel = (
            eff
            * (
                E_iso * (1.0 - np.exp(-lam * centers))
                + lam * N0_iso * np.exp(-lam * centers)
            )
            + B_iso
        )

        # Convert  rate (counts/s)     expected counts per bin = r_rel * bin_width
        model_counts = r_rel * bin_widths
        plt.plot(
            centers,
            model_counts,
            color=colors[iso],
            lw=2,
            ls="--",
            label=f"Model {iso}",
        )

    plt.xlabel("Time since t_start (s)")
    plt.ylabel("Counts per bin")
    plt.title("Po-214 & Po-218 Time Series Fit")
    plt.legend(fontsize="small")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    # Determine which formats to save. If not specified, fall back to the
    # extension of the provided output path.
    fmt_default = os.path.splitext(out_png)[1].lstrip(".") or "png"
    save_fmts = config.get("plot_save_formats", [fmt_default])
    if isinstance(save_fmts, str):
        save_fmts = [save_fmts]

    base = os.path.splitext(out_png)[0]
    for fmt in save_fmts:
        out_file = base + f".{fmt}"
        plt.savefig(out_file, dpi=300)
    plt.close()

    # (Optionally) also write a small JSON of the binned values:
    if config.get("dump_time_series_json", False):
        import json

        ts_summary = {"centers_s": centers.tolist()}
        for iso in iso_list:
            emin, emax = iso_params[iso]["window"]
            ts_summary[f"counts_{iso}"] = np.histogram(
                times_rel[
                    (all_energies >= emin)
                    & (all_energies <= emax)
                    & (all_timestamps >= t_start)
                    & (all_timestamps <= t_end)
                ],
                bins=edges,
            )[0].tolist()
        with open(out_png.replace(".png", "_ts.json"), "w") as jf:
            json.dump(ts_summary, jf, indent=2)


def plot_spectrum(
    energies,
    fit_vals=None,
    out_png="spectrum.png",
    bins=400,
    bin_edges=None,
    config=None,
):
    """Plot energy spectrum and optional fit overlay."""
    if bin_edges is None and config is not None and "plot_spectrum_binsize_adc" in config:
        step = float(config["plot_spectrum_binsize_adc"])
        e_min, e_max = energies.min(), energies.max()
        bin_edges = np.arange(e_min, e_max + step, step)

    if bin_edges is not None:
        hist, edges = np.histogram(energies, bins=bin_edges)
    else:
        hist, edges = np.histogram(energies, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    width = edges[1] - edges[0]

    plt.figure(figsize=(8, 6))
    plt.bar(centers, hist, width=width, color="gray", alpha=0.7, label="Data")

    if fit_vals:
        x = np.linspace(edges[0], edges[-1], 1000)
        sigma_E = fit_vals.get("sigma_E", 1.0)
        y = fit_vals.get("b0", 0.0) + fit_vals.get("b1", 0.0) * x
        for pk in ("Po210", "Po218", "Po214"):
            mu_key = f"mu_{pk}"
            amp_key = f"S_{pk}"
            if mu_key in fit_vals and amp_key in fit_vals:
                mu = fit_vals[mu_key]
                amp = fit_vals[amp_key]
                y += amp / (sigma_E * np.sqrt(2 * np.pi)) * np.exp(-0.5 * ((x - mu) / sigma_E) ** 2)
        plt.plot(x, y * width, color="red", lw=2, label="Fit")

    plt.xlabel("Energy (MeV)")
    plt.ylabel("Counts per bin")
    plt.title("Energy Spectrum")
    if fit_vals:
        plt.legend(fontsize="small")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)

    fmt_default = os.path.splitext(out_png)[1].lstrip(".") or "png"
    save_fmts = []
    if config is not None:
        save_fmts = config.get("plot_save_formats", [])
    if not save_fmts:
        save_fmts = [fmt_default]
    if isinstance(save_fmts, str):
        save_fmts = [save_fmts]

    base = os.path.splitext(out_png)[0]
    for fmt in save_fmts:
        plt.savefig(base + f".{fmt}", dpi=300)
    plt.close()


# -----------------------------------------------------
# End of plot_utils.py
# -----------------------------------------------------
