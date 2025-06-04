# -----------------------------------------------------
# plot_utils.py
# -----------------------------------------------------

import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["plot_spectrum", "plot_time_series"]


def plot_spectrum(energies, fit_vals=None, out_png="spectrum.png", bins=100, bin_edges=None):
    """Plot energy spectrum and optional fit overlay.

    Parameters
    ----------
    energies : array-like
        1D array of energies (MeV).
    fit_vals : dict or None
        Dictionary of fit parameters from ``fit_spectrum``.  The function
        only uses a subset of expected keys (``mu_*``, ``S_*``, ``sigma_E``,
        ``b0`` and ``b1``) if present.  The overlay is skipped when no fit
        values are provided.
    out_png : str
        Output file for the PNG image.
    bins : int
        Number of histogram bins if ``bin_edges`` is not given.
    bin_edges : array-like or None
        Optional array of bin edges to use for the histogram.
    """

    edges = bin_edges if bin_edges is not None else bins
    counts, edges = np.histogram(energies, bins=edges)
    centers = 0.5 * (edges[:-1] + edges[1:])
    widths = np.diff(edges)

    plt.figure(figsize=(8, 5))
    plt.bar(centers, counts, width=widths, color="tab:gray", alpha=0.6, label="Data")

    if isinstance(fit_vals, dict) and fit_vals:
        sigma = fit_vals.get("sigma_E", None)
        model_x = np.linspace(edges[0], edges[-1], 1000)
        model_y = np.zeros_like(model_x)

        if sigma and sigma > 0:
            for iso in ("Po210", "Po218", "Po214"):
                mu_k = f"mu_{iso}"
                amp_k = f"S_{iso}"
                if mu_k in fit_vals and amp_k in fit_vals:
                    mu = fit_vals[mu_k]
                    amp = fit_vals[amp_k]
                    gauss = (
                        amp
                        * np.exp(-(model_x - mu) ** 2 / (2 * sigma ** 2))
                        / (sigma * np.sqrt(2 * np.pi))
                    )
                    model_y += gauss

        b0 = fit_vals.get("b0", 0.0)
        b1 = fit_vals.get("b1", 0.0)
        model_y += b0 + b1 * model_x

        plt.plot(model_x, model_y * np.mean(widths), color="tab:red", lw=2, label="Fit")

    plt.xlabel("Energy (MeV)")
    plt.ylabel("Counts per bin")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    return centers, counts


def plot_time_series(
    all_timestamps, all_energies, fit_results, t_start, t_end, config, out_png
):
    """
    all_timestamps: 1D np.ndarray of absolute UNIX times (s)
    all_energies:   1D np.ndarray of energies (MeV)
    fit_results:    dict from fit_time_series(...)
    t_start, t_end: floats (absolute UNIX times) for the fit window
    config:         JSON dict
    out_png:        output path for the PNG file
    """

    isotopes_cfg = config["isotopes"]
    iso_list = list(isotopes_cfg.keys())
    # Time since t_start:
    times_rel = all_timestamps - t_start

    # 1) Choose binning:
    bin_mode = config.get("time_bin_mode", "fixed")  # "fixed" or "FD"
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
        dt = float(config.get("time_bin_s", 3600.0))
        n_bins = int(np.ceil((t_end - t_start) / dt))
        if n_bins < 1:
            n_bins = 1

    # Now build histogram bins:
    edges = np.linspace(0, (t_end - t_start), n_bins + 1)
    centers = 0.5 * (edges[:-1] + edges[1:])
    bin_widths = np.diff(edges)

    # 2) Plot each isotope s histogram + overlay the model:
    plt.figure(figsize=(8, 6))
    colors = {"Po214": "tab:red", "Po218": "tab:blue"}
    legend_entries = []

    for iso in iso_list:
        emin, emax = isotopes_cfg[iso]["energy_window"]
        mask_iso = (
            (all_energies >= emin)
            & (all_energies <= emax)
            & (all_timestamps >= t_start)
            & (all_timestamps <= t_end)
        )
        t_iso_rel = times_rel[mask_iso]

        # Histogram of observed counts:
        counts_iso, _ = np.histogram(t_iso_rel, bins=edges)
        plt.bar(
            centers,
            counts_iso,
            width=bin_widths,
            color=colors[iso],
            alpha=0.5,
            label=f"Data {iso}",
        )

        # Overlay the continuous model curve (scaled to counts/bin):
        lam = np.log(2.0) / isotopes_cfg[iso]["half_life_s"]
        eff = isotopes_cfg[iso].get("efficiency", 1.0)

        E_iso = fit_results.get(f"E_{iso}", 0.0)
        B_iso = fit_results.get(f"B_{iso}", 0.0)
        N0_iso = fit_results.get(f"N0_{iso}", 0.0)

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
        plt.plot(centers, model_counts,
                 color=colors[iso], lw=2, label=f"Model {iso}")

    plt.xlabel("Time since t_start (s)")
    plt.ylabel("Counts per bin")
    plt.title("Po-214 & Po-218 Time Series Fit")
    plt.legend(fontsize="small")
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()

    # (Optionally) also write a small JSON of the binned values:
    if config.get("dump_time_series_json", False):
        import json

        ts_summary = {
            "centers_s": centers.tolist(),
            "counts_Po214": np.histogram(
                times_rel[
                    (all_energies >= isotopes_cfg["Po214"]["energy_window"][0])
                    & (all_energies <= isotopes_cfg["Po214"]["energy_window"][1])
                    & (all_timestamps >= t_start)
                    & (all_timestamps <= t_end)
                ],
                bins=edges,
            )[0].tolist(),
            "counts_Po218": np.histogram(
                times_rel[
                    (all_energies >= isotopes_cfg["Po218"]["energy_window"][0])
                    & (all_energies <= isotopes_cfg["Po218"]["energy_window"][1])
                    & (all_timestamps >= t_start)
                    & (all_timestamps <= t_end)
                ],
                bins=edges,
            )[0].tolist(),
        }
        with open(out_png.replace(".png", "_ts.json"), "w") as jf:
            json.dump(ts_summary, jf, indent=2)


# -----------------------------------------------------
# End of plot_utils.py
# -----------------------------------------------------
