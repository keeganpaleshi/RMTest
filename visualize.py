import os
import numpy as np
import matplotlib.pyplot as plt

__all__ = ["cov_heatmap", "efficiency_bar"]


def cov_heatmap(cov_matrix, out_png, labels=None):
    """Plot a correlation heatmap from a covariance matrix."""
    cov = np.asarray(cov_matrix, dtype=float)
    if cov.ndim != 2 or cov.shape[0] != cov.shape[1]:
        raise ValueError("cov_matrix must be square")
    n = cov.shape[0]
    std = np.sqrt(np.diag(cov))
    with np.errstate(divide="ignore", invalid="ignore"):
        corr = cov / np.outer(std, std)
    corr = np.nan_to_num(corr)
    if labels is None:
        labels = [str(i) for i in range(n)]
    plt.figure(figsize=(4 + 0.5 * n, 3 + 0.5 * n))
    im = plt.imshow(corr, vmin=-1, vmax=1, cmap="viridis")
    plt.colorbar(im, label="Correlation")
    plt.xticks(range(n), labels, rotation=45, ha="right")
    plt.yticks(range(n), labels)
    for i in range(n):
        for j in range(n):
            plt.text(j, i, f"{corr[i,j]:.2f}", ha="center", va="center", color="white" if abs(corr[i,j])>0.5 else "black", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    return corr


def efficiency_bar(eff_dict, out_png):
    """Bar chart of efficiency sources annotated with BLUE weights."""
    sources = eff_dict.get("sources", {})
    names = list(sources.keys())
    values = [sources[n].get("value", 0.0) for n in names]
    errors = [sources[n].get("error", 0.0) for n in names]
    weights = eff_dict.get("combined", {}).get("weights")

    if "combined" in eff_dict:
        names.append("BLUE")
        values.append(eff_dict["combined"].get("value", 0.0))
        errors.append(eff_dict["combined"].get("error", 0.0))

    x = np.arange(len(names))
    plt.figure(figsize=(6, 4))
    plt.bar(x, values, yerr=errors, capsize=4, color="tab:blue", alpha=0.7)
    plt.xticks(x, names, rotation=45, ha="right")
    plt.ylabel("Efficiency")
    plt.title("Efficiency Estimates")

    if weights is not None:
        for i, w in enumerate(weights):
            plt.text(i, values[i] + errors[i], f"{w:.2f}", ha="center", va="bottom", fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    plt.savefig(out_png, dpi=300)
    plt.close()
    return None
