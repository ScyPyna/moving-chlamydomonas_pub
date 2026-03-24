from __future__ import annotations

from pathlib import Path
from typing import List
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def plot_theta_hist(
    theta_csvs: List[Path],
    out_pdf: Path,
    inner_circle: float = 300.0,
    max_distance: float = 800.0,
    bins: int = 10,
    figsize=(2.2, 2.8),
    dpi: int = 400,
) -> None:
    dfs = []
    for p in theta_csvs:
        if p.exists():
            dfs.append(pd.read_csv(p))

    if not dfs:
        raise FileNotFoundError("No theta.csv files found for plotting.")

    df = pd.concat(dfs, ignore_index=True)
    if df.empty:
        raise ValueError("Theta dataframe is empty: nothing to plot.")

    df = df.replace([np.inf, -np.inf], np.nan).dropna(subset=["theta_rad", "dist_um"])

    inside = df.loc[df["dist_um"] <= inner_circle, "theta_rad"].to_numpy(dtype=float)
    outside = df.loc[(df["dist_um"] > inner_circle) & (df["dist_um"] < max_distance), "theta_rad"].to_numpy(dtype=float)

    out_pdf.parent.mkdir(parents=True, exist_ok=True)

    fig, axs = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

    if inside.size:
        axs[0].hist(inside, bins=bins, density=True)
    else:
        axs[0].text(0.1, 0.5, "No data", transform=axs[0].transAxes)

    if outside.size:
        axs[1].hist(outside, bins=bins, density=True)
    else:
        axs[1].text(0.1, 0.5, "No data", transform=axs[1].transAxes)

    for ax in axs:
        ax.set_xticks([0, np.pi/2, np.pi])
        ax.set_xticklabels(["0", r"$\pi/2$", r"$\pi$"])
        ax.set_ylabel(r"$P(\theta)$")

    axs[1].set_xlabel(r"$\theta$")
    axs[0].set_title("inside (r ≤ {0:.0f} µm)".format(inner_circle))
    axs[1].set_title("outside ({0:.0f} < r < {1:.0f} µm)".format(inner_circle, max_distance))

    fig.tight_layout()
    fig.savefig(out_pdf, transparent=True)
    plt.close(fig)