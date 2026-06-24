from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from analyses.load_results import load_theta_y_file
from analyses.common import COLOR_GREEN, COLOR_ORANGE


def plot_theta_y_distribution(
    results_dir: Path,
    exp_ids: Sequence[int],
    inner_circle: float = 300.0,
    thickness_circle: float = 0.0,
    max_distance: float = 800.0,
    bins: int = 10,
    figsize=(1.75, 2.25),
    dpi: int = 600,
    pol_axis: Optional[int] = None,   # None = legacy ThetaY; 0 = x-axis; 1 = y-axis
    save_path: Optional[Path] = None,
):
    """
    Distribution of the swimming angle with respect to a reference axis.

    pol_axis=None  → legacy ThetaY_exp{id}.txt  (theta_y = arccos(Vy/|V|), y image-down)
    pol_axis=0     → ThetaY_axis0_exp{id}.txt   (theta_x = arccos(Vx/|V|), horizontal pol)
    pol_axis=1     → ThetaY_axis1_exp{id}.txt   (theta_y = arccos(Vy/|V|), vertical pol)

    theta = 0   → swimming along +axis
    theta = π/2 → swimming perpendicular to axis
    theta = π   → swimming along -axis

    Two panels: particles inside the disc (green) and outside (orange).
    """
    theta_list: list[np.ndarray] = []
    dist_list: list[np.ndarray] = []

    for exp_id in exp_ids:
        if pol_axis is None:
            path = results_dir / f"ThetaY_exp{exp_id}.txt"
        else:
            path = results_dir / f"ThetaY_axis{pol_axis}_exp{exp_id}.txt"
        if not path.exists():
            continue
        df = load_theta_y_file(path)
        if df.empty:
            continue
        theta = df["theta_y"].to_numpy(dtype=float)
        dist = df["dist"].to_numpy(dtype=float)
        mask = np.isfinite(theta) & np.isfinite(dist)
        if mask.any():
            theta_list.append(theta[mask])
            dist_list.append(dist[mask])

    if not theta_list:
        suffix = f"axis{pol_axis}" if pol_axis is not None else "legacy"
        raise FileNotFoundError(
            f"No ThetaY result files ({suffix}) found for experiments {list(exp_ids)} "
            f"in {results_dir}."
        )

    theta_all = np.concatenate(theta_list)
    dist_all = np.concatenate(dist_list)

    inside = theta_all[dist_all <= inner_circle]
    outside = theta_all[
        (dist_all > inner_circle + thickness_circle) & (dist_all < max_distance)
    ]

    fig, axs = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

    if inside.size:
        axs[0].hist(inside, color=COLOR_GREEN, alpha=0.5, bins=bins, density=True)
    else:
        axs[0].text(0.1, 0.5, "No inside data", transform=axs[0].transAxes)

    if outside.size:
        axs[1].hist(outside, color=COLOR_ORANGE, alpha=0.5, bins=bins, density=True)
    else:
        axs[1].text(0.1, 0.5, "No outside data", transform=axs[1].transAxes)

    axis_label = {None: r"$\theta_y$ (legacy)", 0: r"$\theta_x$", 1: r"$\theta_y$"}
    for ax in axs:
        ax.set_xticks([0, np.pi / 2, np.pi], ['$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_yticks([0.2, 0.3, 0.4])

    axs[1].set_xlabel(axis_label.get(pol_axis, r"$\theta$"))
    fig.supylabel(r'$P(\theta)$', va='center')
    axs[0].set_xlabel('')
    axs[0].set_ylim([0.25, 0.4])
    axs[1].set_ylim([0.25, 0.4])

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='pdf', transparent=True)

    return fig, axs
