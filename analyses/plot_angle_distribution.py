from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _load_angle_arrays_for_experiments(
    results_dir: Path,
    exp_ids: Sequence[int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    angle_list = []
    dist_list = []
    vel_list = []

    for exp_id in exp_ids:
        path = results_dir / f"angle_exp{exp_id}.txt"
        if not path.exists():
            continue

        data = np.genfromtxt(path, skip_header=1)
        if isinstance(data, float) and np.isnan(data):
            continue
        if getattr(data, "size", 0) == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        angle = data[:, 0].astype(float)
        dist = data[:, 1].astype(float)
        vel = data[:, 2].astype(float)

        mask = np.isfinite(angle) & np.isfinite(dist) & np.isfinite(vel)
        if np.any(mask):
            angle_list.append(angle[mask])
            dist_list.append(dist[mask])
            vel_list.append(vel[mask])

    return angle_list, dist_list, vel_list


def plot_angle_distribution(
    results_dir: Path,
    exp_ids: Sequence[int],
    bins: int = 30,
    figsize=(4.0, 3.0),
    dpi: int = 300,
    save_path: Optional[Path] = None,
):
    angle_list, _, _ = _load_angle_arrays_for_experiments(results_dir, exp_ids)

    if not angle_list:
        raise FileNotFoundError("No angle_exp*.txt files found for selected experiments.")

    angles = np.concatenate(angle_list)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.hist(angles, bins=bins, density=True, alpha=0.7)

    ax.set_xlabel("angle [rad]")
    ax.set_ylabel("probability density")
    ax.set_title("Angle distribution")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", transparent=True)

    return fig, ax