from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt


def _load_omega_arrays_for_experiments(
    results_dir: Path,
    exp_ids: Sequence[int],
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    omega_list = []
    dist_list = []
    time_list = []

    for exp_id in exp_ids:
        path = results_dir / f"omega_exp{exp_id}.txt"
        if not path.exists():
            continue

        data = np.genfromtxt(path, skip_header=1)
        if isinstance(data, float) and np.isnan(data):
            continue
        if getattr(data, "size", 0) == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        omega = data[:, 0].astype(float)
        dist = data[:, 1].astype(float)
        time = data[:, 2].astype(float)

        mask = np.isfinite(omega) & np.isfinite(dist) & np.isfinite(time)
        if np.any(mask):
            omega_list.append(omega[mask])
            dist_list.append(dist[mask])
            time_list.append(time[mask])

    return omega_list, dist_list, time_list


def plot_omega_distribution(
    results_dir: Path,
    exp_ids: Sequence[int],
    bins: int = 30,
    distance_min: float = 0.0,
    distance_max: float = 850.0,
    figsize=(4.0, 3.0),
    dpi: int = 300,
    save_path: Optional[Path] = None,
):
    omega_list, dist_list, _ = _load_omega_arrays_for_experiments(results_dir, exp_ids)

    if not omega_list:
        raise FileNotFoundError("No omega_exp*.txt files found for selected experiments.")

    omega_all = np.concatenate(omega_list)
    dist_all = np.concatenate(dist_list)

    mask = (
        np.isfinite(omega_all)
        & np.isfinite(dist_all)
        & (dist_all >= distance_min)
        & (dist_all <= distance_max)
    )
    omega_sel = omega_all[mask]

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)
    ax.hist(omega_sel, bins=bins, density=True, alpha=0.7)

    ax.set_xlabel(r"omega [rad/s]")
    ax.set_ylabel("probability density")
    ax.set_title(f"Omega distribution, {distance_min:.0f} < d < {distance_max:.0f} µm")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", transparent=True)

    return fig, ax