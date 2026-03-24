from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from analyses.common import bin_data, COLOR_PINK


def _load_tumbling_orientation(results_dir: Path, exp_ids: Sequence[int]):
    theta_list = []
    dist_list = []
    dur_list = []
    time_list = []

    for exp_id in exp_ids:
        path = results_dir / f"tumb_exp{exp_id}_tthres5.0.txt"
        if not path.exists():
            continue

        data = np.genfromtxt(path, skip_header=1)
        if isinstance(data, float) and np.isnan(data):
            continue
        if getattr(data, "size", 0) == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        theta_list.append(data[:, 0].astype(float))
        dist_list.append(data[:, 1].astype(float))
        dur_list.append(data[:, 2].astype(float))
        time_list.append(data[:, 3].astype(float))

    return theta_list, dist_list, dur_list, time_list


def _load_tumbling_duration(results_dir: Path, exp_ids: Sequence[int]):
    dist_list = []
    dur_list = []
    time_list = []

    for exp_id in exp_ids:
        path = results_dir / f"tumb_dur_exp{exp_id}_tthres5.0.txt"
        if not path.exists():
            continue

        data = np.genfromtxt(path, skip_header=1)
        if isinstance(data, float) and np.isnan(data):
            continue
        if getattr(data, "size", 0) == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        dist_list.append(data[:, 0].astype(float))
        dur_list.append(data[:, 1].astype(float))
        time_list.append(data[:, 2].astype(float))

    return dist_list, dur_list, time_list


def plot_tumbling_vs_gradient(
    results_dir: Path,
    exp_ids: Sequence[int],
    distance_min: float = 300.0,
    distance_max: float = 500.0,
    n_bins: int = 5,
    figsize=(3.5, 2.5),
    dpi: int = 300,
    save_path: Optional[Path] = None,
):
    theta_list, dist_list, dur_list, _ = _load_tumbling_orientation(results_dir, exp_ids)

    if not theta_list:
        raise FileNotFoundError("No tumb_exp*.txt files found for selected experiments.")

    theta = np.concatenate(theta_list)
    dist = np.concatenate(dist_list)
    dur = np.concatenate(dur_list)

    mask = (
        np.isfinite(theta)
        & np.isfinite(dist)
        & np.isfinite(dur)
        & (dist > distance_min)
        & (dist < distance_max)
    )

    mean_x, mean_y, std_y = bin_data(theta[mask], dur[mask], n_bins, use_quantile_bins=True)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    ax.plot(mean_x, mean_y, "--", color=COLOR_PINK, alpha=0.2)
    ax.fill_between(mean_x, mean_y - std_y, mean_y + std_y, color=COLOR_PINK, alpha=0.2, linewidth=0)
    ax.scatter(mean_x, mean_y, color=COLOR_PINK, s=20)

    ax.set_xlabel(r"$\theta$")
    ax.set_ylabel(r"average run duration (s)")
    ax.set_xticks([0, np.pi / 2, np.pi], ['$0$', r'$\pi/2$', r'$\pi$'])

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", transparent=True)

    return fig, ax


def plot_tumbling_duration_vs_distance(
    results_dir: Path,
    exp_ids: Sequence[int],
    n_bins: int = 10,
    figsize=(6.0, 4.0),
    dpi: int = 300,
    save_path: Optional[Path] = None,
):
    dist_list, dur_list, _ = _load_tumbling_duration(results_dir, exp_ids)

    if not dist_list:
        raise FileNotFoundError("No tumb_dur_exp*.txt files found for selected experiments.")

    dist = np.concatenate(dist_list)
    dur = np.concatenate(dur_list)

    mask = np.isfinite(dist) & np.isfinite(dur)
    mean_x, mean_y, std_y = bin_data(dist[mask], dur[mask], n_bins)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    ax.errorbar(mean_x, mean_y, yerr=std_y, fmt="o")
    ax.set_xlabel("distance (µm)")
    ax.set_ylabel("tumbling duration (s)")

    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", transparent=True)

    return fig, ax