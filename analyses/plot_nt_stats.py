from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter


def _load_nt_arrays_for_experiments(
    results_dir: Path,
    exp_ids: Sequence[int],
    disc_radius: int,
):
    n_list = []
    ndisc_list = []
    dist_list = []

    for exp_id in exp_ids:
        path = results_dir / f"N_dist_exp{exp_id}_disc{disc_radius}.txt"
        if not path.exists():
            continue

        data = np.genfromtxt(path, skip_header=1)
        if isinstance(data, float) and np.isnan(data):
            continue
        if getattr(data, "size", 0) == 0:
            continue
        if data.ndim == 1:
            data = data.reshape(1, -1)

        n_list.append(data[:, 0].astype(float))
        ndisc_list.append(data[:, 1].astype(float))
        dist_list.append(data[:, 2].astype(float))

    return n_list, ndisc_list, dist_list


def _mean_over_equal_length(series_list):
    if not series_list:
        return np.array([])
    min_len = min(len(x) for x in series_list)
    if min_len == 0:
        return np.array([])
    arr = np.vstack([x[:min_len] for x in series_list])
    return np.mean(arr, axis=0)


def plot_nt_stats(
    results_dir: Path,
    exp_ids: Sequence[int],
    disc_radius: int = 350,
    dt: float = 0.1,
    smooth: bool = True,
    figsize=(6.0, 4.0),
    dpi: int = 300,
    save_path: Optional[Path] = None,
):
    n_list, ndisc_list, dist_list = _load_nt_arrays_for_experiments(results_dir, exp_ids, disc_radius)

    if not n_list:
        raise FileNotFoundError("No N_dist_exp*.txt files found for selected experiments.")

    n_mean = _mean_over_equal_length(n_list)
    ndisc_mean = _mean_over_equal_length(ndisc_list)
    dist_mean = _mean_over_equal_length(dist_list)

    t = np.arange(len(n_mean)) * dt

    if smooth and len(n_mean) >= 11:
        win = min(len(n_mean) // 2 * 2 - 1, 31)
        if win >= 5:
            n_mean = savgol_filter(n_mean, window_length=win, polyorder=2)
            ndisc_mean = savgol_filter(ndisc_mean, window_length=win, polyorder=2)
            dist_mean = savgol_filter(dist_mean, window_length=win, polyorder=2)

    fig, axs = plt.subplots(3, 1, figsize=figsize, dpi=dpi, sharex=True)

    axs[0].plot(t, n_mean)
    axs[0].set_ylabel("N")

    axs[1].plot(t, ndisc_mean)
    axs[1].set_ylabel("N_disc")

    axs[2].plot(t, dist_mean)
    axs[2].set_ylabel("mean dist")
    axs[2].set_xlabel("t [s]")

    fig.suptitle("Time statistics")
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", transparent=True)

    return fig, axs