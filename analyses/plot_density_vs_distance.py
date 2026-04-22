from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import image
from scipy.signal import savgol_filter

import clamutils_exp as cu

from analyses.load_results import load_settings, load_omega_file
from analyses.common import (
    COLOR_PINK,
    COLOR_GREEN,
    count_points_within_distance,
)


def _load_omega_arrays_for_experiments(
    results_dir: Path,
    exp_ids: Sequence[int],
    filename_template: str = "omega_exp{exp_id}.txt",
) -> tuple[list[np.ndarray], list[np.ndarray], list[np.ndarray]]:
    """
    Carica omega, distanza e tempo per una lista di esperimenti.

    Ritorna:
        omega_list
        dist_list
        time_list
    """
    omega_list: list[np.ndarray] = []
    dist_list: list[np.ndarray] = []
    time_list: list[np.ndarray] = []

    for exp_id in exp_ids:
        path = results_dir / filename_template.format(exp_id=exp_id)
        if not path.exists():
            continue

        df = load_omega_file(path)
        if df.empty:
            continue

        omega = df["omega"].to_numpy(dtype=float)
        dist = df["dist"].to_numpy(dtype=float)
        time = df["time"].to_numpy(dtype=float)

        mask = np.isfinite(omega) & np.isfinite(dist) & np.isfinite(time)
        omega = omega[mask]
        dist = dist[mask]
        time = time[mask]

        if omega.size == 0:
            continue

        omega_list.append(omega)
        dist_list.append(dist)
        time_list.append(time)

    return omega_list, dist_list, time_list


def _calculate_radial_shell_measure(
    speckle: cu.Speckle,
    r_outer: float,
    r_inner: float = 0.0,
    num_points: int = 1000,
) -> float:
    """
    Replica l'idea del vecchio calculate_speckle_value:
    integra radial_length(r) tra r_inner e r_outer.
    """
    if r_outer <= r_inner:
        return np.nan

    delta_r = (r_outer - r_inner) / num_points
    points = np.linspace(
        r_inner + delta_r,
        r_outer - delta_r,
        num=num_points,
        endpoint=True,
    )
    result = np.sum(speckle.radial_length(points)) * delta_r
    return float(result)


def _build_reference_speckle(
    illum_dir: Path,
    ill_id: int,
    diam: float,
    pix: float,
    ctrp: int,
) -> cu.Speckle:
    """
    Costruisce il campo di riferimento per la normalizzazione geometrica.
    """
    fname = "illumination1.tiff" if ill_id == 0 else f"illumination{ill_id}.tiff"
    path = illum_dir / fname
    if not path.exists():
        raise FileNotFoundError(f"Missing illumination map for density normalization: {path}")

    field = image.imread(path).astype(float)
    return cu.Speckle(field, diam, pix, ctrp=ctrp)


def plot_density_vs_distance(
    results_dir: Path,
    settings_path: Path,
    illum_dir: Path,
    exp_ids: Optional[Sequence[int]] = None,
    mask=None,
    intensity_values: Optional[Sequence[int]] = None,
    final_window_width: float = 10000.0,
    normalize_to_t0: bool = True,
    illum_id_for_normalization: int = 7,
    pix_for_normalization: float = 1 / 0.96,
    ctr_for_normalization: int = 0,
    diam_for_normalization: float = 15.0,
    num_intervals_by_series: Optional[Sequence[int]] = None,
    xlim=(0, 600),
    use_log_y: bool = True,
    figsize=(3, 2.5),
    dpi: int = 300,
    save_path: Optional[Path] = None,
    filename_template: str = "omega_exp{exp_id}.txt",
):
    """
    Riproduce il blocco densityVsDistance del vecchio script, in particolare
    la parte 'densityFinalVSdistance'.

    Idea:
    - carica omega/dist/time per gruppi di esperimenti
    - seleziona l'ultima finestra temporale disponibile
    - costruisce la densità radiale di punti
    - normalizza geometricamente con radial_length()
    - opzionalmente normalizza rispetto alla densità media al tempo iniziale
    """
    settings = load_settings(settings_path)

    # Se illum_id_for_normalization non è stato sovrascritto dal chiamante (default=7)
    # e sono stati forniti exp_ids specifici, usa l'ill_id del primo esperimento valido.
    if illum_id_for_normalization == 7 and exp_ids is not None:
        for eid in exp_ids:
            row = settings.loc[settings["exps"] == eid]
            if not row.empty:
                illum_id_for_normalization = int(row.iloc[0]["ill"])
                break

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    if exp_ids is not None:
        selected_exp_groups = [list(exp_ids)]
        group_labels = ["selected"]
        group_colors = [COLOR_PINK]
    else:
        if mask is None:
            raise ValueError("Provide either exp_ids or a boolean mask on settings.")

        if intensity_values is None:
            selected = settings.loc[mask]
            intensity_values = sorted(selected["intensity"].unique().tolist())

        selected_exp_groups = []
        group_labels = []
        group_colors = [COLOR_PINK, COLOR_GREEN, "k", "b"]

        for intensity_value in intensity_values:
            group_mask = mask & (settings["intensity"] == intensity_value)
            exp_ids_group = settings.loc[group_mask, "exps"].astype(int).tolist()
            selected_exp_groups.append(exp_ids_group)
            group_labels.append(f"{intensity_value} µW")

    if num_intervals_by_series is None:
        num_intervals_by_series = [30, 50, 50, 30]

    # Speckle di riferimento per la normalizzazione geometrica
    reference_speckle = _build_reference_speckle(
        illum_dir=illum_dir,
        ill_id=illum_id_for_normalization,
        diam=diam_for_normalization,
        pix=pix_for_normalization,
        ctrp=ctr_for_normalization,
    )

    plotted_any = False

    for i, exp_group in enumerate(selected_exp_groups):
        if not exp_group:
            continue

        omega_values_list, omega_distance_list, omega_times_list = _load_omega_arrays_for_experiments(
            results_dir=results_dir,
            exp_ids=exp_group,
            filename_template=filename_template,
        )

        if not omega_distance_list:
            continue

        # tempo massimo disponibile tra gli esperimenti selezionati
        max_time = np.max([np.max(times) for times in omega_times_list if len(times) > 0])

        interval_width = final_window_width
        interval_center = max_time - interval_width / 2.0

        # selezione dell'ultima finestra temporale
        selected_dist_list = []
        selected_val_list = []
        selected_time_list = []

        for dist_arr, val_arr, time_arr in zip(omega_distance_list, omega_values_list, omega_times_list):
            mask_time = (
                (time_arr >= interval_center - interval_width / 2.0) &
                (time_arr <= interval_center + interval_width / 2.0)
            )
            selected_dist_list.append(dist_arr[mask_time])
            selected_val_list.append(val_arr[mask_time])
            selected_time_list.append(time_arr[mask_time])

        if not selected_dist_list:
            continue

        omega_distance_final = np.concatenate(selected_dist_list) if selected_dist_list else np.array([])
        omega_values_final = np.concatenate(selected_val_list) if selected_val_list else np.array([])

        if omega_distance_final.size == 0:
            continue

        max_distance = np.max(omega_distance_final)

        if i < len(num_intervals_by_series):
            num_intervals = num_intervals_by_series[i]
        else:
            num_intervals = num_intervals_by_series[-1]

        centers, counts, lower_bounds, upper_bounds = count_points_within_distance(
            omega_distance_final,
            omega_values_final,
            max_distance,
            num_intervals=num_intervals,
        )

        normalisation = np.array([
            _calculate_radial_shell_measure(reference_speckle, r_outer=ub, r_inner=lb)
            for lb, ub in zip(lower_bounds, upper_bounds)
        ])

        # numero "effettivo" di frame nella finestra finale
        # replica l'idea del vecchio script, ma in modo robusto
        total_frames = 0
        for time_arr in selected_time_list:
            if len(time_arr) == 0:
                continue
            total_frames += len(np.unique(time_arr))

        if total_frames == 0:
            continue

        density = counts / normalisation / total_frames

        if normalize_to_t0:
            # Costruzione della densità di riferimento al tempo iniziale
            dist_t0_list = []
            val_t0_list = []

            for dist_arr, val_arr, time_arr in zip(omega_distance_list, omega_values_list, omega_times_list):
                mask_t0 = (time_arr == 0)
                dist_t0_list.append(dist_arr[mask_t0])
                val_t0_list.append(val_arr[mask_t0])

            if any(len(arr) > 0 for arr in dist_t0_list):
                omega_distance_t0 = np.concatenate([arr for arr in dist_t0_list if len(arr) > 0])
                omega_values_t0 = np.concatenate([arr for arr in val_t0_list if len(arr) > 0])

                centers0, counts0, _, _ = count_points_within_distance(
                    omega_distance_t0,
                    omega_values_t0,
                    max_distance,
                    num_intervals=num_intervals,
                )

                ref_density = np.mean(counts0 / normalisation / total_frames)
                if np.isfinite(ref_density) and ref_density > 0:
                    density = density / ref_density

        color = group_colors[i] if i < len(group_colors) else None
        label = group_labels[i] if i < len(group_labels) else f"group {i}"

        ax.scatter(centers, density, color=color, label=f"{label}")
        if len(centers) >= 5:
            ax.plot(
                centers,
                savgol_filter(density, window_length=5, polyorder=3),
                color=color,
                alpha=1.0,
                label=None,
            )
        else:
            ax.plot(centers, density, color=color, alpha=1.0)

        plotted_any = True

    if not plotted_any:
        raise FileNotFoundError("No usable omega result files found for selected experiments.")

    ax.set_xlabel(r"distance (µm)")
    ax.set_ylabel(r"density (µm$^{-2}$)")
    ax.set_xlim(list(xlim))

    if use_log_y:
        ax.set_yscale("log")

    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", transparent=True)

    return fig, ax


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent

    results_dir = base / "results"
    settings_path = base / "Clamidomoni_settings.txt"
    illum_dir = base / "Illuminations"

    settings = load_settings(settings_path)

    # Esempio vicino al vecchio script:
    # intensity_values = [180, 7, 30]
    # indices = (intensity == intensity_values[i]) & (exps >67) & (exps < 83)
    selection_mask = (settings["exps"] > 67) & (settings["exps"] < 83)

    plot_density_vs_distance(
        results_dir=results_dir,
        settings_path=settings_path,
        illum_dir=illum_dir,
        mask=selection_mask,
        intensity_values=[180, 7, 30],
        save_path=base / "figs" / "density_vs_distance.pdf",
    )

    plt.show()