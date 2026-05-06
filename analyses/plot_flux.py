from __future__ import annotations

from pathlib import Path
from typing import Iterable, Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from analyses.load_results import load_settings, load_flux_file
from analyses.common import COLOR_PINK


def _load_flux_series_for_experiments(results_dir: Path, exp_ids: Iterable[int], disc_radius: int) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Carica i file Flux_exp{exp}_disc{disc_radius}.txt per gli exp richiesti.

    Ritorna:
        flux_times_list: lista di array 1D
        flux_values_list: lista di array 1D
    """
    flux_times_list: list[np.ndarray] = []
    flux_values_list: list[np.ndarray] = []

    for exp_id in exp_ids:
        flux_path = results_dir / f"Flux_exp{exp_id}_disc{disc_radius}.txt"
        if not flux_path.exists():
            continue

        df = load_flux_file(flux_path)
        if df.empty:
            continue

        times = df["time"].to_numpy(dtype=float)
        flux = df["flux"].to_numpy(dtype=float)

        mask = np.isfinite(times) & np.isfinite(flux)
        times = times[mask]
        flux = flux[mask]

        if times.size == 0:
            continue

        flux_times_list.append(times)
        flux_values_list.append(flux)

    return flux_times_list, flux_values_list


def _mean_over_equal_length_series(series_list: Sequence[np.ndarray]) -> np.ndarray:
    """
    Fa la media su serie 1D tagliandole tutte alla lunghezza minima comune.
    """
    if not series_list:
        return np.array([])

    min_len = min(len(s) for s in series_list)
    if min_len == 0:
        return np.array([])

    stacked = np.vstack([s[:min_len] for s in series_list])
    return np.mean(stacked, axis=0)


def plot_flux(
    results_dir: Path,
    settings_path: Path,
    exp_ids: Optional[Sequence[int]] = None,
    mask=None,
    disc_radius: int = 300,
    intensity_values: Optional[Sequence[int]] = None,
    colors: Optional[Sequence] = None,
    labels: Optional[Sequence[str]] = None,
    figsize=(6.5, 6.5),
    dpi: int = 300,
    save_path: Optional[Path] = None,
):
    """
    Plot del flusso medio nel tempo.

    Puoi usarlo in due modi:

    1) passando exp_ids direttamente
       es: exp_ids=[98, 99, 100]

    2) passando una mask booleana su settings
       es: mask = (settings["intensity"] == 180) & (settings["ill"].isin([9,10]))

    Se passi intensity_values + mask, allora per ogni intensità crea una curva media distinta.
    """
    settings = load_settings(settings_path)

    fig, ax = plt.subplots(1, 1, figsize=figsize, dpi=dpi)

    # Caso semplice: una sola selezione esplicita di exp_ids
    if exp_ids is not None:
        flux_times_list, flux_values_list = _load_flux_series_for_experiments(
            results_dir=results_dir,
            exp_ids=exp_ids,
            disc_radius=disc_radius,
        )

        if not flux_times_list:
            raise FileNotFoundError("No flux result files found for selected experiments.")

        mean_t = _mean_over_equal_length_series(flux_times_list)
        mean_f = _mean_over_equal_length_series(flux_values_list)

        ax.scatter(mean_t, mean_f, linewidth=2)
        ax.set_xlabel("t [s]")
        ax.set_ylabel("flux [s$^{-1}$]")
        ax.legend([], frameon=False)

    else:
        # Caso raggruppato per intensità dentro una mask
        if mask is None:
            raise ValueError("Provide either exp_ids or a boolean mask on settings.")

        if intensity_values is None:
            selected_settings = settings.loc[mask]
            intensity_values = sorted(selected_settings["intensity"].unique().tolist())

        if colors is None:
            if len(intensity_values) == 2:
                colors = ["k", COLOR_PINK]
            else:
                colors = [None] * len(intensity_values)

        if labels is None:
            labels = [f"{ival} µW" for ival in intensity_values]

        for i, intensity_value in enumerate(intensity_values):
            group_mask = mask & (settings["intensity"] == intensity_value)
            exp_ids_group = settings.loc[group_mask, "exps"].astype(int).tolist()

            flux_times_list, flux_values_list = _load_flux_series_for_experiments(
                results_dir=results_dir,
                exp_ids=exp_ids_group,
                disc_radius=disc_radius,
            )

            if not flux_times_list:
                continue

            mean_t = _mean_over_equal_length_series(flux_times_list)
            mean_f = _mean_over_equal_length_series(flux_values_list)

            ax.scatter(
                mean_t,
                mean_f,
                color=colors[i] if i < len(colors) else None,
                edgecolors=colors[i] if i < len(colors) else None,
                alpha=1.0,
                linewidth=2,
                label=labels[i] if i < len(labels) else f"{intensity_value} µW",
            )

    ax.set_xlabel("t [s]")
    ax.set_ylabel("flux [s$^{-1}$]")
    #ax.plot(t, flux, label=f"exp {exp_ids}")
    ax.legend(frameon=False)
    plt.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format="pdf", transparent=True)

    return fig, ax


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent

    # Adatta questi path se i risultati stanno dentro runs/<timestamp>/...
    results_dir = base / "results"
    settings_path = base / "Clamidomoni_settings.txt"

    settings = load_settings(settings_path)

    # Esempio coerente col vecchio script:
    # outside ring: ill == 9 oppure 10, intensità 10 e 180
    selection_mask = settings["ill"].isin([9, 10])

    plot_flux(
        results_dir=results_dir,
        settings_path=settings_path,
        mask=selection_mask,
        disc_radius=300,
        intensity_values=[10, 180],
        save_path=base / "figs" / "flux_plot.pdf",
    )

    plt.show()