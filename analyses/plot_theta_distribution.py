from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np
import matplotlib.pyplot as plt

from analyses.load_results import load_settings, load_theta_file
from analyses.common import COLOR_GREEN, COLOR_ORANGE


def _load_theta_arrays_for_experiments(
    results_dir: Path,
    exp_ids: Sequence[int],
    filename_template: str = "Theta_exp{exp_id}_FluoRadialised.txt",
) -> tuple[list[np.ndarray], list[np.ndarray]]:
    """
    Carica theta e distanza per una lista di esperimenti.

    Ritorna:
        theta_list: lista di array theta
        dist_list: lista di array distanza
    """
    theta_list: list[np.ndarray] = []
    dist_list: list[np.ndarray] = []

    for exp_id in exp_ids:
        theta_path = results_dir / filename_template.format(exp_id=exp_id)
        if not theta_path.exists():
            continue

        df = load_theta_file(theta_path)
        if df.empty:
            continue

        theta = df["theta"].to_numpy(dtype=float)
        dist = df["dist"].to_numpy(dtype=float)

        mask = np.isfinite(theta) & np.isfinite(dist)
        theta = theta[mask]
        dist = dist[mask]

        if theta.size == 0:
            continue

        theta_list.append(theta)
        dist_list.append(dist)

    return theta_list, dist_list


def plot_theta_distribution(
    results_dir: Path,
    settings_path: Path,
    exp_ids: Optional[Sequence[int]] = None,
    mask=None,
    intensity_values: Optional[Sequence[int]] = None,
    inner_circle: float = 300.0,
    thickness_circle: float = 0.0,
    buffer: float = 0.0,
    max_distance: float = 800.0,
    bins: int = 10,
    figsize=(1.75, 2.25),
    dpi: int = 600,
    save_path: Optional[Path] = None,
    filename_template: str = "Theta_exp{exp_id}_FluoRadialised.txt",
):
    """
    Riproduce il blocco thetaCalculation del vecchio script.

    Modalità d'uso:
    1) passare exp_ids direttamente
    2) passare una mask su settings
    3) opzionalmente filtrare per intensity_values dentro la mask
    """
    settings = load_settings(settings_path)

    fig, axs = plt.subplots(2, 1, figsize=figsize, dpi=dpi)

    # Caso 1: esperimenti passati direttamente
    if exp_ids is not None:
        theta_list, dist_list = _load_theta_arrays_for_experiments(
            results_dir=results_dir,
            exp_ids=exp_ids,
            filename_template=filename_template,
        )

        if not theta_list:
            raise FileNotFoundError("No theta result files found for selected experiments.")

        theta_values = np.concatenate(theta_list)
        theta_distance = np.concatenate(dist_list)

        theta_inside = theta_values[theta_distance <= inner_circle]
        theta_outside = theta_values[
            (theta_distance > inner_circle + thickness_circle + buffer) &
            (theta_distance < max_distance)
        ]

        if theta_inside.size:
            axs[0].hist(theta_inside, color=COLOR_GREEN, alpha=0.5, bins=bins, density=True)
        else:
            axs[0].text(0.1, 0.5, "No inside data", transform=axs[0].transAxes)

        if theta_outside.size:
            axs[1].hist(theta_outside, color=COLOR_ORANGE, alpha=0.5, bins=bins, density=True)
        else:
            axs[1].text(0.1, 0.5, "No outside data", transform=axs[1].transAxes)

    else:
        # Caso 2: mask su settings e gruppi per intensity
        if mask is None:
            raise ValueError("Provide either exp_ids or a boolean mask on settings.")

        if intensity_values is None:
            selected = settings.loc[mask]
            intensity_values = sorted(selected["intensity"].unique().tolist())

        for intensity_value in intensity_values:
            group_mask = mask & (settings["intensity"] == intensity_value)
            exp_ids_group = settings.loc[group_mask, "exps"].astype(int).tolist()

            theta_list, dist_list = _load_theta_arrays_for_experiments(
                results_dir=results_dir,
                exp_ids=exp_ids_group,
                filename_template=filename_template,
            )

            if not theta_list:
                continue

            theta_values = np.concatenate(theta_list)
            theta_distance = np.concatenate(dist_list)

            theta_inside = theta_values[theta_distance <= inner_circle]
            theta_outside = theta_values[
                (theta_distance > inner_circle + thickness_circle + buffer) &
                (theta_distance < max_distance)
            ]

            if theta_inside.size:
                axs[0].hist(theta_inside, color=COLOR_GREEN, alpha=0.5, bins=bins, density=True)

            if theta_outside.size:
                axs[1].hist(
                    theta_outside,
                    color=COLOR_ORANGE,
                    alpha=0.5,
                    bins=bins,
                    density=True,
                    label=f"{intensity_value} µW",
                )

    for ax in axs:
        ax.set_xticks([0, np.pi / 2, np.pi], ['$0$', r'$\pi/2$', r'$\pi$'])
        ax.set_yticks([0.2, 0.3, 0.4])

    axs[1].set_xlabel(r'$\theta$')
    fig.supylabel(r'$P(\theta)$', va='center')
    axs[0].set_xlabel('')
    axs[0].set_ylim([0.25, 0.4])
    axs[1].set_ylim([0.25, 0.4])

    if axs[1].get_legend_handles_labels()[0]:
        axs[1].legend(frameon=False)

    fig.tight_layout()

    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, format='pdf', transparent=True)

    return fig, axs


if __name__ == "__main__":
    base = Path(__file__).resolve().parent.parent
    results_dir = base / "results"
    settings_path = base / "Clamidomoni_settings.txt"

    settings = load_settings(settings_path)

    # Esempio vicino al vecchio script:
    # indices = ((exps > 67) & (exps < 70))
    selection_mask = (settings["exps"] > 67) & (settings["exps"] < 70)

    plot_theta_distribution(
        results_dir=results_dir,
        settings_path=settings_path,
        mask=selection_mask,
        intensity_values=[180],
        inner_circle=300,
        thickness_circle=0,
        buffer=0,
        max_distance=800,
        save_path=base / "figs" / "theta_distribution.pdf",
    )

    plt.show()