from __future__ import annotations

from pathlib import Path
import argparse
import traceback

from analyses.load_results import load_settings
from analyses.plot_theta_distribution import plot_theta_distribution
from analyses.plot_flux import plot_flux
from analyses.plot_density_vs_distance import plot_density_vs_distance
from analyses.plot_angle_distribution import plot_angle_distribution
from analyses.plot_omega_distribution import plot_omega_distribution
from analyses.plot_nt_stats import plot_nt_stats
from analyses.plot_tumbling import (
    plot_tumbling_vs_gradient,
    plot_tumbling_duration_vs_distance,
)


def _exp_suffix(exp_id: int) -> str:
    return f"exp{exp_id}"


def main():
    parser = argparse.ArgumentParser(
        description="Generate all available legacy plots from .txt result files."
    )
    parser.add_argument(
        "--exp",
        type=int,
        nargs="+",
        required=True,
        help="Experiment IDs to include, e.g. --exp 68 or --exp 68 73",
    )
    parser.add_argument(
        "--illum-dir",
        type=Path,
        required=True,
        help="Path to the folder containing the illumination*.tiff files.",
    )
    args = parser.parse_args()
    exp_ids = args.exp

    base = Path(__file__).resolve().parent.parent
    legacy_results_dir = base
    settings_path = base / "Clamidomoni_settings.txt"
    figs_dir = base / "results" / "figs"
    illum_dir = args.illum_dir.resolve()

    figs_dir.mkdir(parents=True, exist_ok=True)

    settings = load_settings(settings_path)

    print(f"[INFO] Selected experiments: {exp_ids}")
    print(f"[INFO] Legacy results dir: {legacy_results_dir}")
    print(f"[INFO] Figures output dir: {figs_dir}")

    ok = []
    failed = []

    for exp_id in exp_ids:
        suffix = _exp_suffix(exp_id)
        exp_mask = settings["exps"] == exp_id

        print(f"\n[EXP] {exp_id}")

        jobs = []

        jobs.append((
            f"theta_distribution_{suffix}",
            lambda exp_id=exp_id, exp_mask=exp_mask: plot_theta_distribution(
                results_dir=legacy_results_dir,
                settings_path=settings_path,
                exp_ids=[exp_id],
                mask=exp_mask,
                inner_circle=300,
                thickness_circle=0,
                buffer=0,
                max_distance=800,
                save_path=figs_dir / f"theta_distribution_{suffix}.pdf",
            )
        ))

        jobs.append((
            f"flux_plot_{suffix}",
            lambda exp_id=exp_id: plot_flux(
                results_dir=legacy_results_dir,
                settings_path=settings_path,
                exp_ids=[exp_id],
                disc_radius=350,
                save_path=figs_dir / f"flux_plot_{suffix}.pdf",
            )
        ))

        jobs.append((
            f"density_vs_distance_{suffix}",
            lambda exp_id=exp_id: plot_density_vs_distance(
                results_dir=legacy_results_dir,
                settings_path=settings_path,
                illum_dir=illum_dir,
                exp_ids=[exp_id],
                save_path=figs_dir / f"density_vs_distance_{suffix}.pdf",
            )
        ))

        jobs.append((
            f"angle_distribution_{suffix}",
            lambda exp_id=exp_id: plot_angle_distribution(
                results_dir=legacy_results_dir,
                exp_ids=[exp_id],
                save_path=figs_dir / f"angle_distribution_{suffix}.pdf",
            )
        ))

        jobs.append((
            f"omega_distribution_{suffix}",
            lambda exp_id=exp_id: plot_omega_distribution(
                results_dir=legacy_results_dir,
                exp_ids=[exp_id],
                save_path=figs_dir / f"omega_distribution_{suffix}.pdf",
            )
        ))

        jobs.append((
            f"nt_stats_{suffix}",
            lambda exp_id=exp_id: plot_nt_stats(
                results_dir=legacy_results_dir,
                exp_ids=[exp_id],
                disc_radius=350,
                save_path=figs_dir / f"nt_stats_{suffix}.pdf",
            )
        ))

        jobs.append((
            f"tumbling_vs_gradient_{suffix}",
            lambda exp_id=exp_id: plot_tumbling_vs_gradient(
                results_dir=legacy_results_dir,
                exp_ids=[exp_id],
                save_path=figs_dir / f"tumbling_vs_gradient_{suffix}.pdf",
            )
        ))

        jobs.append((
            f"tumbling_duration_vs_distance_{suffix}",
            lambda exp_id=exp_id: plot_tumbling_duration_vs_distance(
                results_dir=legacy_results_dir,
                exp_ids=[exp_id],
                save_path=figs_dir / f"tumbling_duration_vs_distance_{suffix}.pdf",
            )
        ))

        for name, job in jobs:
            print(f"[RUN] {name}")
            try:
                job()
                ok.append(name)
                print(f"[OK]  {name}")
            except Exception as e:
                failed.append((name, e))
                print(f"[FAIL] {name}: {e}")
                traceback.print_exc()

    print("\n=== SUMMARY ===")
    print("Generated plots:")
    for name in ok:
        print(f"  - {name}")

    if failed:
        print("Failed plots:")
        for name, err in failed:
            print(f"  - {name}: {err}")
    else:
        print("All plots generated successfully.")

    print(f"\nOutput directory: {figs_dir}")


if __name__ == "__main__":
    main()