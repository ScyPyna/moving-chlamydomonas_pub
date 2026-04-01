from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from datetime import datetime
from typing import Optional
import json

from clam_pipeline.io import build_paths, load_settings, load_trajectories_txt, save_df_csv
from clam_pipeline.analyze import analyze_experiment, AnalysisParams, FilterParams


def _save_legacy_txt(df, out_path: Path, header: str, float_fmt: str = "%.6f") -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if df.empty:
        with open(out_path, "w", encoding="utf-8") as f:
            f.write(header)
        return

    import numpy as np

    arr = df.to_numpy()
    fmt = [float_fmt] * arr.shape[1]
    np.savetxt(out_path, np.array([]))
    with open(out_path, "a", encoding="utf-8") as f:
        f.write(header)
        np.savetxt(f, arr, fmt=fmt)


def main(
    code_dir: Path,
    traj_dir: Path,
    settings_txt: Path,
    illum_dir: Path,
    radial_dir: Path,
    results_dir: Path,
    exp_id: int = 68,
    params: Optional[AnalysisParams] = None,
    fparams: Optional[FilterParams] = None,
) -> dict:
    paths = build_paths(
        code_dir=code_dir,
        traj_dir=traj_dir,
        settings_txt=settings_txt,
        illum_dir=illum_dir,
        radial_dir=radial_dir,
        results_dir=results_dir,
    )

    if not paths.settings_txt.exists():
        raise FileNotFoundError(f"Missing settings file: {paths.settings_txt}")
    if not paths.traj_dir.exists():
        raise FileNotFoundError(f"Missing trajectories dir: {paths.traj_dir}")
    if not paths.illum_dir.exists():
        raise FileNotFoundError(f"Missing illumination dir: {paths.illum_dir}")

    paths.runs_dir.mkdir(parents=True, exist_ok=True)

    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = paths.runs_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    if params is None:
        params = AnalysisParams(
            diam=10,
            dt=0.1,
            particle_show=5,
            disc_radius=350,
            tumbling_threshold=5.0,
            radial_map=True,
            flux_cg_in=1,
            flux_cg_out=150,
            nskip=1,
        )
    if fparams is None:
        fparams = FilterParams(
            lifetime_min=0,
            threshold_dist=1,
            threshold_dist_tot=3,
            dist_step=1,
            strict=True,
            only_inside=True,
            blinking_ratio=0.5,
            blinking_tolerance=0.1,
            long_track_min_snaps=20,
        )

    if params.radial_map and not paths.radial_dir.exists():
        raise FileNotFoundError(f"radial_map=True but missing radial_dir: {paths.radial_dir}")

    settings = load_settings(paths.settings_txt)

    row_df = settings.loc[settings["exps"] == exp_id]
    if row_df.empty:
        raise ValueError(f"Experiment {exp_id} not found in settings.")
    row = row_df.iloc[0]

    traj_path = paths.traj_dir / f"tAlgae{exp_id}.txt"
    traj_df = load_trajectories_txt(traj_path)

    outputs = analyze_experiment(
        exp_id=exp_id,
        settings_row=row,
        traj_df=traj_df,
        illum_dir=paths.illum_dir,
        radial_dir=paths.radial_dir,
        params=params,
        fparams=fparams,
    )

    # run-specific structured outputs
    out_exp_dir = run_dir / f"exp_{exp_id:04d}"
    out_exp_dir.mkdir(parents=True, exist_ok=True)

    save_df_csv(outputs.theta, out_exp_dir / "theta.csv")
    save_df_csv(outputs.flux, out_exp_dir / "flux.csv")
    save_df_csv(outputs.angle, out_exp_dir / "angle.csv")
    save_df_csv(outputs.omega, out_exp_dir / "omega.csv")
    save_df_csv(outputs.tumbling_orientation, out_exp_dir / "tumbling_orientation.csv")
    save_df_csv(outputs.tumbling_duration, out_exp_dir / "tumbling_duration.csv")
    save_df_csv(outputs.nt_stats, out_exp_dir / "nt_stats.csv")

    # legacy-compatible flat txt outputs in results_dir
    flat_results_dir = paths.runs_dir.parent
    # legacy-compatible N_dist file: keep only the historical filtered columns
    _save_legacy_txt(
        outputs.nt_stats[["N", "n_disc", "dists"]],
        flat_results_dir / f"N_dist_exp{exp_id}_disc{int(params.disc_radius)}.txt",
        "N n_disc dists\n",
    )

    # extended comparison file: raw vs filtered side by side
    _save_legacy_txt(
        outputs.nt_stats[["N_raw", "n_disc_raw", "dists_raw", "N", "n_disc", "dists"]],
        flat_results_dir / f"N_dist_compare_exp{exp_id}_disc{int(params.disc_radius)}.txt",
        "N_raw n_disc_raw dists_raw N n_disc dists\n",
    )
    _save_legacy_txt(
        outputs.flux,
        flat_results_dir / f"Flux_exp{exp_id}_disc{int(params.disc_radius)}.txt",
        "time flux\n",
    )
    _save_legacy_txt(
        outputs.theta,
        flat_results_dir / f"Theta_exp{exp_id}_FluoRadialised.txt",
        "Theta dists\n",
    )
    _save_legacy_txt(
        outputs.angle,
        flat_results_dir / f"angle_exp{exp_id}.txt",
        "angle dists v\n",
    )
    _save_legacy_txt(
        outputs.omega,
        flat_results_dir / f"omega_exp{exp_id}.txt",
        "omega dists time\n",
    )
    _save_legacy_txt(
        outputs.tumbling_orientation,
        flat_results_dir / f"tumb_exp{exp_id}_tthres{params.tumbling_threshold:.1f}.txt",
        "theta dists duration time\n",
    )
    _save_legacy_txt(
        outputs.tumbling_duration,
        flat_results_dir / f"tumb_dur_exp{exp_id}_tthres{params.tumbling_threshold:.1f}.txt",
        "dists duration time\n",
    )

    (run_dir / "config_used.json").write_text(
        json.dumps(
            {
                "code_dir": str(paths.code_dir),
                "traj_dir": str(paths.traj_dir),
                "settings_txt": str(paths.settings_txt),
                "illum_dir": str(paths.illum_dir),
                "radial_dir": str(paths.radial_dir),
                "exp_id": exp_id,
                "params": asdict(params),
                "filters": asdict(fparams),
            },
            indent=2,
        )
    )

    print("[OK] Run saved in:", run_dir)
    print("[OK] Legacy txt results saved in:", flat_results_dir)
    print("[OK] Generated:")
    print("  - Theta_exp*.txt")
    print("  - Flux_exp*.txt")
    print("  - angle_exp*.txt")
    print("  - omega_exp*.txt")
    print("  - tumb_exp*.txt")
    print("  - tumb_dur_exp*.txt")
    print("  - N_dist_exp*.txt")
    print("  - N_dist_compare_exp*.txt")

    return {
        "flat_results_dir": flat_results_dir,
        "settings_txt": paths.settings_txt,
        "illum_dir": paths.illum_dir,
        "run_dir": run_dir,
        "exp_id": exp_id,
        "params": params,
        "disc_radius": params.disc_radius,
        "tumbling_threshold": params.tumbling_threshold,
    }