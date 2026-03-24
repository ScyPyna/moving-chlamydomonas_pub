from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import numpy as np
import pandas as pd

SETTINGS_COLUMNS = [
    "exps", "exp_type", "ill", "intensity", "pix_arr",
    "no_skip", "ctr", "fluo", "lin", "axis"
]

@dataclass(frozen=True)
class Paths:
    code_dir: Path
    traj_dir: Path
    settings_txt: Path
    illum_dir: Path
    radial_dir: Path
    runs_dir: Path

def build_paths(
    code_dir: Path,
    traj_dir: Path,
    settings_txt: Path,
    illum_dir: Path,
    radial_dir: Path,
    results_dir: Path,
) -> Paths:
    return Paths(
        code_dir=code_dir.resolve(),
        traj_dir=traj_dir.resolve(),
        settings_txt=settings_txt.resolve(),
        illum_dir=illum_dir.resolve(),
        radial_dir=radial_dir.resolve(),
        runs_dir=results_dir.resolve(),
    )

def load_settings(settings_txt: Path) -> pd.DataFrame:
    if not settings_txt.exists():
        raise FileNotFoundError(f"Missing settings file: {settings_txt}")

    arr = np.genfromtxt(settings_txt)
    if arr.ndim != 2 or arr.shape[1] < len(SETTINGS_COLUMNS):
        raise ValueError(
            f"Unexpected settings shape in {settings_txt}. "
            f"Expected N x {len(SETTINGS_COLUMNS)} numeric table."
        )

    df = pd.DataFrame(arr[:, :len(SETTINGS_COLUMNS)], columns=SETTINGS_COLUMNS)

    int_cols = ["exps", "exp_type", "ill", "intensity", "no_skip", "ctr", "fluo", "lin", "axis"]
    for c in int_cols:
        df[c] = df[c].astype(int)

    df["pix_arr"] = df["pix_arr"].astype(float)
    df["use"] = df["no_skip"].astype(bool)
    return df

def load_trajectories_txt(traj_path: Path) -> pd.DataFrame:
    if not traj_path.exists():
        raise FileNotFoundError(f"Missing trajectories file: {traj_path}")

    data = np.genfromtxt(traj_path, delimiter=",", skip_header=1)
    if data.ndim != 2:
        raise ValueError(f"Trajectories file {traj_path} doesn't look like a 2D table.")
    if data.shape[1] < 4:
        raise ValueError(
            f"Unexpected trajectories format in {traj_path}. "
            f"Need at least 4 columns: snap,y,x,tag. Got {data.shape[1]}."
        )

    return pd.DataFrame({
        "snap": data[:, 0].astype(int),
        "y_px": data[:, 1].astype(float),
        "x_px": data[:, 2].astype(float),
        "tag": data[:, 3].astype(int),
    })

def save_df_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False, lineterminator="\n")