from __future__ import annotations

from pathlib import Path
import numpy as np
import pandas as pd

SETTINGS_COLUMNS = [
    "exps", "exp_type", "ill", "intensity", "pix_arr",
    "no_skip", "ctr", "fluo", "lin", "axis"
]


def load_settings(settings_path: Path) -> pd.DataFrame:
    arr = np.genfromtxt(settings_path)
    df = pd.DataFrame(arr, columns=SETTINGS_COLUMNS)

    int_cols = ["exps", "exp_type", "ill", "intensity", "no_skip", "ctr", "fluo", "lin", "axis"]
    for c in int_cols:
        df[c] = df[c].astype(int)

    df["pix_arr"] = df["pix_arr"].astype(float)
    df["use"] = df["no_skip"].astype(bool)
    return df


def _load_txt_table(path: Path, columns):
    data = np.genfromtxt(path, skip_header=1)

    if isinstance(data, float) and np.isnan(data):
        return pd.DataFrame(columns=columns)

    if getattr(data, "size", 0) == 0:
        return pd.DataFrame(columns=columns)

    if data.ndim == 1:
        data = data.reshape(1, -1)

    return pd.DataFrame(data, columns=columns)


def load_theta_file(path: Path) -> pd.DataFrame:
    return _load_txt_table(path, ["theta", "dist"])


def load_flux_file(path: Path) -> pd.DataFrame:
    return _load_txt_table(path, ["time", "flux"])


def load_omega_file(path: Path) -> pd.DataFrame:
    return _load_txt_table(path, ["omega", "dist", "time"])


def load_nt_file(path: Path) -> pd.DataFrame:
    return _load_txt_table(path, ["N", "n_disc", "dist"])


def load_tumbling_file(path: Path) -> pd.DataFrame:
    return _load_txt_table(path, ["theta", "dist", "duration", "time"])


def load_tumbling_duration_file(path: Path) -> pd.DataFrame:
    return _load_txt_table(path, ["dist", "duration", "time"])