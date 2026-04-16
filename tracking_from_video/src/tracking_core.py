from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Any

import numpy as np
import pandas as pd
import pims
import trackpy as tp


@dataclass(frozen=True)
class TrackingConfig:
    channel: int
    background_subtraction: bool
    background_method: str
    diameter: int
    minmass: int
    invert: bool
    search_range: int
    adaptive_stop: int
    adaptive_step: float
    memory: Optional[int]
    filter_stubs: int


def load_config_json(path: Path) -> TrackingConfig:
    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    return TrackingConfig(
        channel=int(cfg["channel"]),
        background_subtraction=bool(cfg["background_subtraction"]),
        background_method=str(cfg.get("background_method", "mean")),
        diameter=int(cfg["locate"]["diameter"]),
        minmass=int(cfg["locate"]["minmass"]),
        invert=bool(cfg["locate"]["invert"]),
        search_range=int(cfg["link"]["search_range"]),
        adaptive_stop=int(cfg["link"]["adaptive_stop"]),
        adaptive_step=float(cfg["link"]["adaptive_step"]),
        memory=cfg["link"].get("memory", None),
        filter_stubs=int(cfg["post"]["filter_stubs"]),
    )


def load_video_to_ram(video_path: Path, channel: int) -> tuple[np.ndarray, float, float]:
    v = pims.Video(str(video_path))
    fps = float(v.frame_rate)
    duration = float(v.duration)
    nFrames = int(fps * duration - 1)

    H, W = v[0].shape[0], v[0].shape[1]
    frames = np.zeros((nFrames, H, W), dtype=np.float32)
    for i in range(nFrames):
        frames[i] = v.get_frame(i)[:, :, channel]
    return frames, fps, duration


def subtract_background(frames: np.ndarray, method: str) -> np.ndarray:
    if method == "mean":
        bg = frames.mean(axis=0)
    elif method == "median":
        bg = np.median(frames, axis=0)
    else:
        raise ValueError("background method must be 'mean' or 'median'")
    return frames - bg


def run_locate_to_h5(frames: np.ndarray, h5_path: Path, cfg: TrackingConfig) -> None:
    tp.quiet()
    with tp.PandasHDFStoreBig(str(h5_path)) as s:
        tp.batch(
            frames,
            cfg.diameter,
            minmass=cfg.minmass,
            invert=cfg.invert,
            processes="auto",
            engine="numba",
            output=s,
        )


def run_link_in_h5(h5_path: Path, cfg: TrackingConfig, nFrames: int) -> None:
    memory = cfg.memory
    if memory is None:
        memory = nFrames - 1

    with tp.PandasHDFStoreBig(str(h5_path)) as s:
        for linked in tp.link_df_iter(
            s,
            search_range=cfg.search_range,
            adaptive_stop=cfg.adaptive_stop,
            adaptive_step=cfg.adaptive_step,
            memory=memory,
        ):
            s.put(linked)


def load_linked_df(h5_path: Path) -> pd.DataFrame:
    with tp.PandasHDFStoreBig(str(h5_path)) as s:
        return pd.concat(iter(s), ignore_index=True)


def save_outputs(df: pd.DataFrame, out_base: Path, filter_stubs: int) -> None:
    df_f = tp.filter_stubs(df, filter_stubs)

    pkl_path = out_base.with_suffix(".pkl")
    df_f.to_pickle(pkl_path)

    # Export txt compatibile col resto della pipeline
    txt_path = out_base.with_suffix(".txt")
    header = ["y", "x", "particle"]
    df_f.to_csv(txt_path, index=True, columns=header)