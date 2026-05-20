from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import matplotlib.image as mpimg
from scipy.interpolate import interp1d
from scipy.spatial.distance import cdist

import clamutils_exp as cu


@dataclass
class AnalysisParams:
    diam: float = 10
    dt: float = 0.1
    particle_show: int = 5
    disc_radius: float = 350
    flux_cg_in: int = 1
    flux_cg_out: int = 150
    tumbling_threshold: float = 5.0
    radial_map: bool = True
    nskip: int = 1


@dataclass
class FilterParams:
    lifetime_min: float = 0
    threshold_dist: float = 1
    threshold_dist_tot: float = 3
    dist_step: int = 1
    strict: bool = True
    only_inside: bool = True
    blinking_ratio: float = 0.5
    blinking_tolerance: float = 0.0
    long_track_min_snaps: int = 0


@dataclass
class AnalysisOutputs:
    theta: pd.DataFrame
    theta_y: pd.DataFrame
    flux: pd.DataFrame
    angle: pd.DataFrame
    omega: pd.DataFrame
    tumbling_orientation: pd.DataFrame
    tumbling_duration: pd.DataFrame
    nt_stats: pd.DataFrame


def _load_image_as_float(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing image: {path}")
    return mpimg.imread(path).astype(float)


def _load_illumination(illum_dir: Path, ill_id: int) -> np.ndarray:
    fname = "illumination1.tiff" if ill_id == 0 else f"illumination{ill_id}.tiff"
    return _load_image_as_float(illum_dir / fname)


def _radialize_map_to_speckle(
    speckle: cu.Speckle,
    diam: float,
    pix: float,
    ctr: int,
    lin: int,
    axis: int,
    intensity_uW: int,
    radial_dir: Path,
) -> cu.Speckle:
    fname = f"radialEnergyDensityMap_{intensity_uW}µW.txt"
    fpath = radial_dir / fname
    if not fpath.exists():
        raise FileNotFoundError(
            f"Missing radial map file: {fpath}\n"
            f"radial_map=True requires this file. Otherwise set radial_map=False."
        )

    radial_int = np.genfromtxt(fpath, skip_header=1)
    if radial_int.ndim != 1 or len(radial_int) < 2:
        raise ValueError(f"Radial map {fpath} is not a valid 1D vector.")

    r_grid = np.arange(len(radial_int)) * pix
    int_converter = interp1d(
        r_grid,
        radial_int,
        kind="linear",
        bounds_error=False,
        fill_value=float(radial_int[-1]),
    )

    xg, yg = speckle.x_grid, speckle.y_grid
    dist = np.sqrt((xg - speckle.center[0]) ** 2 + (yg - speckle.center[1]) ** 2)
    ill_map = int_converter(dist)

    return cu.Speckle(ill_map, diam, pix, ctrp=ctr, lin=lin, axis=axis)


def _build_fluo_field(
    speckle: cu.Speckle,
    diam: float,
    pix: float,
    ctr: int,
    lin: int,
    axis: int,
    intensity_uW: int,
    radial_map: bool,
    radial_dir: Path,
    fluo_flag: int,
) -> cu.Speckle:
    if radial_map:
        return _radialize_map_to_speckle(
            speckle=speckle,
            diam=diam,
            pix=pix,
            ctr=ctr,
            lin=lin,
            axis=axis,
            intensity_uW=intensity_uW,
            radial_dir=radial_dir,
        )

    if fluo_flag == 0:
        xg, yg = speckle.x_grid, speckle.y_grid
        r = np.sqrt((xg - speckle.center[0]) ** 2 + (yg - speckle.center[1]) ** 2)
        return cu.Speckle(-r, diam, pix, ctrp=ctr, lin=lin, axis=axis)

    raise RuntimeError(
        "fluo_flag != 0 but no real fluo image naming convention is implemented.\n"
        "Use radial_map=True or extend _build_fluo_field()."
    )


def analyze_experiment(
    exp_id: int,
    settings_row: pd.Series,
    traj_df: pd.DataFrame,
    illum_dir: Path,
    radial_dir: Path,
    params: AnalysisParams,
    fparams: FilterParams,
) -> AnalysisOutputs:
    if traj_df.empty:
        return AnalysisOutputs(
            theta=pd.DataFrame(columns=["theta", "dist"]),
            theta_y=pd.DataFrame(columns=["theta_y", "dist"]),
            flux=pd.DataFrame(columns=["time", "flux"]),
            angle=pd.DataFrame(columns=["angle", "dist", "v"]),
            omega=pd.DataFrame(columns=["omega", "dist", "time"]),
            tumbling_orientation=pd.DataFrame(columns=["theta", "dist", "duration", "time"]),
            tumbling_duration=pd.DataFrame(columns=["dist", "duration", "time"]),
            nt_stats=pd.DataFrame(columns=["N", "n_disc", "dists"]),
        )

    pix = float(settings_row["pix_arr"])
    ill_id = int(settings_row["ill"])
    ctr = int(settings_row["ctr"])
    lin = int(settings_row.get("lin", 0))
    axis = int(settings_row.get("axis", 0))
    intensity_uW = int(settings_row["intensity"])
    fluo_flag = int(settings_row.get("fluo", 0))

    field = _load_illumination(illum_dir, ill_id)
    speckle = cu.Speckle(field, params.diam, pix, ctrp=ctr, lin=lin, axis=axis)

    fluo_field = _build_fluo_field(
        speckle=speckle,
        diam=params.diam,
        pix=pix,
        ctr=ctr,
        lin=lin,
        axis=axis,
        intensity_uW=intensity_uW,
        radial_map=params.radial_map,
        radial_dir=radial_dir,
        fluo_flag=fluo_flag,
    )

    # filt = cu.Filters(
    #     lifetime_min=fparams.lifetime_min,
    #     threshold_dist=fparams.threshold_dist,
    #     threshold_dist_tot=fparams.threshold_dist_tot,
    #     dist_step=fparams.dist_step,
    #     strict=fparams.strict,
    #     only_inside=fparams.only_inside,
    #     blinking_ratio=fparams.blinking_ratio,
    # )

    filt = cu.Filters(
        lifetime_min=fparams.lifetime_min,
        threshold_dist=fparams.threshold_dist,
        threshold_dist_tot=fparams.threshold_dist_tot,
        dist_step=fparams.dist_step,
        strict=fparams.strict,
        only_inside=fparams.only_inside,
        blinking_ratio=fparams.blinking_ratio,

        # NUOVI PARAMETRI
        blinking_tolerance=getattr(fparams, "blinking_tolerance", 0.0),
        long_track_min_snaps=getattr(fparams, "long_track_min_snaps", 0),
    )

    Xtot = traj_df["x_px"].to_numpy(dtype=float) * pix
    Ytot = traj_df["y_px"].to_numpy(dtype=float) * pix
    snaps = traj_df["snap"].to_numpy(dtype=int)
    tags = traj_df["tag"].to_numpy(dtype=int)

    unique_tags = np.unique(tags)
    unique_times = np.sort(np.unique(snaps)) * params.dt

    all_theta = []
    all_theta_dist = []

    all_theta_y = []
    all_theta_y_dist = []

    all_fluxes = []
    all_flux_times = []

    all_angle = []
    all_angle_dist = []
    all_v = []

    all_omega = []
    all_omega_dist = []
    all_omega_time = []

    all_tumb_theta = []
    all_tumb_dist = []
    all_tumb_duration = []
    all_tumb_time = []

    all_untumb_dist = []
    all_untumb_duration = []
    all_untumb_time = []

    t_new = []
    X_new = []
    Y_new = []
    Tags_new = []

    for tag in unique_tags:
        cond = tags == tag
        Xp = Xtot[cond]
        Yp = Ytot[cond]
        tp = snaps[cond]

        p = cu.particle_class(int(tag), tp, Xp, Yp, speckle, params.dt)
        p.existance(filt)
        p.traj_cleaner(filt)
        if not p.exist:
            continue

        t_new.append(p.snaps)
        X_new.append(p.xtraj)
        Y_new.append(p.ytraj)
        Tags_new.append(np.full(len(p.snaps), p.tag))

        try:
            _, _, theta, dists_theta = p.grad_theta(fluo_field, params.particle_show)
            theta = np.asarray(theta, dtype=float)
            dists_theta = np.asarray(dists_theta, dtype=float)
            mask_theta = np.isfinite(theta) & np.isfinite(dists_theta)
            if np.any(mask_theta):
                all_theta.append(theta[mask_theta])
                all_theta_dist.append(dists_theta[mask_theta])
        except Exception:
            pass

        try:
            ps = params.particle_show
            _inside = (
                (p.ytraj >= 0) & (p.ytraj <= speckle.Ly) &
                (p.xtraj >= 0) & (p.xtraj <= speckle.Lx)
            )
            _Xin = p.xtraj[_inside]
            _Yin = p.ytraj[_inside]
            _Sin = p.snaps[_inside]
            if len(_Xin) > ps:
                _dt = (_Sin[ps:] - _Sin[:-ps]) * params.dt
                _ok = _dt > 0
                _Vy = np.where(_ok, (_Yin[ps:] - _Yin[:-ps]) / np.where(_ok, _dt, 1), np.nan)
                _Vx = np.where(_ok, (_Xin[ps:] - _Xin[:-ps]) / np.where(_ok, _dt, 1), np.nan)
                _V = np.sqrt(_Vx**2 + _Vy**2)
                _dist_y = cu.moving_average(p.dist[_inside], ps + 1)
                _n = min(len(_V), len(_dist_y))
                _V = _V[:_n]; _Vy = _Vy[:_n]; _dist_y = _dist_y[:_n]
                _ty = np.full(_n, np.nan)
                _good = _V > 0
                _ty[_good] = np.arccos(np.clip(_Vy[_good] / _V[_good], -1.0, 1.0))
                _m = np.isfinite(_ty) & np.isfinite(_dist_y)
                if np.any(_m):
                    all_theta_y.append(_ty[_m])
                    all_theta_y_dist.append(_dist_y[_m])
        except Exception:
            pass

        try:
            angle, dists_angle, vh, omegah, dists_omega, times_omega = p.angular_velocity()

            angle = np.asarray(angle, dtype=float)
            dists_angle = np.asarray(dists_angle, dtype=float)
            vh = np.asarray(vh, dtype=float)

            min_len_angle = min(len(angle), len(dists_angle), len(vh))
            if min_len_angle > 0:
                angle = angle[:min_len_angle]
                dists_angle = dists_angle[:min_len_angle]
                vh = vh[:min_len_angle]
                mask_angle = np.isfinite(angle) & np.isfinite(dists_angle) & np.isfinite(vh)
                if np.any(mask_angle):
                    all_angle.append(angle[mask_angle])
                    all_angle_dist.append(dists_angle[mask_angle])
                    all_v.append(vh[mask_angle])

            omegah = np.asarray(omegah, dtype=float)
            dists_omega = np.asarray(dists_omega, dtype=float)
            times_omega = np.asarray(times_omega, dtype=float)
            min_len_omega = min(len(omegah), len(dists_omega), len(times_omega))
            if min_len_omega > 0:
                omegah = omegah[:min_len_omega]
                dists_omega = dists_omega[:min_len_omega]
                times_omega = times_omega[:min_len_omega]
                mask_omega = np.isfinite(omegah) & np.isfinite(dists_omega) & np.isfinite(times_omega)
                if np.any(mask_omega):
                    all_omega.append(omegah[mask_omega])
                    all_omega_dist.append(dists_omega[mask_omega])
                    all_omega_time.append(times_omega[mask_omega])
        except Exception:
            pass

        try:
            tumb_theta, tumb_dist, tumb_duration, tumb_time = p.tumbling_orientation(
                fluo_field, params.tumbling_threshold
            )
            tumb_theta = np.asarray(tumb_theta, dtype=float)
            tumb_dist = np.asarray(tumb_dist, dtype=float)
            tumb_duration = np.asarray(tumb_duration, dtype=float)
            tumb_time = np.asarray(tumb_time, dtype=float)
            min_len_tumb = min(len(tumb_theta), len(tumb_dist), len(tumb_duration), len(tumb_time))
            if min_len_tumb > 0:
                tumb_theta = tumb_theta[:min_len_tumb]
                tumb_dist = tumb_dist[:min_len_tumb]
                tumb_duration = tumb_duration[:min_len_tumb]
                tumb_time = tumb_time[:min_len_tumb]
                mask_tumb = (
                    np.isfinite(tumb_theta)
                    & np.isfinite(tumb_dist)
                    & np.isfinite(tumb_duration)
                    & np.isfinite(tumb_time)
                )
                if np.any(mask_tumb):
                    all_tumb_theta.append(tumb_theta[mask_tumb])
                    all_tumb_dist.append(tumb_dist[mask_tumb])
                    all_tumb_duration.append(tumb_duration[mask_tumb])
                    all_tumb_time.append(tumb_time[mask_tumb])
        except Exception:
            pass

        try:
            untumb_dist, untumb_duration, untumb_time = p.tumbling_duration(
                fluo_field, params.tumbling_threshold
            )
            untumb_dist = np.asarray(untumb_dist, dtype=float)
            untumb_duration = np.asarray(untumb_duration, dtype=float)
            untumb_time = np.asarray(untumb_time, dtype=float)
            min_len_untumb = min(len(untumb_dist), len(untumb_duration), len(untumb_time))
            if min_len_untumb > 0:
                untumb_dist = untumb_dist[:min_len_untumb]
                untumb_duration = untumb_duration[:min_len_untumb]
                untumb_time = untumb_time[:min_len_untumb]
                mask_untumb = (
                    np.isfinite(untumb_dist)
                    & np.isfinite(untumb_duration)
                    & np.isfinite(untumb_time)
                )
                if np.any(mask_untumb):
                    all_untumb_dist.append(untumb_dist[mask_untumb])
                    all_untumb_duration.append(untumb_duration[mask_untumb])
                    all_untumb_time.append(untumb_time[mask_untumb])
        except Exception:
            pass

        try:
            p.flux_calculator(params.disc_radius, params.flux_cg_in)
            flux_particle = np.asarray(p.flux, dtype=float)
            flux_snaps_particle = np.asarray(p.flux_snaps, dtype=float) * params.dt
            min_len_flux = min(len(flux_particle), len(flux_snaps_particle))
            if min_len_flux > 0:
                flux_particle = flux_particle[:min_len_flux]
                flux_snaps_particle = flux_snaps_particle[:min_len_flux]
                mask_flux = np.isfinite(flux_particle) & np.isfinite(flux_snaps_particle)
                if np.any(mask_flux):
                    all_fluxes.append(flux_particle[mask_flux])
                    all_flux_times.append(flux_snaps_particle[mask_flux])
        except Exception:
            pass

    if t_new:
        t_new = np.concatenate(t_new)
        X_new = np.concatenate(X_new)
        Y_new = np.concatenate(Y_new)
        Tags_new = np.concatenate(Tags_new)
    else:
        t_new = np.array([], dtype=float)
        X_new = np.array([], dtype=float)
        Y_new = np.array([], dtype=float)
        Tags_new = np.array([], dtype=float)

    if len(X_new) > 0:
        points = np.column_stack([X_new, Y_new])
        dists_new = cdist(points, [speckle.center])[:, 0]
    else:
        dists_new = np.array([], dtype=float)

    # flux coarse-grained
    if all_fluxes and all_flux_times and len(t_new) > 0:
        all_fluxes_arr = np.concatenate(all_fluxes)
        all_flux_times_arr = np.concatenate(all_flux_times)
        fluxes_times = np.arange(0, np.max(t_new), params.flux_cg_out) * params.dt
        fluxes = np.zeros(len(fluxes_times), dtype=float)
        for i, t0 in enumerate(fluxes_times):
            in_bin = (all_flux_times_arr >= t0) & (all_flux_times_arr < t0 + params.flux_cg_out * params.dt)
            fluxes[i] = np.sum(all_fluxes_arr[in_bin]) / (params.flux_cg_out * params.dt)
        flux_df = pd.DataFrame({"time": fluxes_times, "flux": fluxes})
    else:
        flux_df = pd.DataFrame(columns=["time", "flux"])

    #-----------------------inizio MODIFICA-----------------------------------
    # -------------------------
    # time statistics: raw
    # -------------------------
    points_raw = np.column_stack([Xtot, Ytot])
    dists_raw_all = cdist(points_raw, [speckle.center])[:, 0]

    total_time = len(unique_times) // params.nskip

    N_raw_t = np.zeros(total_time, dtype=int)
    N_disc_raw_t = np.zeros(total_time, dtype=int)
    dists_raw_t = np.full(total_time, np.nan)

    for i in range(total_time):
        snap_value = i * params.nskip
        condition_raw = snaps == snap_value

        Xr = Xtot[condition_raw]
        Yr = Ytot[condition_raw]
        Tags_r = tags[condition_raw]
        dists_r = dists_raw_all[condition_raw]

        if len(Tags_r) > 0:
            # tag unici per frame
            N_raw_t[i] = len(np.unique(Tags_r))

            # tag unici dentro il disco
            tags_disc = Tags_r[dists_r < params.disc_radius]
            N_disc_raw_t[i] = len(np.unique(tags_disc))

            dists_raw_t[i] = np.nanmean(dists_r)

    # -------------------------
    # time statistics: filtered
    # -------------------------
    N_t = np.zeros(total_time, dtype=int)
    N_disc_t = np.zeros(total_time, dtype=int)
    dists_t = np.full(total_time, np.nan)

    for i in range(total_time):
        snap_value = i * params.nskip
        condition_t = t_new == snap_value

        X = X_new[condition_t]
        Y = Y_new[condition_t]
        Tags = Tags_new[condition_t]
        dists = dists_new[condition_t]

        if len(Tags) > 0:
            N_t[i] = len(np.unique(Tags))

            tags_disc = Tags[dists < params.disc_radius]
            N_disc_t[i] = len(np.unique(tags_disc))

            dists_t[i] = np.nanmean(dists)

    nt_df = pd.DataFrame({
        "N_raw": N_raw_t,
        "n_disc_raw": N_disc_raw_t,
        "dists_raw": dists_raw_t,
        "N": N_t,
        "n_disc": N_disc_t,
        "dists": dists_t,
    })

    #-----------------------FINE MODIFICA-----------------------------------

    theta_df = (
        pd.DataFrame({
            "theta": np.concatenate(all_theta),
            "dist": np.concatenate(all_theta_dist),
        })
        if all_theta else pd.DataFrame(columns=["theta", "dist"])
    )

    theta_y_df = (
        pd.DataFrame({
            "theta_y": np.concatenate(all_theta_y),
            "dist": np.concatenate(all_theta_y_dist),
        })
        if all_theta_y else pd.DataFrame(columns=["theta_y", "dist"])
    )

    angle_df = (
        pd.DataFrame({
            "angle": np.concatenate(all_angle),
            "dist": np.concatenate(all_angle_dist),
            "v": np.concatenate(all_v),
        })
        if all_angle else pd.DataFrame(columns=["angle", "dist", "v"])
    )

    omega_df = (
        pd.DataFrame({
            "omega": np.concatenate(all_omega),
            "dist": np.concatenate(all_omega_dist),
            "time": np.concatenate(all_omega_time),
        })
        if all_omega else pd.DataFrame(columns=["omega", "dist", "time"])
    )

    tumb_df = (
        pd.DataFrame({
            "theta": np.concatenate(all_tumb_theta),
            "dist": np.concatenate(all_tumb_dist),
            "duration": np.concatenate(all_tumb_duration),
            "time": np.concatenate(all_tumb_time),
        })
        if all_tumb_theta else pd.DataFrame(columns=["theta", "dist", "duration", "time"])
    )

    tumb_dur_df = (
        pd.DataFrame({
            "dist": np.concatenate(all_untumb_dist),
            "duration": np.concatenate(all_untumb_duration),
            "time": np.concatenate(all_untumb_time),
        })
        if all_untumb_dist else pd.DataFrame(columns=["dist", "duration", "time"])
    )

    return AnalysisOutputs(
        theta=theta_df,
        theta_y=theta_y_df,
        flux=flux_df,
        angle=angle_df,
        omega=omega_df,
        tumbling_orientation=tumb_df,
        tumbling_duration=tumb_dur_df,
        nt_stats=nt_df,
    )