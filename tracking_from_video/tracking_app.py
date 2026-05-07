"""
Streamlit app — interactive tuning + batch tracking launcher.
Run with:
    cd tracking_from_video
    streamlit run tracking_app.py
"""
from __future__ import annotations

import io
import json
import subprocess
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import streamlit as st

# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
st.set_page_config(page_title="Tracking Tuning", page_icon="🎯", layout="wide")
st.title("🎯 Tracking — Tuning parametri")

MACHINE_GLOB = {
    "microscope2D":  "tAlgae*.avi",
    "photonicsLab":  "tPhot*.avi",
    "lightfield":    "td*.avi",
}
machine = st.radio("Macchina", list(MACHINE_GLOB.keys()), horizontal=True)
glob_pattern = MACHINE_GLOB[machine]

CONFIG_PATH = Path(__file__).parent / "configs" / "config_tracking.json"


def _load_saved_config() -> dict:
    if CONFIG_PATH.exists():
        try:
            with open(CONFIG_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            pass
    return {}


_cfg = _load_saved_config()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
@st.cache_data(show_spinner="Caricamento video in RAM...")
def _load_frames(video_path: str, channel: int) -> np.ndarray:
    from src.tracking_core import load_video_to_ram
    frames, _, _ = load_video_to_ram(Path(video_path), channel)
    return frames


@st.cache_data(show_spinner="Sottrazione background...")
def _subtract_bg(video_path: str, channel: int, method: str) -> np.ndarray:
    frames = _load_frames(video_path, channel)
    from src.tracking_core import subtract_background
    return subtract_background(frames, method)


def _get_frames(video_path: str, channel: int, bg_sub: bool, bg_method: str) -> np.ndarray:
    if bg_sub:
        return _subtract_bg(video_path, channel, bg_method)
    return _load_frames(video_path, channel)


# ---------------------------------------------------------------------------
# 1. Video input
# ---------------------------------------------------------------------------
st.header("1. Video")
video_path = st.text_input(
    "Path al video (.avi)",
    value=_cfg.get("video_example", ""),
    placeholder="/home/utente/dati/tAlgae68.avi",
)

channel = st.selectbox("Canale colore", options=[0, 1, 2],
                        index=_cfg.get("channel", 2),
                        help="0=R, 1=G, 2=B")

# ---------------------------------------------------------------------------
# 2. Background
# ---------------------------------------------------------------------------
st.header("2. Background")
bg_sub = st.checkbox("Sottrai background", value=_cfg.get("background_subtraction", True))
_saved_method = _cfg.get("background_method", "mean")
bg_method = st.radio("Metodo", ["median", "mean"], horizontal=True,
                     index=0 if _saved_method == "median" else 1,
                     disabled=not bg_sub)

# ---------------------------------------------------------------------------
# 3. Parametri locate
# ---------------------------------------------------------------------------
st.header("3. Tuning `tp.locate`")

col1, col2, col3 = st.columns(3)
with col1:
    diameter = st.slider("diameter (px, dispari)", min_value=3, max_value=21,
                         value=_cfg.get("locate", {}).get("diameter", 5), step=2)
with col2:
    minmass = st.slider("minmass", min_value=10, max_value=500,
                        value=_cfg.get("locate", {}).get("minmass", 120), step=10)
with col3:
    invert = st.checkbox("invert", value=_cfg.get("locate", {}).get("invert", False))

# ---------------------------------------------------------------------------
# 4. Preview su frame singolo
# ---------------------------------------------------------------------------
st.header("4. Preview")

if not video_path.strip():
    st.info("Inserisci il path al video per continuare.")
    st.stop()

video_path = video_path.strip()
if not Path(video_path).exists():
    st.error(f"File non trovato: {video_path}")
    st.stop()

try:
    frames = _get_frames(video_path, channel, bg_sub, bg_method if bg_sub else "median")
except Exception:
    st.error("Errore nel caricamento del video:")
    st.code(traceback.format_exc())
    st.stop()

nFrames = frames.shape[0]
frame_idx = st.slider("Frame da visualizzare", min_value=0, max_value=nFrames - 1,
                      value=nFrames - 1, step=1)

col_prev, col_grid = st.columns([1, 2])

with col_prev:
    st.subheader("Frame + rilevamento attuale")
    try:
        import trackpy as tp
        tp.quiet()
        feats = tp.locate(frames[frame_idx], diameter, minmass=minmass, invert=invert)
        fig, ax = plt.subplots(figsize=(5, 4), dpi=120)
        ax.imshow(frames[frame_idx], cmap="gray")
        tp.annotate(feats, frames[frame_idx], ax=ax)
        ax.set_title(f"frame {frame_idx} | n={len(feats)}")
        ax.axis("off")
        fig.tight_layout()
        st.pyplot(fig)
        plt.close(fig)
    except Exception:
        st.error("Errore in tp.locate:")
        st.code(traceback.format_exc())

with col_grid:
    st.subheader("Confronto minmass")
    minmass_values = st.multiselect(
        "minmass da confrontare",
        options=[50, 80, 100, 120, 150, 200, 300],
        default=[80, 120, 150, 200],
    )
    if minmass_values and st.button("Genera griglia confronto"):
        try:
            n = len(minmass_values)
            cols_grid = 2
            rows_grid = (n + cols_grid - 1) // cols_grid
            fig2, axes = plt.subplots(rows_grid, cols_grid,
                                      figsize=(10, 4 * rows_grid), dpi=120)
            axes = np.array(axes).reshape(-1)
            for ax, mm in zip(axes, minmass_values):
                f = tp.locate(frames[frame_idx], diameter, minmass=mm, invert=invert)
                ax.imshow(frames[frame_idx], cmap="gray")
                tp.annotate(f, frames[frame_idx], ax=ax)
                ax.set_title(f"minmass={mm} | n={len(f)}")
                ax.axis("off")
            for ax in axes[n:]:
                ax.axis("off")
            fig2.tight_layout()
            st.pyplot(fig2)
            plt.close(fig2)
        except Exception:
            st.error("Errore nella griglia:")
            st.code(traceback.format_exc())

# ---------------------------------------------------------------------------
# 5. Parametri link
# ---------------------------------------------------------------------------
st.header("5. Parametri `tp.link`")
col1, col2, col3, col4 = st.columns(4)
_link = _cfg.get("link", {})
with col1:
    search_range = st.number_input("search_range (px)", min_value=1,
                                   value=_link.get("search_range", 30), step=1)
with col2:
    adaptive_stop = st.number_input("adaptive_stop (px)", min_value=1,
                                    value=_link.get("adaptive_stop", 5), step=1)
with col3:
    adaptive_step = st.number_input("adaptive_step", min_value=0.01, max_value=1.0,
                                    value=_link.get("adaptive_step", 0.98),
                                    step=0.01, format="%.2f")
with col4:
    filter_stubs = st.number_input("filter_stubs (frame minimi)", min_value=0,
                                   value=_cfg.get("post", {}).get("filter_stubs", 0), step=1)

# ---------------------------------------------------------------------------
# 6. Salva config
# ---------------------------------------------------------------------------
st.header("6. Salva config")

cfg_dict = {
    "channel": channel,
    "background_subtraction": bg_sub,
    "background_method": bg_method,
    "locate": {
        "diameter": diameter,
        "minmass": minmass,
        "invert": invert,
    },
    "link": {
        "search_range": search_range,
        "adaptive_stop": adaptive_stop,
        "adaptive_step": adaptive_step,
        "memory": None,
    },
    "post": {
        "filter_stubs": filter_stubs,
    },
    "video_example": video_path,
}

col_name, col_btn = st.columns([3, 1])
with col_name:
    config_name = st.text_input(
        "Nome config",
        value="config_tracking",
        help="Salvata in configs/<nome>.json — usa nomi diversi per parametri diversi (es. config_bright, config_dim)",
    )
with col_btn:
    st.write("")
    st.write("")
    if st.button("💾 Salva"):
        save_path = CONFIG_PATH.parent / f"{config_name.strip() or 'config_tracking'}.json"
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            json.dump(cfg_dict, f, indent=2)
        st.success(f"Salvata: `{save_path.name}`")
        with st.expander("Contenuto"):
            st.json(cfg_dict)

# ---------------------------------------------------------------------------
# 7. Avvia tracking batch
# ---------------------------------------------------------------------------
st.header("7. Avvia tracking batch")

CONFIGS_DIR = CONFIG_PATH.parent
available_configs = sorted(CONFIGS_DIR.glob("*.json"))

if not available_configs:
    st.warning("Nessuna config trovata in `configs/`. Salvane almeno una nella sezione 6.")
    st.stop()

video_dir = st.text_input(
    f"Cartella video ({glob_pattern})",
    value=str(Path(video_path).parent) if video_path.strip() else "",
    placeholder="/home/utente/nas_public/tAlgae_exp",
)

available_videos = []
if video_dir.strip() and Path(video_dir.strip()).is_dir():
    available_videos = sorted(Path(video_dir.strip()).glob(glob_pattern))
    if not available_videos:
        st.warning(f"Nessun file {glob_pattern} trovato nella cartella.")
else:
    st.info("Inserisci una cartella valida per vedere i video disponibili.")

out_dir = st.text_input(
    "Cartella output tracking",
    placeholder="/home/utente/dati/risultati_tracking",
)
overwrite = st.checkbox("Ricalcola anche se output già esiste", value=False)

# Assegnazione video → config
st.subheader("Assegna video ai gruppi di config")
st.caption("Per ogni config disponibile seleziona i video da processare con quei parametri.")

groups: dict[str, list[str]] = {}
assigned: set[str] = set()

for cfg_file in available_configs:
    with st.expander(f"📄 {cfg_file.name}", expanded=False):
        try:
            with open(cfg_file) as f:
                c = json.load(f)
            st.caption(
                f"diameter={c.get('locate',{}).get('diameter','?')}  "
                f"minmass={c.get('locate',{}).get('minmass','?')}  "
                f"bg={c.get('background_method','?')}"
            )
        except Exception:
            pass
        opts = [v.name for v in available_videos if v.name not in assigned]
        sel = st.multiselect(
            "Video da processare con questa config",
            options=[v.name for v in available_videos],
            default=[],
            key=f"grp_{cfg_file.name}",
        )
        if sel:
            groups[str(cfg_file)] = sel
            assigned.update(sel)

if st.button("▶ Avvia tutti i gruppi", type="primary"):
    if not out_dir.strip():
        st.error("Inserisci la cartella di output.")
    elif not groups:
        st.error("Assegna almeno un video a una config.")
    else:
        script = Path(__file__).parent / "src" / "tracking_algae.py"
        for cfg_path_str, files in groups.items():
            st.write(f"**{Path(cfg_path_str).name}** → {', '.join(files)}")
            cmd = [
                sys.executable, str(script),
                "--input", video_dir.strip(),
                "--config", cfg_path_str,
                "--out", out_dir.strip(),
                "--files", *files,
            ]
            if overwrite:
                cmd.append("--overwrite")
            with st.spinner(f"Tracking {Path(cfg_path_str).name} ({len(files)} video)..."):
                try:
                    result = subprocess.run(
                        cmd, capture_output=True, text=True,
                        cwd=str(Path(__file__).parent),
                    )
                    if result.returncode == 0:
                        st.success(f"✅ {Path(cfg_path_str).name}: {len(files)} video completati")
                    else:
                        st.error(f"❌ Errore in {Path(cfg_path_str).name}")
                    if result.stdout:
                        with st.expander("stdout"):
                            st.text(result.stdout)
                    if result.stderr:
                        with st.expander("stderr"):
                            st.text(result.stderr)
                except Exception:
                    st.error("Errore nel lancio:")
                    st.code(traceback.format_exc())
