"""
Streamlit app — clam_pipeline analysis launcher.
Run with:  streamlit run app.py
"""
from __future__ import annotations

import sys
import io
import traceback
from contextlib import redirect_stdout
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import streamlit as st


def _fig_to_png_bytes(fig, dpi: int = 250) -> bytes:
    """Render a matplotlib figure to PNG bytes at the given DPI."""
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi, bbox_inches="tight")
    buf.seek(0)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Page config
# ---------------------------------------------------------------------------
def _show_plots(info: dict) -> None:
    from analyses.plot_theta_distribution import plot_theta_distribution
    from analyses.plot_flux import plot_flux
    from analyses.plot_nt_stats import plot_nt_stats
    from analyses.plot_angle_distribution import plot_angle_distribution
    from analyses.plot_omega_distribution import plot_omega_distribution
    from analyses.plot_density_vs_distance import plot_density_vs_distance
    from analyses.plot_tumbling import plot_tumbling_vs_gradient, plot_tumbling_duration_vs_distance
    from analyses.plot_theta_y_distribution import plot_theta_y_distribution

    rd = info["flat_results_dir"]
    sp = info["settings_txt"]
    illum = info["illum_dir"]
    eids = info["exp_ids"]
    disc = int(info["disc_radius"])
    thr = info["tumbling_threshold"]
    t_inner = info.get("theta_inner_circle", 300.0)
    t_thick = info.get("theta_thickness", 0.0)
    t_maxd  = info.get("theta_max_distance", 800.0)
    machine = info.get("machine", "microscope2D")

    tab_names = [
        "Theta", "Theta Y (legacy)", "Theta X", "Theta Y",
        "Flux", "N(t)", "Angle", "Omega",
        "Density vs dist", "Tumbling θ", "Tumbling duration",
    ]
    tabs = st.tabs(tab_names)

    plots = [
        ("Theta", lambda: plot_theta_distribution(
            results_dir=rd, settings_path=sp, exp_ids=eids,
            inner_circle=t_inner, thickness_circle=t_thick,
            max_distance=t_maxd, save_path=None)),
        ("Theta Y (legacy)", lambda: plot_theta_y_distribution(
            results_dir=rd, exp_ids=eids,
            inner_circle=t_inner, thickness_circle=t_thick,
            max_distance=t_maxd, axis="y", save_path=None)),
        ("Theta X", lambda: plot_theta_y_distribution(
            results_dir=rd, exp_ids=eids,
            inner_circle=t_inner, thickness_circle=t_thick,
            max_distance=t_maxd, axis="x", save_path=None)),
        ("Theta Y", lambda: plot_theta_y_distribution(
            results_dir=rd, exp_ids=eids,
            inner_circle=t_inner, thickness_circle=t_thick,
            max_distance=t_maxd, axis="y_only", save_path=None)),
        ("Flux", lambda: plot_flux(
            results_dir=rd, settings_path=sp, exp_ids=eids, disc_radius=disc, save_path=None)),
        ("N(t)", lambda: plot_nt_stats(
            results_dir=rd, exp_ids=eids, disc_radius=disc, save_path=None)),
        ("Angle", lambda: plot_angle_distribution(
            results_dir=rd, exp_ids=eids, save_path=None)),
        ("Omega", lambda: plot_omega_distribution(
            results_dir=rd, exp_ids=eids, save_path=None)),
        ("Density vs dist", lambda: plot_density_vs_distance(
            results_dir=rd, settings_path=sp, illum_dir=illum, exp_ids=eids, save_path=None)),
        ("Tumbling θ", lambda: plot_tumbling_vs_gradient(
            results_dir=rd, exp_ids=eids, tumbling_threshold=thr, save_path=None)),
        ("Tumbling duration", lambda: plot_tumbling_duration_vs_distance(
            results_dir=rd, exp_ids=eids, tumbling_threshold=thr, save_path=None)),
    ]

    for tab, (name, fn) in zip(tabs, plots):
        with tab:
            try:
                fig, _ = fn()
                st.pyplot(fig)
                png_bytes = _fig_to_png_bytes(fig, dpi=250)
                st.download_button(
                    label="⬇️ Scarica PNG (250 dpi)",
                    data=png_bytes,
                    file_name=f"{name.replace(' ', '_').replace('(', '').replace(')', '')}.png",
                    mime="image/png",
                    key=f"download_{name}",
                )
            except Exception as e:
                st.warning(f"Plot non disponibile: {e}")

    # --- PhotonicsLab: per-experiment polar preview ---
    if machine == "photonicsLab":
        st.subheader("📡 Polar diagrams — PhotonicsLab preview")

        run_dir = info.get("run_dir")

        # --- helper ---
        def _load_polar(csv_path: Path, col: int = 0, mirror: bool = False, wrap: bool = False) -> np.ndarray | None:
            """Load a column from a CSV and return finite values.
            wrap=True  → converts [-π,π] to [0,2π] (for arctan2 angle)
            mirror=True → reflects [0,π] to full circle [0,2π] (for arccos theta)"""
            if not csv_path.exists():
                return None
            try:
                df = pd.read_csv(csv_path)
                vals = df.iloc[:, col].dropna().to_numpy(dtype=float)
                vals = vals[np.isfinite(vals)]
                if vals.size == 0:
                    return None
                if wrap:
                    vals = np.where(vals < 0, vals + 2 * np.pi, vals)
                if mirror:
                    vals = np.concatenate([vals, 2 * np.pi - vals])
                return vals
            except Exception:
                return None

        def _polar_bar(ax, data: np.ndarray, bins: int = 36, color="steelblue"):
            counts, edges = np.histogram(data, bins=bins, range=(0, 2 * np.pi), density=True)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.bar(centers, counts, width=2 * np.pi / bins, alpha=0.6,
                   align="center", color=color)
            ax.set_theta_zero_location("N")
            ax.set_theta_direction(-1)
            ax.tick_params(labelsize=6)

        metrics = [
            ("angle",   "angle.csv",   False, True,  "steelblue",   r"angle — arctan2(Vy,Vx)"),
            ("theta_x", "theta_x.csv", True,  False, "darkorange",  r"θₓ = arccos(Vx/|V|)"),
            ("theta_y", "theta_y.csv", True,  False, "seagreen",    r"θᵧ = arccos(Vy/|V|)"),
        ]

        n_exp = len(eids)

        # --- preload all data once, used both for the grid and the combined figure ---
        loaded = {}
        for metric_key, fname, mirror, wrap, color, row_label in metrics:
            loaded[metric_key] = {}
            for exp_id in eids:
                data = None
                if run_dir is not None:
                    csv_path = run_dir / f"exp_{exp_id:04d}" / fname
                    data = _load_polar(csv_path, col=0, mirror=mirror, wrap=wrap)
                loaded[metric_key][exp_id] = data

        # --- combined single-image figure (all rows x all experiments) ---
        n_rows = len(metrics)
        fig_combined, axs_combined = plt.subplots(
            n_rows, n_exp,
            subplot_kw={"projection": "polar"},
            figsize=(2.6 * n_exp, 2.6 * n_rows),
            dpi=120,
        )
        if n_exp == 1:
            axs_combined = axs_combined.reshape(n_rows, 1)
        if n_rows == 1:
            axs_combined = axs_combined.reshape(1, n_exp)

        for row_idx, (metric_key, fname, mirror, wrap, color, row_label) in enumerate(metrics):
            for col_idx, exp_id in enumerate(eids):
                ax_c = axs_combined[row_idx, col_idx]
                data = loaded[metric_key][exp_id]
                if data is not None:
                    _polar_bar(ax_c, data, color=color)
                else:
                    ax_c.text(0.5, 0.5, "no data", transform=ax_c.transAxes,
                              ha="center", va="center", fontsize=7)
                if row_idx == 0:
                    ax_c.set_title(f"exp {exp_id}", fontsize=9, pad=10)
                if col_idx == 0:
                    ax_c.text(-0.3, 0.5, row_label, transform=ax_c.transAxes,
                               rotation=90, va="center", ha="center", fontsize=9)

        fig_combined.suptitle("Polar diagrams — angle / θₓ / θᵧ", fontsize=12)
        fig_combined.tight_layout()

        combined_png = _fig_to_png_bytes(fig_combined, dpi=250)
        st.download_button(
            label="⬇️ Scarica griglia completa (PNG, 250 dpi)",
            data=combined_png,
            file_name="polar_diagrams_combined.png",
            mime="image/png",
            key="download_polar_combined",
        )
        plt.close(fig_combined)

        st.divider()

        for metric_key, fname, mirror, wrap, color, row_label in metrics:
            st.caption(f"**{row_label}**")
            cols = st.columns(n_exp)
            for col_ui, exp_id in zip(cols, eids):
                with col_ui:
                    data = loaded[metric_key][exp_id]

                    if data is not None:
                        fig_p, ax_p = plt.subplots(
                            subplot_kw={"projection": "polar"},
                            figsize=(2.5, 2.5), dpi=120,
                        )
                        _polar_bar(ax_p, data, color=color)
                        ax_p.set_title(f"exp {exp_id}", fontsize=8, pad=4)
                        st.pyplot(fig_p)
                        png_bytes = _fig_to_png_bytes(fig_p, dpi=250)
                        st.download_button(
                            label="⬇️ PNG",
                            data=png_bytes,
                            file_name=f"{metric_key}_exp{exp_id}.png",
                            mime="image/png",
                            key=f"download_{metric_key}_{exp_id}",
                        )
                        plt.close(fig_p)
                    else:
                        st.info(f"exp {exp_id}\nno data")


st.set_page_config(page_title="Clam Pipeline", page_icon="🔬", layout="centered")
st.title("🔬 Clam Pipeline — Analisi traiettorie")

# ---------------------------------------------------------------------------
# 1. Macchina
# ---------------------------------------------------------------------------
st.header("1. Macchina")
machine = st.radio("Quale macchina è stata usata?", ["microscope2D", "photonicsLab", "lightfield"], horizontal=True)

# ---------------------------------------------------------------------------
# 2. Percorsi
# ---------------------------------------------------------------------------
st.header("2. Percorsi")

col1, col2 = st.columns(2)
with col1:
    traj_dir = st.text_input(
        "Cartella traiettorie (tAlgae*.txt)",
        placeholder="/home/utente/dati/traj",
    )
    illum_dir = st.text_input(
        "Cartella illuminazione (illumination*.tiff)",
        placeholder="/home/utente/dati/illuminazione",
    )
with col2:
    settings_txt = st.text_input(
        "File settings (.txt)",
        placeholder="/home/utente/dati/Clamidomoni_settings.txt",
    )
    radial_dir = st.text_input(
        "Cartella mappa radiale (opzionale se radial_map=False)",
        placeholder="/home/utente/dati/radial",
    )

results_dir = st.text_input(
    "Cartella output risultati",
    placeholder="/home/utente/risultati",
)

exp_ids_input = st.text_input(
    "Experiment ID (uno o più, separati da virgola)",
    value="68",
    help="Es: 68   oppure   68, 69, 73  per analizzarli insieme e ottenere anche il merge",
)

# ---------------------------------------------------------------------------
# 3. Parametri
# ---------------------------------------------------------------------------
with st.expander("Parametri analisi (AnalysisParams)"):
    col1, col2 = st.columns(2)
    with col1:
        diam            = st.number_input("diam (µm)", value=10.0, step=0.5)
        dt              = st.number_input("dt (s)", value=0.1, step=0.01, format="%.3f")
        disc_radius     = st.number_input("disc_radius (µm)", value=350.0, step=10.0)
        tumbling_thr    = st.number_input("tumbling_threshold (°/s)", value=5.0, step=0.5)
    with col2:
        flux_cg_in      = st.number_input("flux_cg_in", value=1, step=1)
        flux_cg_out     = st.number_input("flux_cg_out (frames)", value=150, step=10)
        nskip           = st.number_input("nskip", value=1, step=1)
        radial_map      = st.checkbox("Usa mappa radiale (radial_map)", value=True)

with st.expander("Parametri plot Theta P(θ)"):
    col1, col2, col3 = st.columns(3)
    with col1:
        theta_inner_circle  = st.number_input("inner_circle (µm)", value=300.0, step=10.0,
                                              help="Soglia radiale: sotto = verde (dentro), sopra = giallo (fuori)")
    with col2:
        theta_thickness     = st.number_input("thickness_circle (µm)", value=0.0, step=5.0,
                                              help="Spessore zona anulare da escludere attorno a inner_circle")
    with col3:
        theta_max_distance  = st.number_input("max_distance (µm)", value=800.0, step=10.0,
                                              help="Distanza massima considerata nel subplot giallo")

with st.expander("Filtri traiettorie (FilterParams)"):
    col1, col2 = st.columns(2)
    with col1:
        lifetime_min        = st.number_input("lifetime_min", value=0.0, step=0.1)
        threshold_dist      = st.number_input("threshold_dist", value=1.0, step=0.1)
        threshold_dist_tot  = st.number_input("threshold_dist_tot", value=3.0, step=0.5)
        dist_step           = st.number_input("dist_step", value=1, step=1)
    with col2:
        blinking_ratio      = st.number_input("blinking_ratio", value=0.5, step=0.05, format="%.2f")
        blinking_tolerance  = st.number_input("blinking_tolerance", value=0.1, step=0.05, format="%.2f")
        long_track_min_snaps = st.number_input("long_track_min_snaps", value=20, step=1)
        strict              = st.checkbox("strict", value=True)
        only_inside         = st.checkbox("only_inside", value=True)

# ---------------------------------------------------------------------------
# 4. Avvio
# ---------------------------------------------------------------------------
st.header("4. Avvia")

if st.button("▶ Avvia analisi", type="primary"):

    # Parsing exp_ids
    try:
        exp_ids_list = [int(x.strip()) for x in exp_ids_input.split(",") if x.strip()]
    except ValueError:
        st.error("Experiment ID non valido. Inserisci numeri interi separati da virgola (es: 68, 69).")
        st.stop()
    if not exp_ids_list:
        st.error("Inserisci almeno un Experiment ID.")
        st.stop()

    # Validazione percorsi obbligatori
    missing = []
    for label, val in [
        ("Cartella traiettorie", traj_dir),
        ("Cartella illuminazione", illum_dir),
        ("File settings", settings_txt),
        ("Cartella output", results_dir),
    ]:
        if not val.strip():
            missing.append(label)
    if missing:
        st.error("Campi obbligatori mancanti: " + ", ".join(missing))
        st.stop()

    # Aggiunge prefisso macchina alla cartella runs
    results_path = Path(results_dir.strip()) / machine

    # Importa il modulo (l'utente deve avere il package nel PYTHONPATH)
    try:
        from clam_pipeline.analyze import AnalysisParams, FilterParams
        from clam_pipeline.cli import main as run_analysis
    except ImportError as e:
        st.error(f"Impossibile importare clam_pipeline: {e}\n\nAssicurati di lanciare l'app dalla cartella del progetto.")
        st.stop()

    params = AnalysisParams(
        diam=float(diam),
        dt=float(dt),
        particle_show=5,
        disc_radius=float(disc_radius),
        tumbling_threshold=float(tumbling_thr),
        radial_map=bool(radial_map),
        flux_cg_in=int(flux_cg_in),
        flux_cg_out=int(flux_cg_out),
        nskip=int(nskip),
    )
    fparams = FilterParams(
        lifetime_min=float(lifetime_min),
        threshold_dist=float(threshold_dist),
        threshold_dist_tot=float(threshold_dist_tot),
        dist_step=int(dist_step),
        strict=bool(strict),
        only_inside=bool(only_inside),
        blinking_ratio=float(blinking_ratio),
        blinking_tolerance=float(blinking_tolerance),
        long_track_min_snaps=int(long_track_min_snaps),
    )

    # Calcola run_dir in anticipo per mostrarlo
    run_id = datetime.now().strftime("%Y-%m-%d_%H%M%S")
    run_dir = results_path / "runs" / run_id

    plot_info = None
    with st.spinner("Analisi in corso..."):
        log_buffer = io.StringIO()
        try:
            with redirect_stdout(log_buffer):
                plot_info = run_analysis(
                    code_dir=Path(__file__).parent,
                    traj_dir=Path(traj_dir.strip()),
                    settings_txt=Path(settings_txt.strip()),
                    illum_dir=Path(illum_dir.strip()),
                    radial_dir=Path(radial_dir.strip()) if radial_dir.strip() else Path("."),
                    results_dir=results_path,
                    exp_ids=exp_ids_list,
                    params=params,
                    fparams=fparams,
                )
            ids_str = ", ".join(str(i) for i in plot_info["exp_ids"])
            merge_note = "\n\n📁 Merge disponibile in: `merged/`" if len(plot_info["exp_ids"]) > 1 else ""
            st.success(f"✅ Analisi completata! (exp: {ids_str})\n\n**Output salvato in:** `{plot_info['run_dir']}`{merge_note}")
        except Exception:
            st.error("❌ Errore durante l'analisi:")
            st.code(traceback.format_exc())

    log_output = log_buffer.getvalue()
    if log_output.strip():
        with st.expander("Log"):
            st.text(log_output)

    if plot_info is not None:
        plot_info["theta_inner_circle"] = float(theta_inner_circle)
        plot_info["theta_thickness"]    = float(theta_thickness)
        plot_info["theta_max_distance"] = float(theta_max_distance)
        plot_info["machine"]            = machine
        st.header("5. Risultati")
        _show_plots(plot_info)

