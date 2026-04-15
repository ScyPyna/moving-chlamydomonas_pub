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

import streamlit as st
import matplotlib
matplotlib.use("Agg")

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

    rd = info["flat_results_dir"]
    sp = info["settings_txt"]
    illum = info["illum_dir"]
    eid = info["exp_ids"][0]
    disc = int(info["disc_radius"])
    thr = info["tumbling_threshold"]

    tabs = st.tabs([
        "Theta", "Flux", "N(t)", "Angle", "Omega",
        "Density vs dist", "Tumbling θ", "Tumbling duration",
    ])

    plots = [
        ("Theta", lambda: plot_theta_distribution(
            results_dir=rd, settings_path=sp, exp_ids=[eid], save_path=None)),
        ("Flux", lambda: plot_flux(
            results_dir=rd, settings_path=sp, exp_ids=[eid], disc_radius=disc, save_path=None)),
        ("N(t)", lambda: plot_nt_stats(
            results_dir=rd, exp_ids=[eid], disc_radius=disc, save_path=None)),
        ("Angle", lambda: plot_angle_distribution(
            results_dir=rd, exp_ids=[eid], save_path=None)),
        ("Omega", lambda: plot_omega_distribution(
            results_dir=rd, exp_ids=[eid], save_path=None)),
        ("Density vs dist", lambda: plot_density_vs_distance(
            results_dir=rd, settings_path=sp, illum_dir=illum, exp_ids=[eid], save_path=None)),
        ("Tumbling θ", lambda: plot_tumbling_vs_gradient(
            results_dir=rd, exp_ids=[eid], save_path=None)),
        ("Tumbling duration", lambda: plot_tumbling_duration_vs_distance(
            results_dir=rd, exp_ids=[eid], save_path=None)),
    ]

    for tab, (name, fn) in zip(tabs, plots):
        with tab:
            try:
                fig, _ = fn()
                st.pyplot(fig)
            except Exception as e:
                st.warning(f"Plot non disponibile: {e}")


st.set_page_config(page_title="Clam Pipeline", page_icon="🔬", layout="centered")
st.title("🔬 Clam Pipeline — Analisi traiettorie")

# ---------------------------------------------------------------------------
# 1. Macchina
# ---------------------------------------------------------------------------
st.header("1. Macchina")
machine = st.radio("Quale macchina è stata usata?", ["superK", "laserGlow"], horizontal=True)

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
        st.header("5. Risultati")
        _show_plots(plot_info)
