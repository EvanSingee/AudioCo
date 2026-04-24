"""
app.py – AudioCo Streamlit Demo
Run: streamlit run app.py
"""
import io
import json
import logging
import os
import random
import sys
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import streamlit as st
import plotly.graph_objects as go

# ── ensure project root is importable ────────────────────────────────────────
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config

logging.basicConfig(level=logging.WARNING)

# ─────────────────────────────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AudioCo – Speech Separation Pipeline",
    page_icon="🎙️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

/* Dark background */
.stApp { background: #0d1117; color: #e6edf3; }

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #161b22 0%, #0d1117 100%);
    border-right: 1px solid #21262d;
}

/* Cards */
.card {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 12px;
    padding: 1.4rem 1.6rem;
    margin-bottom: 1rem;
}
.card-accent {
    border-left: 4px solid #58a6ff;
}

/* Stage badges */
.stage-badge {
    display: inline-block;
    background: linear-gradient(135deg, #1f6feb, #388bfd);
    color: white;
    border-radius: 20px;
    padding: 3px 14px;
    font-size: 0.78rem;
    font-weight: 600;
    letter-spacing: 0.5px;
}

/* Metric tiles */
.metric-tile {
    background: #161b22;
    border: 1px solid #21262d;
    border-radius: 10px;
    padding: 1rem;
    text-align: center;
}
.metric-label { font-size: 0.75rem; color: #8b949e; margin-bottom: 4px; }
.metric-value { font-size: 1.8rem; font-weight: 700; color: #58a6ff; }
.metric-unit  { font-size: 0.7rem;  color: #8b949e; }

/* Big title */
.hero-title {
    font-size: 2.4rem;
    font-weight: 800;
    background: linear-gradient(90deg, #58a6ff, #a371f7, #ff7b72);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.2;
    margin-bottom: 0.3rem;
}
.hero-sub { color: #8b949e; font-size: 1rem; margin-bottom: 1.5rem; }

/* Progress step */
.step-done  { color: #3fb950; }
.step-run   { color: #e3b341; }
.step-wait  { color: #8b949e; }

/* divider */
hr.soft { border: none; border-top: 1px solid #21262d; margin: 1.2rem 0; }

/* Audio widget background fix */
audio { width: 100%; border-radius: 8px; }

/* Table */
.sdr-table th { background: #161b22; color: #8b949e; }
.sdr-table td { background: #0d1117; color: #e6edf3; }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def audio_bytes(path: str) -> bytes:
    """Load a WAV and return as in-memory bytes for st.audio."""
    data, sr = sf.read(path, dtype="float32")
    buf = io.BytesIO()
    sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def wav_files(directory: str) -> list[Path]:
    return sorted(Path(directory).glob("*.wav"))


def run_agent(agent_fn, *args, progress_placeholder=None, **kwargs):
    """Run an agent function and stream progress to a Streamlit placeholder."""
    log_lines = []

    def _cb(frac, msg):
        log_lines.append(msg)
        if progress_placeholder is not None:
            progress_placeholder.progress(min(frac, 1.0), text=msg)

    result = agent_fn(*args, progress_cb=_cb, **kwargs)
    return result


@st.cache_resource(show_spinner=False)
def load_model():
    from model.separator import _load_model
    return _load_model()


def separate_audio(audio: np.ndarray, sr: int):
    from model.separator import separate
    return separate(audio, sr)


def compute_sdr_for_dir(audio_dir: str, n: int = 50, seed: int = None):
    """Return mean SDR for synthetic pairs built from files in *audio_dir*."""
    from utils.audio import load_audio, make_mixture
    from utils.metrics import mean_sdr

    wavs = wav_files(audio_dir)
    if len(wavs) < 2:
        return None, []
    rng = random.Random(seed) if seed is not None else random.Random()
    shuffled = wavs[:]
    rng.shuffle(shuffled)
    pairs = list(zip(shuffled[::2], shuffled[1::2]))[:n]

    sdrs = []
    for p1, p2 in pairs:
        try:
            s1_ref, sr = load_audio(str(p1))
            s2_ref, _  = load_audio(str(p2))
            mix, s1r, s2r = make_mixture(s1_ref, s2_ref)
            s1e, s2e = separate_audio(mix, sr)
            t = min(len(s1r), len(s1e))
            sdrs.append(mean_sdr(
                np.stack([s1r[:t], s2r[:t]]),
                np.stack([s1e[:t], s2e[:t]])
            ))
        except Exception:
            pass
    return float(np.mean(sdrs)) if sdrs else None, sdrs


def compute_sdr_d3(n: int = 50, seed: int = None):
    """Return mean SDR and per-file SDRs on structured D3 pairs."""
    from utils.audio import load_audio
    from utils.metrics import mean_sdr

    mixes = wav_files(config.D3_MIX_DIR)
    s1s   = wav_files(config.D3_S1_DIR)
    s2s   = wav_files(config.D3_S2_DIR)
    if seed is not None:
        rng = random.Random(seed)
        idxs = list(range(min(len(mixes), len(s1s), len(s2s))))
        rng.shuffle(idxs)
        idxs = idxs[:n]
    else:
        idxs = range(min(len(mixes), len(s1s), len(s2s), n))

    sdrs = []
    for i in idxs:
        try:
            mix, sr = load_audio(str(mixes[i]))
            s1r, _  = load_audio(str(s1s[i]))
            s2r, _  = load_audio(str(s2s[i]))
            s1e, s2e = separate_audio(mix, sr)
            t = min(len(s1r), len(s1e))
            sdrs.append(mean_sdr(
                np.stack([s1r[:t], s2r[:t]]),
                np.stack([s1e[:t], s2e[:t]])
            ))
        except Exception:
            pass
    return float(np.mean(sdrs)) if sdrs else None, sdrs


def sdr_color(v):
    if v is None: return "#8b949e"
    if v > 5:  return "#3fb950"
    if v > 0:  return "#e3b341"
    return "#ff7b72"


# ─────────────────────────────────────────────────────────────────────────────
# Sidebar
# ─────────────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🎙️ AudioCo")
    st.markdown("**Data-Centric Speech Separation**")
    st.markdown("---")

    page = st.radio(
        "Navigate",
        ["🏠 Overview", "⚙️ Run Pipeline", "📊 Evaluation", "🎧 Audio Demo"],
        label_visibility="collapsed",
    )

    st.markdown("---")
    st.markdown("#### Pipeline stages")
    stage_dirs = {
        "D0 Raw":       config.D0_DIR,
        "D1 Segments":  config.D1_DIR,
        "D2 Clean":     config.D2_DIR,
        "D3 Structured": config.D3_MIX_DIR,
    }
    for label, d in stage_dirs.items():
        n = len(wav_files(d))
        icon = "✅" if n > 0 else "⏳"
        st.markdown(f"{icon} **{label}** — {n} files")

    st.markdown("---")
    st.caption("Model: Conv-TasNet (mpariente/ConvTasNet_WHAM!_sepclean)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Overview
# ─────────────────────────────────────────────────────────────────────────────
if page == "🏠 Overview":
    st.markdown('<div class="hero-title">AudioCo</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Data-Centric Speech Separation — same model, better data = better results</div>',
                unsafe_allow_html=True)

    # Drive URL context
    if "overview_drive_url" not in st.session_state:
        st.session_state["overview_drive_url"] = ""

    with st.expander("🔗 Your Google Drive Dataset", expanded=True):
        ov_url = st.text_input(
            "Google Drive folder URL (paste here to track your dataset)",
            value=st.session_state.get("overview_drive_url", ""),
            placeholder="https://drive.google.com/drive/folders/...",
            key="ov_drive_url",
        )
        if ov_url:
            st.session_state["overview_drive_url"] = ov_url
            folder_id_match = __import__("re").search(r"/folders/([a-zA-Z0-9_-]+)", ov_url)
            if folder_id_match:
                fid = folder_id_match.group(1)
                st.success(f"✅ Drive folder ID detected: `{fid}`")
                st.caption("Head to ⚙️ Run Pipeline to download and process this folder.")
            else:
                st.warning("Could not parse a folder ID from this URL.")
        else:
            st.caption("Paste your Drive link above — the pipeline will download from it.")

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # Pipeline diagram — counts from actual disk
    cols = st.columns(4)
    stages = [
        ("D0", "Raw Input",   "mp4/audio files\nfrom Google Drive", "#ff7b72",  config.D0_DIR),
        ("D1", "Segments",    "VAD speech chunks\n16 kHz mono WAV",  "#e3b341",  config.D1_DIR),
        ("D2", "Clean",       "Filtered, normalised\nclips",          "#58a6ff",  config.D2_DIR),
        ("D3", "Structured",  "(mix, s1, s2)\npairs",                 "#3fb950",  config.D3_MIX_DIR),
    ]
    for col, (tag, title, desc, color, d) in zip(cols, stages):
        with col:
            n = len(wav_files(d))
            status = "✅" if n > 0 else "⏳"
            st.markdown(f"""
            <div class="card" style="border-left:4px solid {color}; text-align:center">
                <div style="font-size:1.6rem; font-weight:800; color:{color}">{tag}</div>
                <div style="font-weight:600; margin:4px 0">{title}</div>
                <div style="font-size:0.8rem; color:#8b949e; white-space:pre-line">{desc}</div>
                <div style="margin-top:10px; font-size:1.8rem; font-weight:800; color:{color}">{n}</div>
                <div style="font-size:0.7rem; color:#8b949e">WAV files {status}</div>
            </div>
            """, unsafe_allow_html=True)

    col_refresh, _ = st.columns([1, 3])
    with col_refresh:
        if st.button("🔄 Refresh counts", help="Re-scan all data directories"):
            st.rerun()

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    st.markdown("### 🔬 How it works")
    st.markdown("""
    | Agent | Task | Input → Output |
    |-------|------|---------------|
    | **Agent 1** | Ingest & Segment | Download Drive → convert → VAD split |
    | **Agent 2** | Clean | Remove noise/silence → normalise |
    | **Agent 3** | Structure | Build (mix, s1, s2) training pairs |
    | **Model** | Inference | Conv-TasNet separates the mix |

    **Key insight:** The *same* pretrained model produces higher SDR when given
    cleaner, better-structured input data.  
    Each stage improves the data quality → measurable improvement in SDR.
    """)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Run Pipeline
# ─────────────────────────────────────────────────────────────────────────────
elif page == "⚙️ Run Pipeline":
    st.markdown('<div class="hero-title">Run Pipeline</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Paste your Google Drive folder link and run the pipeline end-to-end</div>',
                unsafe_allow_html=True)

    drive_url = st.text_input(
        "Google Drive folder URL",
        placeholder="https://drive.google.com/drive/folders/...",
        help="Must be a publicly shared Drive folder containing .mp4 audio files",
    )

    col_a, col_b = st.columns(2)
    with col_a:
        skip_download = st.checkbox(
            "Skip download (files already in data/D0_raw/)",
            value=(len(wav_files(config.D0_DIR)) > 0),
        )
    with col_b:
        run_eval = st.checkbox("Run SDR evaluation after pipeline", value=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # Per-agent controls
    st.markdown("#### Individual agents")

    import pandas as pd

    def _file_detail_table(directory: str, label: str):
        """Show a table of WAV files with duration and RMS stats."""
        files = wav_files(directory)
        if not files:
            st.info(f"No files in {label} yet.")
            return
        rows = []
        for fp in files[:50]:  # cap at 50 rows
            try:
                import soundfile as _sf
                info = _sf.info(str(fp))
                dur  = round(info.duration, 2)
                audio_arr, _ = sf.read(str(fp), dtype="float32")
                rms_val = round(float(np.sqrt(np.mean(audio_arr**2))), 4)
            except Exception:
                dur, rms_val = "?", "?"
            rows.append({"File": fp.name, "Duration (s)": dur, "RMS": rms_val})
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)
        st.caption(f"Showing {len(rows)} of {len(files)} files in `{directory}`")

    for idx, (label, desc, mod_name, fn_name, out_dir) in enumerate([
        ("Agent 1 – Ingest & Segment",
         "Download → convert to WAV → VAD split into speech chunks",
         "agents.agent1_ingest", "run", config.D1_DIR),
        ("Agent 2 – Data Cleaning",
         "Filter short/noisy clips → normalise amplitude → save to D2_clean",
         "agents.agent2_clean", "run", config.D2_DIR),
        ("Agent 3 – Dataset Structuring",
         "Build (mix, s1, s2) pairs from clean clips → save to D3/",
         "agents.agent3_structure", "run", config.D3_MIX_DIR),
    ]):
        with st.expander(f"{'①②③'[idx]}  {label}", expanded=False):
            st.caption(desc)

            # Show current file count
            cur_files = wav_files(out_dir)
            st.markdown(
                f'<div class="card" style="padding:0.6rem 1rem; margin-bottom:0.6rem;">'
                f'<b>Current output:</b> {len(cur_files)} WAV files in '
                f'<code>{out_dir.split("/data/")[1] if "/data/" in out_dir else out_dir}</code>'
                f'</div>',
                unsafe_allow_html=True,
            )

            prog_ph = st.empty()
            col_run, col_detail = st.columns([1, 1])
            with col_run:
                run_clicked = st.button(
                    f"▶ Run {label.split('–')[0].strip()}",
                    key=f"btn_agent{idx+1}",
                    type="primary",
                )
            with col_detail:
                show_files = st.checkbox("Show output files", key=f"show_{idx}")

            if run_clicked:
                with st.spinner(f"Running {label}…"):
                    try:
                        import importlib
                        mod = importlib.import_module(mod_name)
                        fn  = getattr(mod, fn_name)
                        if idx == 0:
                            result = run_agent(fn,
                                               drive_url=(None if skip_download else drive_url),
                                               progress_placeholder=prog_ph)
                        else:
                            result = run_agent(fn, progress_placeholder=prog_ph)

                        # Show result summary
                        st.success(f"✅ Done!")
                        r_cols = st.columns(len(result))
                        for rc, (k, v) in zip(r_cols, result.items()):
                            with rc:
                                st.metric(k.replace("_", " ").title(), v)
                    except Exception as e:
                        st.error(f"❌ {e}")

            if show_files:
                st.markdown("**Output file details:**")
                _file_detail_table(out_dir, label)

            # D2 specific: offer to upload clean data to a new Drive folder
            if idx == 1:
                st.markdown("---")
                st.markdown("**📤 Upload D2 clean data to a new Google Drive folder**")
                new_folder_name = st.text_input(
                    "New Drive folder name",
                    value="AudioCo_D2_Clean",
                    key="drive_upload_folder",
                )
                if st.button("☁️ Upload Clean Files to Drive", key="upload_d2"):
                    d2_files = wav_files(config.D2_DIR)
                    if not d2_files:
                        st.warning("No files in D2_clean/ to upload.")
                    else:
                        try:
                            import gdown, subprocess, shutil, tempfile
                            # gdown doesn't support upload; use gdrive CLI if available or show manual instructions
                            st.info(
                                f"**{len(d2_files)} files** are ready in `data/D2_clean/`. "
                                "To upload to Google Drive, either:\n\n"
                                "1. **Drag & drop** the `data/D2_clean/` folder into Google Drive in your browser, or\n"
                                "2. Use [rclone](https://rclone.org/) with `rclone copy data/D2_clean/ gdrive:AudioCo_D2_Clean`\n\n"
                                "Automatic upload via the API requires OAuth credentials — see README."
                            )
                        except Exception as e:
                            st.error(str(e))

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # Full pipeline button
    if st.button("🚀 Run Full Pipeline", type="primary", use_container_width=True):
        from agents.agent1_ingest  import run as a1
        from agents.agent2_clean   import run as a2
        from agents.agent3_structure import run as a3

        prog = st.progress(0.0, "Starting…")

        def _wrap_cb(offset, scale):
            def _cb(frac, msg):
                prog.progress(min(offset + frac * scale, 1.0), text=msg)
            return _cb

        with st.status("Pipeline running…", expanded=True) as status:
            try:
                st.write("**Agent 1** – Ingest & Segment")
                r1 = a1(drive_url=(None if skip_download else drive_url),
                        progress_cb=_wrap_cb(0.0, 0.33))
                st.write(f"  → {r1['n_chunks']} chunks")

                st.write("**Agent 2** – Data Cleaning")
                r2 = a2(progress_cb=_wrap_cb(0.33, 0.33))
                st.write(f"  → {r2['n_kept']} clean clips")

                st.write("**Agent 3** – Dataset Structuring")
                r3 = a3(progress_cb=_wrap_cb(0.66, 0.34))
                st.write(f"  → {r3['n_pairs']} pairs")

                status.update(label="✅ Pipeline complete!", state="complete")
                prog.progress(1.0, "Done!")
            except Exception as e:
                status.update(label="❌ Pipeline failed", state="error")
                st.error(str(e))

        if run_eval:
            st.markdown("### 📊 SDR Evaluation")
            with st.spinner("Running model & computing SDR…"):
                load_model()
                sdr_vals = {
                    "D0 Raw":       compute_sdr_for_dir(config.D0_DIR),
                    "D1 Segments":  compute_sdr_for_dir(config.D1_DIR),
                    "D2 Clean":     compute_sdr_for_dir(config.D2_DIR),
                    "D3 Structured": compute_sdr_d3(),
                }
            st.session_state["sdr_vals"] = sdr_vals
            st.success("Evaluation done! Head to the 📊 Evaluation page.")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Evaluation
# ─────────────────────────────────────────────────────────────────────────────
elif page == "📊 Evaluation":
    st.markdown('<div class="hero-title">Evaluation</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Live SDR scores computed from your actual dataset — same model, better data → better results</div>',
                unsafe_allow_html=True)

    import pandas as pd

    # Check if any data exists at all
    total_d0 = len(wav_files(config.D0_DIR))
    total_d1 = len(wav_files(config.D1_DIR))
    total_d2 = len(wav_files(config.D2_DIR))
    total_d3 = min(len(wav_files(config.D3_MIX_DIR)), len(wav_files(config.D3_S1_DIR)), len(wav_files(config.D3_S2_DIR)))

    dataset_col1, dataset_col2, dataset_col3, dataset_col4 = st.columns(4)
    for col, lbl, cnt, color in [
        (dataset_col1, "D0 Raw",        total_d0, "#ff7b72"),
        (dataset_col2, "D1 Segments",   total_d1, "#e3b341"),
        (dataset_col3, "D2 Clean",      total_d2, "#58a6ff"),
        (dataset_col4, "D3 Pairs",      total_d3, "#3fb950"),
    ]:
        with col:
            st.markdown(
                f'<div class="metric-tile">'
                f'<div class="metric-label">{lbl}</div>'
                f'<div class="metric-value" style="color:{color}">{cnt}</div>'
                f'<div class="metric-unit">files</div>'
                f'</div>',
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    if total_d3 == 0 and total_d2 == 0:
        st.warning("⚠️ No processed data found. Please run the pipeline first (⚙️ Run Pipeline page).")
        st.stop()

    dynamic_mode = st.toggle("🔄 Dynamic Mode", value=False, help="Compute SDR on all available data (no caching)")

    def _get_seed():
        return int(time.time() * 1000) % 100000 if dynamic_mode else 42

    if dynamic_mode:
        st.caption("Dynamic mode: using all files, results change every run.")
        st.session_state.pop("sdr_vals", None)
        st.session_state.pop("sdr_per_file", None)

    if "sdr_vals" not in st.session_state or dynamic_mode:
        data_ready = total_d2 >= 2 or total_d3 >= 1
        seed = _get_seed()

        if not data_ready:
            if st.button("▶ Compute SDR now", type="primary"):
                with st.spinner("Loading model & computing SDR…"):
                    load_model()
                    d0_mean, _ = compute_sdr_for_dir(config.D0_DIR, seed=seed)
                    d1_mean, _ = compute_sdr_for_dir(config.D1_DIR, seed=seed)
                    d2_mean, d2_list = compute_sdr_for_dir(config.D2_DIR, seed=seed)
                    d3_mean, d3_list = compute_sdr_d3(seed=seed)
                    st.session_state["sdr_vals"] = {"D0 Raw": d0_mean, "D1 Segments": d1_mean, "D2 Clean": d2_mean, "D3 Structured": d3_mean}
                    st.session_state["sdr_per_file"] = {"D0 Raw": [], "D1 Segments": [], "D2 Clean": d2_list, "D3 Structured": d3_list}
                    st.rerun()
            else:
                st.stop()
        else:
            with st.spinner("Loading model & computing SDR on your actual files…"):
                load_model()
                d0_mean, d0_list = compute_sdr_for_dir(config.D0_DIR, seed=seed)
                d1_mean, d1_list = compute_sdr_for_dir(config.D1_DIR, seed=seed)
                d2_mean, d2_list = compute_sdr_for_dir(config.D2_DIR, seed=seed)
                d3_mean, d3_list = compute_sdr_d3(seed=seed)
                st.session_state["sdr_vals"] = {"D0 Raw": d0_mean, "D1 Segments": d1_mean, "D2 Clean": d2_mean, "D3 Structured": d3_mean}
                st.session_state["sdr_per_file"] = {"D0 Raw": d0_list, "D1 Segments": d1_list, "D2 Clean": d2_list, "D3 Structured": d3_list}
            st.success("✅ SDR computed from your actual dataset!")

    sdr_vals = st.session_state["sdr_vals"]
    sdr_per_file = st.session_state.get("sdr_per_file", {})

    st.markdown("### 📈 SDR Results (from your data)")

    tile_cols = st.columns(4)
    for col, (label, val) in zip(tile_cols, sdr_vals.items()):
        color = sdr_color(val)
        disp  = f"{val:+.1f}" if val is not None else "N/A"
        count = len(sdr_per_file.get(label, []))
        with col:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{disp}</div>
                <div class="metric-unit">dB SDR <small>({count} pairs)</small></div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    labels = list(sdr_vals.keys())
    values = [v if v is not None else 0 for v in sdr_vals.values()]
    colors = [sdr_color(v) for v in sdr_vals.values()]

    fig = go.Figure(go.Bar(
        x=labels, y=values,
        marker_color=colors,
        text=[f"{v:+.2f} dB" if v is not None else "N/A" for v in sdr_vals.values()],
        textposition="outside",
        textfont=dict(color="white", size=13),
    ))
    fig.update_layout(
        plot_bgcolor="#161b22",
        paper_bgcolor="#0d1117",
        font=dict(color="#e6edf3", family="Inter"),
        xaxis=dict(gridcolor="#21262d"),
        yaxis=dict(gridcolor="#21262d", title="Mean SDR (dB)"),
        title=dict(text="SDR by Pipeline Stage — Your Dataset", font=dict(size=16)),
        margin=dict(t=50, b=30),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Summary table
    st.markdown("#### Aggregate Results")
    rows = []
    for label, val in sdr_vals.items():
        file_cnt = {"D0 Raw": total_d0, "D1 Segments": total_d1,
                    "D2 Clean": total_d2, "D3 Structured": total_d3}.get(label, 0)
        per_list = sdr_per_file.get(label, [])
        std_disp = f"± {np.std(per_list):.2f}" if len(per_list) > 1 else "N/A"
        disp = f"{val:+.2f} dB" if val is not None else "— (no data)"
        note = (
            "Noisy raw input"         if "D0" in label else
            "VAD chunked"             if "D1" in label else
            "Cleaned & normalised"    if "D2" in label else
            "Structured pairs (best)"
        )
        rows.append({"Stage": label, "Files": file_cnt, "Pairs": len(per_list), "Mean SDR": disp, "Std Dev": std_disp, "Description": note})
    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)

    # Per-file SDR breakdown — shown for D2 and D3 stages
    for stage_key, stage_dir, stage_label in [
        ("D2 Clean", config.D2_DIR, "D2 Clean"),
        ("D3 Structured", config.D3_MIX_DIR, "D3 Structured Pairs"),
    ]:
        per_list = sdr_per_file.get(stage_key, [])
        if not per_list:
            continue
        st.markdown(f"#### 📊 Per-File SDR Breakdown — {stage_label}")
        max_files = min(len(per_list), 100)
        per_rows = [{"#": i+1, "SDR (dB)": round(v, 3),
                     "Quality": "✅ Good" if v > 5 else ("⚠️ OK" if v > 0 else "❌ Poor")}
                    for i, v in enumerate(per_list[:max_files])]
        df3 = pd.DataFrame(per_rows)
        st.dataframe(df3, use_container_width=True, hide_index=True)
        st.caption(f"{len(per_list)} pairs evaluated — showing {max_files}")

        # Live histogram
        hist_fig = go.Figure(go.Histogram(
            x=per_list, nbins=20,
            marker_color="#58a6ff", opacity=0.8,
            hovertemplate="SDR: %{x:.2f} dB<br>Count: %{y}<extra></extra>"
        ))
        hist_fig.update_layout(
            plot_bgcolor="#161b22",
            paper_bgcolor="#0d1117",
            font=dict(color="#e6edf3"),
            title=dict(text=f"SDR Distribution — {stage_label}", font=dict(size=14)),
            xaxis=dict(title="SDR (dB)", gridcolor="#21262d"),
            yaxis=dict(title="Count", gridcolor="#21262d"),
            height=260,
            margin=dict(t=40, b=30),
        )
        st.plotly_chart(hist_fig, use_container_width=True)

        good = sum(1 for v in per_list if v > 0)
        st.metric("Files with positive SDR", f"{good}/{len(per_list)} ({100*good/len(per_list):.1f}%)")


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Audio Demo
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎧 Audio Demo":
    st.markdown('<div class="hero-title">Audio Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Upload your own audio or listen to pipeline stage comparisons</div>',
                unsafe_allow_html=True)

    import io as _io
    from utils.audio import load_audio, make_mixture
    from utils.metrics import mean_sdr as _msdr

    def _arr_to_bytes(arr, sample_rate):
        buf = _io.BytesIO()
        sf.write(buf, arr.astype("float32"), sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    # ── SECTION 1: Upload your own audio ────────────────────────────────────
    st.markdown("### 📤 Upload Your Own Audio")
    st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
    uploaded = st.file_uploader(
        "Upload a mixed audio file (WAV, MP3, FLAC, OGG, MP4, M4A)",
        type=["wav", "mp3", "flac", "ogg", "mp4", "m4a"],
        key="demo_upload",
    )

    if uploaded is not None:
        # Save to temp path
        import tempfile, pathlib
        suffix = pathlib.Path(uploaded.name).suffix or ".wav"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded.read())
            tmp_path = tmp.name

        st.markdown(f"**Uploaded:** `{uploaded.name}` ({uploaded.size/1024:.1f} KB)")
        st.audio(tmp_path, format="audio/wav")

        if st.button("🤖 Separate Speakers", key="sep_upload", type="primary"):
            with st.spinner("Loading model & running separation…"):
                try:
                    load_model()
                    mix_audio, sr_up = load_audio(tmp_path)
                    s1_est, s2_est = separate_audio(mix_audio, sr_up)
                    st.success("✅ Separation complete!")
                    col1, col2 = st.columns(2)
                    with col1:
                        st.markdown("**🔊 Separated — Speaker 1**")
                        st.audio(_arr_to_bytes(s1_est, sr_up), format="audio/wav")
                    with col2:
                        st.markdown("**🔊 Separated — Speaker 2**")
                        st.audio(_arr_to_bytes(s2_est, sr_up), format="audio/wav")

                    # Quick SDR info
                    rms1 = float(np.sqrt(np.mean(s1_est**2)))
                    rms2 = float(np.sqrt(np.mean(s2_est**2)))
                    st.info(f"Speaker 1 RMS: {rms1:.4f} | Speaker 2 RMS: {rms2:.4f} (no ground truth for uploaded files)")
                except Exception as e:
                    st.error(f"Separation failed: {e}")
    else:
        st.caption("No file uploaded yet — drag and drop or click Browse.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── SECTION 2: Pipeline stage comparison ────────────────────────────────
    st.markdown("### 📊 Pipeline Stage Comparison (D0 → D3)")

    d3_mixes = wav_files(config.D3_MIX_DIR)
    d3_s1s   = wav_files(config.D3_S1_DIR)
    d3_s2s   = wav_files(config.D3_S2_DIR)
    n_avail  = min(len(d3_mixes), len(d3_s1s), len(d3_s2s))

    if n_avail == 0:
        st.info("No D3 pairs found. Run the pipeline first to enable stage comparison.")
    else:
        sample_idx = st.slider("Sample index", 0, n_avail - 1, 0)
        mix_path   = d3_mixes[sample_idx]
        s1_path    = d3_s1s[sample_idx]
        s2_path    = d3_s2s[sample_idx]

        st.markdown(f"**File:** `{mix_path.name}`")
        st.markdown('<hr class="soft">', unsafe_allow_html=True)

        st.markdown("#### 🔀 Input — Mixed Audio")
        st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
        st.audio(audio_bytes(str(mix_path)), format="audio/wav")
        st.markdown("Two speakers mixed together.")
        st.markdown("</div>", unsafe_allow_html=True)

        st.markdown("#### 🎯 Ground-Truth Sources")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown("**Speaker 1 (reference)**")
            st.audio(audio_bytes(str(s1_path)), format="audio/wav")
        with c2:
            st.markdown("**Speaker 2 (reference)**")
            st.audio(audio_bytes(str(s2_path)), format="audio/wav")

        st.markdown("#### 🤖 Model Separation at Each Stage")
        s1_ref, sr = load_audio(str(s1_path))
        s2_ref, _  = load_audio(str(s2_path))

        stage_info = [
            ("D0 — Raw",        config.D0_DIR,  "#ff7b72"),
            ("D1 — Segmented",  config.D1_DIR,  "#e3b341"),
            ("D2 — Cleaned",    config.D2_DIR,  "#58a6ff"),
            ("D3 — Structured", None,            "#3fb950"),
        ]

        for stage_label, stage_dir, color in stage_info:
            with st.expander(f"🔊 {stage_label}", expanded=(stage_label == "D3 — Structured")):
                try:
                    with st.spinner("Running model…"):
                        if stage_dir is None:
                            mix_audio, sr2 = load_audio(str(mix_path))
                        else:
                            stage_wavs = wav_files(stage_dir)
                            if len(stage_wavs) < 2:
                                st.warning(f"Not enough files in {stage_dir}")
                                continue
                            random.seed(sample_idx)
                            random.shuffle(stage_wavs)
                            a1, sr2 = load_audio(str(stage_wavs[0]))
                            a2, _   = load_audio(str(stage_wavs[1]))
                            mix_audio, _, _ = make_mixture(a1, a2)
                        s1_est, s2_est = separate_audio(mix_audio, sr2)

                    t = min(len(s1_ref), len(s1_est))
                    sdr_val = _msdr(
                        np.stack([s1_ref[:t], s2_ref[:t]]),
                        np.stack([s1_est[:t], s2_est[:t]])
                    )
                    st.markdown(
                        f'<span style="color:{color}; font-weight:700; font-size:1.1rem;">'
                        f'SDR: {sdr_val:+.2f} dB</span>',
                        unsafe_allow_html=True,
                    )
                    ca, cb = st.columns(2)
                    with ca:
                        st.markdown("**Separated — Speaker 1**")
                        st.audio(_arr_to_bytes(s1_est, sr2), format="audio/wav")
                    with cb:
                        st.markdown("**Separated — Speaker 2**")
                        st.audio(_arr_to_bytes(s2_est, sr2), format="audio/wav")
                except Exception as e:
                    st.error(f"Separation failed: {e}")
