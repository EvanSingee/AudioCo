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


def compute_sdr_for_dir(audio_dir: str, n: int = 3):
    """Return mean SDR for synthetic pairs built from files in *audio_dir*."""
    from utils.audio import load_audio, make_mixture
    from utils.metrics import mean_sdr

    wavs = wav_files(audio_dir)
    if len(wavs) < 2:
        return None
    random.seed(0)
    shuffled = list(wavs)
    random.shuffle(shuffled)
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
    return float(np.mean(sdrs)) if sdrs else None


def compute_sdr_d3(n: int = 3):
    """Return mean SDR on structured D3 pairs."""
    from utils.audio import load_audio
    from utils.metrics import mean_sdr

    mixes = wav_files(config.D3_MIX_DIR)
    s1s   = wav_files(config.D3_S1_DIR)
    s2s   = wav_files(config.D3_S2_DIR)
    k = min(len(mixes), len(s1s), len(s2s), n)
    if k == 0:
        return None

    sdrs = []
    for mf, s1f, s2f in zip(mixes[:k], s1s[:k], s2s[:k]):
        try:
            mix, sr = load_audio(str(mf))
            s1r, _  = load_audio(str(s1f))
            s2r, _  = load_audio(str(s2f))
            s1e, s2e = separate_audio(mix, sr)
            t = min(len(s1r), len(s1e))
            sdrs.append(mean_sdr(
                np.stack([s1r[:t], s2r[:t]]),
                np.stack([s1e[:t], s2e[:t]])
            ))
        except Exception:
            pass
    return float(np.mean(sdrs)) if sdrs else None


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

    # Pipeline diagram
    cols = st.columns(4)
    stages = [
        ("D0", "Raw Input", "mp4 recordings\nfrom Google Drive", "#ff7b72"),
        ("D1", "Segments",  "VAD speech chunks\n16 kHz mono WAV",  "#e3b341"),
        ("D2", "Clean",     "Filtered, normalised\nclips",          "#58a6ff"),
        ("D3", "Structured","(mix, s1, s2)\npairs",                 "#3fb950"),
    ]
    for col, (tag, title, desc, color) in zip(cols, stages):
        with col:
            n = len(wav_files({
                "D0": config.D0_DIR,
                "D1": config.D1_DIR,
                "D2": config.D2_DIR,
                "D3": config.D3_MIX_DIR,
            }[tag]))
            st.markdown(f"""
            <div class="card" style="border-left:4px solid {color}; text-align:center">
                <div style="font-size:1.6rem; font-weight:800; color:{color}">{tag}</div>
                <div style="font-weight:600; margin:4px 0">{title}</div>
                <div style="font-size:0.8rem; color:#8b949e; white-space:pre-line">{desc}</div>
                <div style="margin-top:10px; font-size:1.2rem; font-weight:700; color:{color}">{n}</div>
                <div style="font-size:0.7rem; color:#8b949e">files</div>
            </div>
            """, unsafe_allow_html=True)

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
    for idx, (label, desc, mod_name, fn_name) in enumerate([
        ("Agent 1 – Ingest & Segment",
         "Download → convert to WAV → VAD split",
         "agents.agent1_ingest", "run"),
        ("Agent 2 – Data Cleaning",
         "Filter short/noisy clips → normalise",
         "agents.agent2_clean", "run"),
        ("Agent 3 – Dataset Structuring",
         "Detect/create (mix, s1, s2) pairs",
         "agents.agent3_structure", "run"),
    ]):
        with st.expander(f"{'①②③'[idx]}  {label}", expanded=False):
            st.caption(desc)
            prog_ph = st.empty()
            if st.button(f"▶ Run {label.split('–')[0].strip()}", key=f"btn_agent{idx+1}"):
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
                        st.success(f"✅ Done: {result}")
                    except Exception as e:
                        st.error(f"❌ {e}")

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
    st.markdown('<div class="hero-sub">SDR improvement across pipeline stages</div>',
                unsafe_allow_html=True)

    # Load cached or compute fresh
    if "sdr_vals" not in st.session_state:
        if st.button("▶ Compute SDR now", type="primary"):
            with st.spinner("Loading model & computing SDR…"):
                load_model()
                st.session_state["sdr_vals"] = {
                    "D0 Raw":       compute_sdr_for_dir(config.D0_DIR),
                    "D1 Segments":  compute_sdr_for_dir(config.D1_DIR),
                    "D2 Clean":     compute_sdr_for_dir(config.D2_DIR),
                    "D3 Structured": compute_sdr_d3(),
                }
        else:
            st.info("Run the pipeline first or click 'Compute SDR now'.")
            st.stop()

    sdr_vals = st.session_state["sdr_vals"]

    # Metric tiles
    tile_cols = st.columns(4)
    for col, (label, val) in zip(tile_cols, sdr_vals.items()):
        color = sdr_color(val)
        disp  = f"{val:+.1f}" if val is not None else "N/A"
        with col:
            st.markdown(f"""
            <div class="metric-tile">
                <div class="metric-label">{label}</div>
                <div class="metric-value" style="color:{color}">{disp}</div>
                <div class="metric-unit">dB SDR</div>
            </div>
            """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Bar chart
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
        title=dict(text="SDR by Pipeline Stage", font=dict(size=16)),
        margin=dict(t=50, b=30),
        height=380,
    )
    st.plotly_chart(fig, use_container_width=True)

    # Table
    st.markdown("#### Detailed Results")
    rows = []
    for label, val in sdr_vals.items():
        disp = f"{val:+.2f} dB" if val is not None else "—"
        note = (
            "Noisy raw input" if "D0" in label else
            "VAD chunked"    if "D1" in label else
            "Cleaned & normalised" if "D2" in label else
            "Structured pairs (best)"
        )
        rows.append({"Stage": label, "Mean SDR": disp, "Description": note})

    import pandas as pd
    df = pd.DataFrame(rows)
    st.dataframe(df, use_container_width=True, hide_index=True)


# ─────────────────────────────────────────────────────────────────────────────
# PAGE: Audio Demo
# ─────────────────────────────────────────────────────────────────────────────
elif page == "🎧 Audio Demo":
    st.markdown('<div class="hero-title">Audio Demo</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Listen to separated speech at each pipeline stage</div>',
                unsafe_allow_html=True)

    from utils.audio import load_audio, make_mixture

    d3_mixes = wav_files(config.D3_MIX_DIR)
    d3_s1s   = wav_files(config.D3_S1_DIR)
    d3_s2s   = wav_files(config.D3_S2_DIR)

    n_avail = min(len(d3_mixes), len(d3_s1s), len(d3_s2s))

    if n_avail == 0:
        st.warning("No D3 pairs found. Run the pipeline first (Run Pipeline page).")
        st.stop()

    sample_idx = st.slider("Sample index", 0, n_avail - 1, 0)
    mix_path = d3_mixes[sample_idx]
    s1_path  = d3_s1s[sample_idx]
    s2_path  = d3_s2s[sample_idx]

    st.markdown(f"**File:** `{mix_path.name}`")
    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── Mixed input ──────────────────────────────────────────────────────────
    st.markdown("### 🔀 Input — Mixed Audio")
    st.markdown('<div class="card card-accent">', unsafe_allow_html=True)
    st.audio(audio_bytes(str(mix_path)), format="audio/wav")
    st.markdown("Two speakers mixed together — hard to understand either one.")
    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── Reference sources ────────────────────────────────────────────────────
    st.markdown("### 🎯 Ground-Truth Sources")
    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**Speaker 1 (reference)**")
        st.audio(audio_bytes(str(s1_path)), format="audio/wav")
    with c2:
        st.markdown("**Speaker 2 (reference)**")
        st.audio(audio_bytes(str(s2_path)), format="audio/wav")

    st.markdown('<hr class="soft">', unsafe_allow_html=True)

    # ── Model separation per stage ───────────────────────────────────────────
    st.markdown("### 🤖 Model Separation at Each Stage")

    stage_info = [
        ("D0 — Raw", config.D0_DIR, "#ff7b72"),
        ("D1 — Segmented", config.D1_DIR, "#e3b341"),
        ("D2 — Cleaned", config.D2_DIR, "#58a6ff"),
        ("D3 — Structured", None, "#3fb950"),  # None = use d3 mix directly
    ]

    s1_ref, sr = load_audio(str(s1_path))
    s2_ref, _  = load_audio(str(s2_path))

    from utils.metrics import mean_sdr as compute_mean_sdr
    import io as _io

    def _arr_to_bytes(arr, sample_rate):
        buf = _io.BytesIO()
        sf.write(buf, arr.astype("float32"), sample_rate, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    for stage_label, stage_dir, color in stage_info:
        with st.expander(f"🔊 {stage_label}", expanded=(stage_label == "D3 — Structured")):
            try:
                with st.spinner("Running model…"):
                    if stage_dir is None:
                        # D3: use the actual structured mix
                        mix_audio, sr2 = load_audio(str(mix_path))
                    else:
                        # Build synthetic mix from stage files
                        stage_wavs = wav_files(stage_dir)
                        if len(stage_wavs) < 2:
                            st.warning(f"Not enough files in {stage_dir}")
                            continue
                        random.seed(sample_idx)
                        random.shuffle(stage_wavs)
                        a1, sr2 = load_audio(str(stage_wavs[0]))
                        a2, _   = load_audio(str(stage_wavs[1]))
                        mix_audio, s1_syn, s2_syn = make_mixture(a1, a2)

                    s1_est, s2_est = separate_audio(mix_audio, sr2)

                # SDR
                t = min(len(s1_ref), len(s1_est))
                from utils.metrics import mean_sdr as _msdr
                import numpy as np
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
