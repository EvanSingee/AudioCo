# AudioCo — Data-Centric Speech Separation via AI Agents

> **Research Project** | 4th Sem PS | Data-Centric AI

---

## 🧠 Core Idea

Most speech separation models assume clean, well-labeled data.  
**AudioCo proves that improving data quality—not the model—is the key bottleneck.**

A 4-stage AI agent pipeline converts messy, real-world phone recordings into a structured, supervised dataset.  
The same fixed Conv-TasNet model is evaluated at each stage to quantify the data quality gain.

---

## 📁 Project Structure

```
AudioCo/
├── config.py                  # Central configuration
├── utils.py                   # Shared audio utilities
├── run_pipeline.py            # Orchestrator (run all 4 agents)
├── evaluate.py                # Conv-TasNet evaluation (SDR/SI-SDR/STOI)
├── requirements.txt
│
├── agents/
│   ├── agent1_vad.py          # VAD Segmentation        D0 → D1
│   ├── agent2_cluster.py      # Speaker Clustering      D1 → D2
│   ├── agent3_quality.py      # Data Quality Filter     D2 (in-place)
│   └── agent4_builder.py      # Dataset Builder         D2 → D3
│
├── demo/
│   └── app.py                 # Streamlit Demo UI
│
├── scripts/
│   ├── prepare_raw.py         # Ingest & convert raw recordings → D0
│   └── ablation.py            # Ablation study runner
│
└── data/
    ├── D0_raw/                # Raw recordings (wav)
    ├── D1_segmented/          # Speech-only chunks
    ├── D2_clustered/          # Per-speaker directories
    ├── D3_structured/         # train/val/test mix·s1·s2 pairs
    ├── clean_reference/       # Optional clean single-speaker reference
    └── logs/                  # Evaluation & timing outputs
```

---

## ⚡ Quick Start

### 1. Install dependencies

```bash
# Create virtual environment (recommended)
python -m venv .venv && source .venv/bin/activate

# Install (requires ffmpeg on system)
pip install -r requirements.txt
```

> macOS: `brew install ffmpeg`

---

### 2. Add your data

```bash
# Copy or convert raw recordings into D0_raw
python scripts/prepare_raw.py --src /path/to/your/recordings

# Or manually place .wav/.mp4/.m4a/.mpeg files in:
# data/D0_raw/
```

---

### 3. Run the full pipeline

```bash
python run_pipeline.py
```

Or run individual agents:

```bash
python run_pipeline.py --only 1   # VAD only
python run_pipeline.py --from 2   # resume from Agent 2
```

---

### 4. Evaluate

```bash
python evaluate.py --dataset D3          # evaluate structured dataset
python evaluate.py --all                 # compare D0 vs D3
python scripts/ablation.py              # full ablation study
```

---

### 5. Launch the demo

```bash
streamlit run demo/app.py
```

Open http://localhost:8501

---

## 🤖 Agents

| Agent | Goal | Input | Output | Tools |
|-------|------|-------|--------|-------|
| **1 – VAD** | Remove silence, split recordings | D0 raw | D1 chunks | webrtcvad / librosa |
| **2 – Cluster** | Group chunks by speaker | D1 chunks | D2 per-speaker dirs | SpeechBrain ECAPA-TDNN, sklearn |
| **3 – Quality** | Remove noisy / overlapping chunks | D2 | D2 filtered | SNR, RMS, spectral flatness |
| **4 – Builder** | Create (mix, s1, s2) pairs | D2 clean | D3 train/val/test | utils.mix_signals |

---

## 📊 Dataset Versions

| Version | Description |
|---------|-------------|
| **D0** | Raw recordings (converted to 16kHz WAV) |
| **D1** | VAD-segmented speech chunks |
| **D2** | Speaker-clustered chunks (pseudo labels) |
| **D3** | Supervised (mix, s1, s2) training pairs |

---

## 🧪 Experiment Design

```
D0  ──────────────────► Conv-TasNet ──► SDR(D0)  (poor)
D1  ──────────────────► Conv-TasNet ──► SDR(D1)  (better)
D2  ──────────────────► Conv-TasNet ──► SDR(D2)  (good)
D3  ──────────────────► Conv-TasNet ──► SDR(D3)  (best)
```

**Same model, different data quality → measurable SDR improvement.**

---

## 🎛️ Configuration

All key parameters live in `config.py`:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `VAD_AGGRESSIVENESS` | 3 | webrtcvad sensitivity (0–3) |
| `CLUSTER_METHOD` | `agglomerative` | `kmeans` / `agglomerative` / `spectral` |
| `N_CLUSTERS` | `None` | `None` = auto-detect |
| `SNR_THRESHOLD_DB` | 5.0 | Minimum SNR to keep a chunk |
| `MIX_SNR_RANGE` | `(-5, 5)` | dB range for synthetic mixtures |
| `PAIRS_PER_SPEAKER` | 20 | Mix pairs per speaker combination |
| `CONV_TASNET_MODEL` | `mpariente/ConvTasNet_WHAM!_sepclean` | Pretrained HuggingFace model |

---

## 📈 Metrics

| Metric | Description |
|--------|-------------|
| **SDR** | Signal-to-Distortion Ratio (mir_eval) |
| **SI-SDR** | Scale-Invariant SDR – permutation-invariant |
| **STOI** | Short-Time Objective Intelligibility (0–1) |

---

## 🛠️ Tech Stack

- **Python 3.10+**
- **PyTorch + torchaudio** — model inference
- **Asteroid** — pretrained Conv-TasNet
- **SpeechBrain** — ECAPA-TDNN speaker embeddings
- **librosa + soundfile** — audio I/O
- **webrtcvad** — Voice Activity Detection
- **scikit-learn** — clustering
- **pydub + ffmpeg** — format conversion
- **Streamlit + Plotly** — demo UI
- **mir_eval + pystoi** — evaluation metrics
