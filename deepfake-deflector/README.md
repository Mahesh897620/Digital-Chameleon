# Deepfake Deflector

### *Your identity is not a dataset.*

> **Deepfake Deflector** is a real-time adversarial defense system that makes your face and voice **untouchable** by deepfake AI. It injects imperceptible perturbations into your webcam feed and audio stream — disrupting face-swap networks, voice cloners, and biometric harvesters before they can harvest a single frame.

Built by **Digital Chameleon** for the **AMD Slingshot Hackathon**.

---

## The Problem

Deepfake models (face-swap GANs, voice cloners, lip-sync networks) train and operate on **clean, unperturbed input**. A single Zoom call gives an attacker hundreds of high-quality face frames and minutes of voice data — enough to generate a convincing deepfake in hours.

## Our Solution

Deepfake Deflector sits between your camera/microphone and the outside world. It applies **adversarial noise** tuned to exploit the weaknesses of generative models:

| Layer | Technique | Effect on Attacker |
|-------|-----------|-------------------|
| **Face Shield** | Gaussian perturbation with temporal shift, applied to the face ROI detected by MediaPipe Face Mesh | Disrupts face encoders — swap networks produce artifacts and fail to converge |
| **Full-Frame Shield** | Half-strength noise injected across the entire frame as a secondary defense | Poisons background features used by reconstruction networks |
| **Voice Shimmer** | Micro pitch-shift + -60 dB white noise + IIR formant jitter applied to the live audio stream | Degrades voice embeddings — cloners produce garbled, unusable output |
| **Virtual Camera** | Protected video piped through pyvirtualcam to any video-call app (Zoom, Meet, Teams) | Transparent defense — no app changes needed |
| **Threat Dashboard** | Live Streamlit dashboard showing frames protected, faces detected, shield status, and threat events | Real-time visibility into the defense posture |

---

## Demo Mode

Press **`D`** to toggle split-screen demo mode — a side-by-side comparison designed for live presentations:

```
┌─────────────────────┬─────────────────────┐
│   ⚠️ UNPROTECTED     │   🛡️ PROTECTED       │
│                     │                     │
│  Red scanning line  │  Green border       │
│  "AI Harvester      │  "HARVEST BLOCKED"  │
│   Active"           │   (flashing)        │
│                     │                     │
│  ← What attackers   │  ← What attackers   │
│     see normally    │     get from you →  │
└─────────────────────┴─────────────────────┘
```

---

## Quick Start

```bash
# 1. Clone
git clone https://github.com/Digital-Chameleon/deepfake-deflector.git
cd deepfake-deflector

# 2. Install dependencies
pip3 install -r requirements.txt

# 3. Download the MediaPipe face model (one-time)
#    (Already included in the repo as face_landmarker.task)

# 4. Launch everything (webcam + dashboard)
./run.sh            # macOS/Linux
# or
run.bat             # Windows
```

### Run components individually

```bash
# Webcam feed only
python3 main.py

# Webcam feed + demo mode, no virtual camera
python3 main.py --demo --no-vcam

# Dashboard only
streamlit run modules/dashboard.py
```

---

## Keyboard Controls

| Key | Action |
|-----|--------|
| **Q** | Quit |
| **D** | Toggle demo split-screen mode |
| **V** | Toggle voice shield (VoiceShimmer) |
| **H** | Toggle HUD overlay |
| **S** | Save screenshot to `screenshots/` |
| **+** / **=** | Increase perturbation strength (1–15) |
| **-** | Decrease perturbation strength (1–15) |

---

## Project Structure

```
deepfake-deflector/
├── main.py                     # Entry point — webcam loop, demo mode, HUD, key controls
├── face_landmarker.task        # MediaPipe Face Mesh model (3.6 MB)
├── requirements.txt            # Python dependencies
├── run.sh / run.bat            # One-click launchers (webcam + dashboard)
├── modules/
│   ├── __init__.py
│   ├── video_processor.py      # FaceProcessor, AdversarialShield
│   ├── audio_processor.py      # VoiceShimmer (real-time audio perturbation)
│   └── dashboard.py            # Streamlit threat monitoring dashboard
├── data/
│   └── threat_log.json         # Shared state between main.py and dashboard
├── screenshots/                # Auto-created when you press S
└── tests/
    ├── test_runner.py
    ├── test_shield.py
    ├── test_voice.py
    ├── test_vcam.py
    └── test_dashboard.py
```

---

## How It Works — Technical Deep Dive

### 1. Face Detection (MediaPipe Tasks API)

We use MediaPipe's `FaceLandmarker` (Tasks API, not the deprecated Solutions API) in **VIDEO** running mode. Each frame receives a monotonically increasing timestamp. The model returns 478 face landmarks plus a bounding box.

### 2. Adversarial Perturbation

The `AdversarialShield` generates Gaussian noise matched to the face ROI dimensions. Every 3 frames, the noise cache is regenerated with a temporal shift — this prevents deepfake models from learning to filter out a static pattern. The noise is clipped and blended into the frame using NumPy vectorized operations for real-time performance.

Perturbation strength is adjustable from **1** (barely visible, effective against simple models) to **15** (visible grain, effective against state-of-the-art face-swap networks).

### 3. Voice Shimmer

Real-time audio processing via PyAudio:
- **Pitch shift**: Resampling-based micro pitch shift (±0.3%)
- **Noise floor**: -60 dB white noise added to the audio stream
- **Formant jitter**: Random IIR filter (scipy `lfilter`) applied per chunk to randomize formant structure

The combination degrades voice embeddings used by cloning systems like SV2TTS and VALL-E without making speech unintelligible to humans.

### 4. Virtual Camera

Protected frames are piped through `pyvirtualcam` to a virtual camera device. Select "OBS Virtual Camera" (or your v4l2loopback device on Linux) in Zoom/Meet/Teams to broadcast the defended feed.

---

## Requirements

- **Python 3.9+**
- **Webcam** (built-in or USB)
- **macOS** (tested on Apple Silicon), Linux, or Windows

### Virtual Camera Setup

| OS | Steps |
|----|-------|
| macOS | Install [OBS Studio](https://obsproject.com), start OBS, enable **Start Virtual Camera** |
| Windows | Install [OBS Studio](https://obsproject.com) — virtual camera driver is bundled |
| Linux | `sudo apt install v4l2loopback-dkms && sudo modprobe v4l2loopback devices=1` |

---

## Dependencies

| Package | Version | Purpose |
|---------|---------|---------|
| opencv-python | 4.x | Video capture, display, drawing |
| mediapipe | 0.10.x | Face detection (Tasks API) |
| numpy | 2.x | Array operations, noise generation |
| scipy | 1.x | IIR filter for formant jitter |
| pyvirtualcam | 0.4.x | Virtual camera output |
| pyaudio | 0.2.x | Real-time audio I/O |
| streamlit | 1.x | Threat monitoring dashboard |

---

## Running Tests

```bash
python3 -m pytest test_*.py -v
```

All tests run without a webcam or microphone — they validate shield logic, dashboard I/O, and module APIs using synthetic data.

---

## For Judges

1. **Run `./run.sh`** — this starts both the live webcam feed and the Streamlit dashboard.
2. Press **D** to enter demo mode — you'll see the unprotected vs. protected feed side by side.
3. Use **+/-** to crank perturbation strength up and down — notice how the noise texture changes.
4. Press **V** to enable voice shield — speak into your microphone and listen to the subtle shimmer.
5. Press **S** to capture a screenshot for your records.
6. Open the **dashboard** ([http://localhost:8501](http://localhost:8501)) to see real-time threat metrics.

> *The best defense against deepfakes isn't detection — it's making yourself impossible to clone in the first place.*

---

## Team

**Digital Chameleon** — AMD Slingshot Hackathon 2024

## License

MIT
