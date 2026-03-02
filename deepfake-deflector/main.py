"""
Digital Chameleon — Main Entry Point
======================================
The Digital Chameleon acts as an invisible privacy layer between you and
anyone you video-call on WhatsApp, Zoom, Google Meet, Teams, FaceTime, etc.

HOW IT WORKS
─────────────
         ┌─────────────────────────────────────────────────┐
         │                 YOUR WEBCAM                      │
         └──────────────────────┬──────────────────────────┘
                                │  raw frames
                                ▼
         ┌─────────────────────────────────────────────────┐
         │            DIGITAL CHAMELEON LAYER               │
         │  ┌──────────────┐  ┌───────────────────────┐   │
         │  │ Face Detect  │  │  Recording Detector   │   │
         │  │ (MediaPipe)  │  │  (Zoom/Meet/WA/etc.)  │   │
         │  └──────┬───────┘  └──────────┬────────────┘   │
         │         │ bbox                 │ is_recording?   │
         │         ▼                      ▼                 │
         │  ┌────────────────────────────────────────────┐  │
         │  │           PRIVACY SHIELD                   │  │
         │  │  NOT recording → pass frame through clean  │  │
         │  │  IS  recording → blur / pixelate / noise   │  │
         │  └─────────────────┬──────────────────────────┘  │
         │                    │ protected frame              │
         │  ┌─────────────────▼──────────────────────────┐  │
         │  │        ADVERSARIAL SHIELD (always on)      │  │
         │  │  Stealth Gaussian noise against deepfakes  │  │
         │  └─────────────────┬──────────────────────────┘  │
         └────────────────────┼────────────────────────────-┘
                              │
                              ▼
         ┌─────────────────────────────────────────────────┐
         │          VIRTUAL CAMERA DRIVER                   │
         │  (OBS / v4l2loopback)  ← Zoom/Meet sees this    │
         └─────────────────────────────────────────────────┘

• During a normal call   → caller sees YOUR REAL FACE.
• The moment recording   → your face is automatically replaced with
  starts (auto-detected)   the privacy transform (blur / pixelate /
                           noise / avatar).  The recorded video will
                           never contain your real face.
• When recording stops   → instantly restored, caller sees real face again.
• Fully automatic        — no manual action needed.

Keyboard shortcuts:
  q   — Quit
  r   — Toggle recording simulation (for testing without a real recorder)
  p   — Cycle privacy mode  (combined→blur→pixelate→noise→blackout→avatar)
  d   — Toggle demo split-screen
  m   — Toggle adversarial stealth / visual mode
  v   — Toggle voice shield
  h   — Toggle HUD overlay
  s   — Save screenshot
  +   — Increase adversarial perturbation strength
  -   — Decrease adversarial perturbation strength
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from typing import Any, Dict, List, Optional

import cv2
import numpy as np

# ─────────────────────────────────────────────────────────────────────────────
# Graceful optional imports
# ─────────────────────────────────────────────────────────────────────────────
try:
    from modules.video_processor import FaceProcessor, AdversarialShield
except ImportError:
    FaceProcessor = None     # type: ignore[assignment,misc]
    AdversarialShield = None # type: ignore[assignment,misc]

try:
    from modules.audio_processor import VoiceShimmer
except ImportError:
    VoiceShimmer = None      # type: ignore[assignment,misc]

try:
    from modules.recording_detector import RecordingDetector
except ImportError:
    RecordingDetector = None # type: ignore[assignment,misc]

try:
    from modules.privacy_shield import (
        PrivacyShield,
        BLUR_MODE, PIXELATE_MODE, NOISE_MODE,
        BLACKOUT_MODE, AVATAR_MODE, COMBINED_MODE,
    )
except ImportError:
    PrivacyShield = None     # type: ignore[assignment,misc]
    BLUR_MODE = PIXELATE_MODE = NOISE_MODE = "combined"
    BLACKOUT_MODE = AVATAR_MODE = COMBINED_MODE = "combined"

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────
_VERSION = "2.0"
_SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")
_PRIVACY_MODES_CYCLE = [COMBINED_MODE, BLUR_MODE, PIXELATE_MODE,
                         NOISE_MODE, BLACKOUT_MODE, AVATAR_MODE]


# ─────────────────────────────────────────────────────────────────────────────
# Camera index auto-detection
# ─────────────────────────────────────────────────────────────────────────────
def find_camera_index(max_index: int = 6, read_timeout: float = 1.5) -> int:
    """Return the best camera index, preferring the built-in system camera.

    On macOS with Continuity Camera (iPhone as webcam) the phone cameras
    occupy index 0 and 1 at 1920×1080, while the built-in FaceTime HD
    Camera sits at a higher index and natively reports 1280×720.
    We scan all indices, collect working ones, and prefer whichever has a
    native height of 720 px (built-in) before falling back to the first
    working camera.
    """
    import threading as _t

    working: List[int] = []
    dims: dict = {}

    for idx in range(max_index):
        cap = cv2.VideoCapture(idx)
        if not cap.isOpened():
            cap.release()
            continue
        result: list = []
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        def _probe(cap=cap, result=result):
            import time
            time.sleep(0.3)
            ok, _ = cap.read()
            result.append(ok)
        th = _t.Thread(target=_probe, daemon=True)
        th.start()
        th.join(timeout=read_timeout)
        cap.release()
        if result and result[0]:
            working.append(idx)
            dims[idx] = (w, h)

    if not working:
        return 0  # fallback

    # Prefer the built-in camera: natively 1280×720 on all MacBook models.
    # Continuity Camera (iPhone) reports 1920×1080 or higher.
    for idx in working:
        if dims[idx][1] == 720:
            print(f"[INFO] Auto-selected built-in camera at index {idx} "
                  f"({dims[idx][0]}×{dims[idx][1]})")
            return idx

    # Fallback: use the last working index (usually built-in on macOS
    # because Continuity Camera registers before the built-in camera).
    fallback = working[-1]
    print(f"[INFO] Camera auto-select fallback: index {fallback} "
          f"({dims[fallback][0]}×{dims[fallback][1]})")
    return fallback


# ─────────────────────────────────────────────────────────────────────────────
# Startup dependency check
# ─────────────────────────────────────────────────────────────────────────────
def startup_check() -> Dict[str, bool]:
    """Probe each optional component and print a status table."""
    status: Dict[str, bool] = {}

    try:
        _cam_idx = find_camera_index()
        cap = cv2.VideoCapture(_cam_idx)
        ok = cap.isOpened()
        cap.release()
        status["Webcam"] = ok
    except Exception:
        status["Webcam"] = False

    for lib, key in [
        ("mediapipe",    "MediaPipe"),
        ("pyaudio",      "PyAudio"),
        ("pyvirtualcam", "pyvirtualcam"),
        ("streamlit",    "Streamlit"),
        ("scipy",        "scipy"),
        ("psutil",       "psutil"),
    ]:
        try:
            __import__(lib)
            status[key] = True
        except ImportError:
            status[key] = False

    status["RecordingDetector"] = RecordingDetector is not None
    status["PrivacyShield"]     = PrivacyShield is not None

    print()
    print("=" * 58)
    print(f"  Digital Chameleon v{_VERSION} — Component Status")
    print("=" * 58)
    for name, available in status.items():
        icon = "\u2713" if available else "\u2717"
        print(f"  {icon}  {name:<22} {'ready' if available else 'missing'}")
    print("=" * 58)
    print()
    return status


# ─────────────────────────────────────────────────────────────────────────────
# Dashboard data writer
# ─────────────────────────────────────────────────────────────────────────────
_DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
_LOG_PATH = os.path.join(_DATA_DIR, "threat_log.json")


def write_dashboard_data(
    frames_protected: int,
    faces_detected: int,
    protection_level: str,
    voice_shield_active: bool,
    virtual_cam_active: bool,
    session_start: str,
    threat_events: List[Dict[str, Any]],
    recording_active: bool = False,
    privacy_mode: str = "off",
) -> None:
    """Write current session state to data/threat_log.json for the dashboard."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    payload = {
        "faces_detected":      faces_detected,
        "frames_protected":    frames_protected,
        "protection_level":    protection_level,
        "voice_shield_active": voice_shield_active,
        "virtual_cam_active":  virtual_cam_active,
        "session_start":       session_start,
        "threat_events":       threat_events[-200:],
        "recording_active":    recording_active,
        "privacy_mode":        privacy_mode,
        "timestamp":           datetime.now().isoformat(),
    }
    try:
        with open(_LOG_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        print(f"[WARN] Could not write dashboard data: {exc}")


# ─────────────────────────────────────────────────────────────────────────────
# Screenshot helper
# ─────────────────────────────────────────────────────────────────────────────
def save_screenshot(frame: np.ndarray) -> str:
    """Save frame to screenshots/ folder. Returns the file path."""
    os.makedirs(_SCREENSHOTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_SCREENSHOTS_DIR, f"chameleon_{ts}.png")
    cv2.imwrite(path, frame)
    return path


# ─────────────────────────────────────────────────────────────────────────────
# HUD drawing
# ─────────────────────────────────────────────────────────────────────────────
def draw_hud(
    frame: np.ndarray,
    fps: float,
    shield: Optional[Any],
    voice_active: bool,
    vcam_active: bool,
    demo_mode: bool,
    is_recording: bool,
    privacy_mode: str,
    privacy_fade: float,
) -> None:
    """Draw the Digital Chameleon Heads-Up Display."""
    h, w = frame.shape[:2]

    # Top bar
    cv2.rectangle(frame, (0, 0), (w, 38), (20, 20, 20), -1)
    title = f"DIGITAL CHAMELEON v{_VERSION} — AUTO PRIVACY LAYER"
    cv2.putText(frame, title, (10, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.52,
                (0, 255, 180), 1, cv2.LINE_AA)
    cv2.putText(frame, f"FPS: {fps:.1f}", (w - 130, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 255, 0), 1, cv2.LINE_AA)

    y0 = 62

    # Recording status indicator
    if is_recording:
        pulse = int(80 * abs(np.sin(time.time() * 4)))
        rc = (0, 0, 180 + pulse)
        cv2.rectangle(frame, (0, y0 - 18), (255, y0 + 8), rc, -1)
        cv2.putText(frame, "● REC DETECTED — SHIELD ON", (6, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    else:
        cv2.putText(frame, "● STANDBY — Live Call Mode", (6, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.52, (0, 220, 80), 1, cv2.LINE_AA)
    y0 += 28

    # Adversarial shield level
    if shield is not None:
        level = shield.get_protection_level()
        lc = {"LOW": (0, 255, 255), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}[level]
        cv2.putText(frame, f"Adv.Shield: {level}", (10, y0),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, lc, 1, cv2.LINE_AA)
        mode_txt = "Mode: VISUAL" if shield.demo_visual_mode else "Mode: STEALTH"
        mode_clr = (0, 255, 255) if shield.demo_visual_mode else (0, 200, 0)
        cv2.putText(frame, mode_txt, (10, y0 + 22), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, mode_clr, 1, cv2.LINE_AA)
        y0 += 46

    # Privacy mode
    pm_colour = (0, 100, 255) if is_recording else (100, 100, 100)
    cv2.putText(frame, f"Privacy: {privacy_mode.upper()} ({privacy_fade*100:.0f}%)",
                (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, pm_colour, 1, cv2.LINE_AA)
    y0 += 26

    # Voice / VCam
    vc = "ON" if voice_active else "OFF"
    vc_c = (0, 255, 0) if voice_active else (80, 80, 80)
    cv2.putText(frame, f"Voice: {vc}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, vc_c, 1, cv2.LINE_AA)
    y0 += 24
    vm = "ON" if vcam_active else "OFF"
    vm_c = (0, 255, 0) if vcam_active else (80, 80, 80)
    cv2.putText(frame, f"VCam: {vm}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, vm_c, 1, cv2.LINE_AA)
    y0 += 24
    dm_txt = "Demo: ON" if demo_mode else "Demo: OFF"
    dm_clr = (0, 255, 255) if demo_mode else (80, 80, 80)
    cv2.putText(frame, dm_txt, (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.55, dm_clr, 1, cv2.LINE_AA)

    # Demo visual layer indicators (right side)
    if shield is not None and shield.demo_visual_mode:
        layers = shield.get_demo_layers()
        lx = w - 260
        ly_start = 60
        cv2.rectangle(frame, (lx - 8, ly_start - 18), (w - 4, ly_start + len(layers) * 24 + 4), (20, 20, 20), -1)
        cv2.putText(frame, "ACTIVE LAYERS:", (lx, ly_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        for i, layer_name in enumerate(layers):
            ly = ly_start + 22 + i * 24
            cv2.circle(frame, (lx + 6, ly - 5), 5, (0, 255, 0), -1)
            cv2.putText(frame, layer_name, (lx + 18, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)

    # Bottom bar
    bar_h = 50
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (20, 20, 20), -1)
    if shield is not None:
        strength = shield.perturbation_strength
        label = f"Strength: {strength}/15"
        cv2.putText(frame, label, (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        bar_x, bar_w_px, bar_y = 170, 200, h - 34
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_px, bar_y + 14), (80, 80, 80), -1)
        filled = int(bar_w_px * strength / 15)
        bar_color = (0, 255, 0) if strength <= 5 else (0, 165, 255) if strength <= 10 else (0, 0, 255)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 14), bar_color, -1)
        cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w_px, bar_y + 14), (200, 200, 200), 1)

    shortcuts = "[Q]quit [R]sim-rec [P]privacy [D]demo [M]adv [V]voice [H]HUD [S]shot [+/-]str"
    cv2.putText(frame, shortcuts, (10, h - 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.36, (120, 120, 120), 1, cv2.LINE_AA)


# ─────────────────────────────────────────────────────────────────────────────
# Demo split-screen
# ─────────────────────────────────────────────────────────────────────────────
def build_demo_frame(
    raw_frame: np.ndarray,
    output_frame: np.ndarray,
    frame_index: int,
    is_recording: bool,
    landmarks: Optional[list] = None,
    bounding_box: Optional[tuple] = None,
) -> np.ndarray:
    """Side-by-side: LEFT = live (what caller sees), RIGHT = recorded (what recording captures)."""
    h, w = raw_frame.shape[:2]

    # Left: live clean feed
    left = raw_frame.copy()
    cv2.rectangle(left, (0, 0), (w - 1, h - 1), (0, 200, 0), 3)
    cv2.rectangle(left, (0, 0), (w, 38), (0, 100, 0), -1)
    cv2.putText(left, "LIVE (CALLER SEES THIS)", (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    if bounding_box and landmarks:
        bx, by, bw_f, bh_f = bounding_box
        cv2.rectangle(left, (bx, by), (bx + bw_f, by + bh_f), (0, 255, 0), 2)
        cv2.putText(left, "REAL FACE", (bx, by - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

    # Right: recording output
    right = output_frame.copy()
    if is_recording:
        border_colour = (0, 0, 200)
        label_right = "RECORDED — FACE HIDDEN"
        pulse_r = 200 + int(55 * abs(np.sin(frame_index * 0.15)))
        cv2.circle(right, (w - 30, 20), 10, (0, 0, pulse_r), -1)
        cv2.putText(right, "REC", (w - 58, 26), cv2.FONT_HERSHEY_SIMPLEX,
                    0.5, (0, 0, 255), 1, cv2.LINE_AA)
        cv2.putText(right, "IDENTITY PROTECTED", (10, h - 15),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 1, cv2.LINE_AA)
    else:
        border_colour = (200, 200, 0)
        label_right = "RECORDING READY (standby)"

    cv2.rectangle(right, (0, 0), (w - 1, h - 1), border_colour, 3)
    cv2.rectangle(right, (0, 0), (w, 38), (80, 40, 0), -1)
    cv2.putText(right, label_right, (8, 26),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)

    return np.hstack((left, right))


# ─────────────────────────────────────────────────────────────────────────────
# Virtual camera setup
# ─────────────────────────────────────────────────────────────────────────────
_VCAM_SETUP_MSG = """
╔══════════════════════════════════════════════════════════╗
║            Virtual Camera Setup                          ║
╠══════════════════════════════════════════════════════════╣
║  macOS:                                                  ║
║    1. Install OBS Studio → https://obsproject.com        ║
║    2. In OBS: Tools → Virtual Camera → Start             ║
║    3. pip3 install pyvirtualcam                          ║
║                                                          ║
║  Windows:                                                ║
║    1. Install OBS Studio (includes virtual cam driver)   ║
║    2. pip3 install pyvirtualcam                          ║
║                                                          ║
║  Linux:                                                  ║
║    sudo modprobe v4l2loopback devices=1                  ║
║    pip3 install pyvirtualcam                             ║
╚══════════════════════════════════════════════════════════╝
"""


def setup_virtual_camera(width: int, height: int, fps: float) -> Optional[object]:
    """Try to open a pyvirtualcam virtual camera. Returns instance or None."""
    try:
        import pyvirtualcam
        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
        print(f"[INFO] Virtual camera: {cam.device}  ({width}x{height} @ {fps:.0f}fps)")
        return cam
    except ImportError:
        print("[WARN] pyvirtualcam not installed. Run: pip3 install pyvirtualcam")
        print(_VCAM_SETUP_MSG)
        return None
    except Exception as exc:
        print(f"[WARN] Virtual camera failed: {exc}")
        print(_VCAM_SETUP_MSG)
        return None


# ─────────────────────────────────────────────────────────────────────────────
# Main loop
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(description="Digital Chameleon — Auto Privacy Layer")
    parser.add_argument("--no-vcam",       action="store_true", help="Disable virtual camera")
    parser.add_argument("--demo",          action="store_true", help="Start in demo split-screen")
    parser.add_argument("--no-check",      action="store_true", help="Skip dependency check")
    parser.add_argument("--privacy-mode",  default=COMBINED_MODE,
                        choices=_PRIVACY_MODES_CYCLE,
                        help=f"Privacy transform (default: {COMBINED_MODE})")
    parser.add_argument("--sensitivity",   default="high",
                        choices=["high", "medium", "low"],
                        help="Recording detection sensitivity (default: high)")
    parser.add_argument("--strength",      type=int, default=8,
                        help="Adversarial perturbation strength 1-15 (default: 8)")
    parser.add_argument("--camera",        type=int, default=-1,
                        help="Camera index to use (default: auto-detect built-in)")
    args = parser.parse_args()

    # Startup check
    if not args.no_check:
        dep = startup_check()
        if not dep.get("Webcam", False):
            print("[ERROR] No webcam detected.")
            return
    else:
        dep: Dict[str, bool] = {}

    # Open webcam — use explicit index if given, otherwise auto-detect built-in.
    _cam_index = args.camera if args.camera >= 0 else find_camera_index()
    try:
        cap = cv2.VideoCapture(_cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"VideoCapture({_cam_index}) returned closed handle")
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # minimise latency / buffer overflow
        # Cap resolution to 1280×720 — keeps display usable and speeds up processing.
        # MediaPipe inference is done on a 640-px-wide copy regardless.
        cap.set(cv2.CAP_PROP_FRAME_WIDTH,  1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT,  720)
    except Exception as exc:
        print(f"[ERROR] Webcam: {exc}")
        return

    cam_w   = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h   = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    _raw_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    cam_fps = max(15.0, min(120.0, _raw_fps))  # clamp: bad drivers report 0 or 1
    print(f"[INFO] Webcam: {cam_w}×{cam_h} @ {cam_fps:.0f} fps")
    print("[INFO] Controls: q=quit  r=sim-rec  p=privacy  d=demo  m=adv  v=voice  h=HUD  s=shot  +/-=str")

    # Face processor
    face_proc = None
    if FaceProcessor is not None:
        try:
            face_proc = FaceProcessor()
            print("[INFO] FaceProcessor ready (MediaPipe).")
        except Exception as exc:
            print(f"[WARN] FaceProcessor: {exc}")

    # Adversarial shield
    shield = None
    if AdversarialShield is not None:
        try:
            shield = AdversarialShield(perturbation_strength=args.strength)
            print(f"[INFO] AdversarialShield: {shield.get_protection_level()}")
        except Exception as exc:
            print(f"[WARN] AdversarialShield: {exc}")

    # Privacy shield
    privacy = None
    if PrivacyShield is not None:
        try:
            privacy = PrivacyShield(mode=args.privacy_mode)
            print(f"[INFO] PrivacyShield: mode={args.privacy_mode}")
        except Exception as exc:
            print(f"[WARN] PrivacyShield: {exc}")

    # Recording detector
    rec_detector = None
    threat_events: List[Dict[str, Any]] = []

    def _on_recording_start() -> None:
        threat_events.append({
            "type": "recording_detected",
            "ts": datetime.now().isoformat(),
            "action": "privacy_shield_activated",
        })

    def _on_recording_stop() -> None:
        threat_events.append({
            "type": "recording_stopped",
            "ts": datetime.now().isoformat(),
            "action": "privacy_shield_deactivated",
        })

    if RecordingDetector is not None:
        try:
            rec_detector = RecordingDetector(
                on_recording_start=_on_recording_start,
                on_recording_stop=_on_recording_stop,
                sensitivity=args.sensitivity,
            )
            rec_detector.start()
            print(f"[INFO] RecordingDetector started (sensitivity={args.sensitivity}).")
        except Exception as exc:
            print(f"[WARN] RecordingDetector: {exc}")

    # Voice shimmer
    voice_shimmer = None
    if VoiceShimmer is not None:
        try:
            voice_shimmer = VoiceShimmer()
            print("[INFO] VoiceShimmer ready (press 'v' to toggle).")
        except Exception as exc:
            print(f"[WARN] VoiceShimmer: {exc}")

    # Webcam warm-up — discard the first N frames so the sensor stabilises
    # before the virtual camera driver opens (avoids pipeline-reset read failures)
    print("[INFO] Warming up webcam…", end=" ", flush=True)
    time.sleep(0.5)  # allow macOS camera pipeline to fully initialise
    _warmup_ok = 0

    import threading as _threading
    def _warmup_read(results: list) -> None:
        """Read up to 20 warmup frames, storing success count."""
        count = 0
        for _ in range(20):
            ok, _ = cap.read()
            if ok:
                count += 1
        results.append(count)

    _warmup_results: list = []
    _warmup_thread = _threading.Thread(target=_warmup_read, args=(_warmup_results,), daemon=True)
    _warmup_thread.start()
    _warmup_thread.join(timeout=5.0)  # give macOS AVFoundation at most 5 s to deliver frames
    _warmup_ok = _warmup_results[0] if _warmup_results else 0
    print(f"OK ({_warmup_ok}/20 frames)")

    # Virtual camera
    vcam = None
    vcam_active = False
    if not args.no_vcam:
        vcam = setup_virtual_camera(cam_w, cam_h, cam_fps)
        vcam_active = vcam is not None

    # Post-vcam warm-up: drain any stale webcam buffer that accumulated while
    # the virtual camera driver was initialising (takes ~0.5 s on Apple M1)
    if vcam_active:
        time.sleep(0.4)
        for _ in range(5):
            cap.read()  # discard stale frames

    # UI state
    demo_mode:     bool = args.demo
    hud_visible:   bool = True
    sim_recording: bool = False   # 'r' key: simulate recording for testing
    screenshot_flash: int = 0
    privacy_mode_idx: int = _PRIVACY_MODES_CYCLE.index(args.privacy_mode)

    session_start   = datetime.now().isoformat()
    frames_protected = 0
    total_faces      = 0
    frame_index      = 0

    prev_time = time.time()
    fps = 0.0

    print()
    print("╔══════════════════════════════════════════════════════╗")
    print("║  Digital Chameleon is ACTIVE                         ║")
    print("║  • Real face shown during normal live calls          ║")
    print("║  • Face auto-hidden the INSTANT recording starts     ║")
    print("║  • Restored INSTANTLY when recording stops           ║")
    print("║  • Press [R] in preview window to test it now        ║")
    print("╚══════════════════════════════════════════════════════╝")
    print()

    # Pre-create the display window as WINDOW_NORMAL so it:
    #  1. Appears reliably on macOS regardless of frame dimensions.
    #  2. Can be freely resized by the user.
    #  3. Is displayed at a sensible default size even on large monitors.
    _win_title_live = "Digital Chameleon — Live"
    _win_title_demo = "Digital Chameleon — DEMO"
    _windows_created: set = set()  # track which windows have been initialised

    def _ensure_window(title: str, w: int, h: int) -> None:
        """Create the window ONCE, only when we have a real frame to show."""
        if title not in _windows_created:
            cv2.namedWindow(title, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(title, min(w, 1280), min(h, 720))
            _windows_created.add(title)

    _read_fails = 0
    while True:
        ret, raw_frame = cap.read()
        if not ret:
            _read_fails += 1
            if _read_fails <= 10:
                time.sleep(0.05)
                continue  # transient glitch — retry
            # Try to reconnect the webcam
            print("[WARN] Webcam read failing — attempting reconnect…")
            cap.release()
            time.sleep(1.0)
            cap = cv2.VideoCapture(_cam_index)
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            if not cap.isOpened():
                print("[ERROR] Webcam reconnect failed — exiting.")
                break
            _read_fails = 0
            print("[INFO] Webcam reconnected.")
            continue
        _read_fails = 0  # reset on success

        # FPS
        now = time.time()
        elapsed = now - prev_time
        fps = 1.0 / elapsed if elapsed > 0 else fps
        prev_time = now

        # Face detection
        face_detected = False
        landmarks: list = []
        bbox = None
        if face_proc is not None:
            try:
                face_detected, landmarks, bbox = face_proc.detect_face(raw_frame)
            except Exception:
                pass

        # Feed frame to recording detector (pixel-burst analysis)
        if rec_detector is not None:
            rec_detector.feed_frame(raw_frame)

        # Determine recording status (real detector OR manual simulation)
        is_recording = sim_recording
        if rec_detector is not None:
            is_recording = is_recording or rec_detector.is_recording

        # ─── Step 1: Privacy Shield ───────────────────────────────────────
        # NOT recording → frame passes clean (real face visible to caller)
        # IS  recording → face region is replaced by the privacy transform
        frame = raw_frame.copy()
        if privacy is not None:
            frame = privacy.apply(frame, bbox, is_recording, landmarks)
        elif is_recording and bbox is not None:
            # Fallback heavy blur when PrivacyShield module is unavailable
            x1 = max(0, bbox[0] - 30)
            y1 = max(0, bbox[1] - 30)
            x2 = min(frame.shape[1], bbox[0] + bbox[2] + 30)
            y2 = min(frame.shape[0], bbox[1] + bbox[3] + 30)
            roi = frame[y1:y2, x1:x2]
            k = max(51, (roi.shape[0] // 4) | 1)
            frame[y1:y2, x1:x2] = cv2.GaussianBlur(roi, (k, k), 0)

        # ─── Step 2: Adversarial Shield (always on) ───────────────────────
        if shield is not None:
            if shield.demo_visual_mode:
                frame = shield.apply_demo_shield(frame, landmarks, bbox)
            elif face_detected and bbox is not None:
                frame = shield.apply_face_shield(frame, bbox)
            else:
                frame = shield.apply_full_shield(frame)

        # Face debug overlay (only in live mode, not when recording)
        if face_proc is not None and not is_recording:
            face_proc.draw_debug(frame, landmarks, bbox)

        # Session counters
        frames_protected += 1
        if face_detected:
            total_faces += 1
        frame_index += 1

        # ─── Virtual camera output ────────────────────────────────────────
        # The virtual camera sends the PRIVACY-PROTECTED frame to the call app.
        # Zoom/Meet/WhatsApp reads THIS as "your camera".
        vcam_frame = frame.copy()
        if vcam_active and vcam is not None:
            try:
                rgb = cv2.cvtColor(vcam_frame, cv2.COLOR_BGR2RGB)
                vcam.send(rgb)                    # type: ignore[union-attr]
                vcam.sleep_until_next_frame()     # type: ignore[union-attr]
            except Exception as exc:
                print(f"[WARN] VCam send failed: {exc}")
                vcam_active = False

        # Build display frame
        if demo_mode:
            display = build_demo_frame(
                raw_frame, vcam_frame, frame_index,
                is_recording, landmarks, bbox,
            )
        else:
            display = frame

        # HUD
        if hud_visible:
            privacy_fade = (privacy.get_status()["fade_alpha"]
                            if privacy else (1.0 if is_recording else 0.0))
            cur_privacy_mode = privacy.mode if privacy else args.privacy_mode
            draw_hud(
                display, fps, shield,
                voice_shimmer.is_active if voice_shimmer else False,
                vcam_active, demo_mode,
                is_recording, cur_privacy_mode, privacy_fade,
            )
        else:
            cv2.putText(display, f"FPS:{fps:.0f}", (8, 28),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2, cv2.LINE_AA)
            if is_recording:
                cv2.putText(display, "REC SHIELD", (8, 56),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2, cv2.LINE_AA)

        # Screenshot flash
        if screenshot_flash > 0:
            ov = display.copy()
            cv2.rectangle(ov, (0, 0), (display.shape[1], display.shape[0]), (255, 255, 255), -1)
            cv2.addWeighted(ov, 0.4, display, 0.6, 0, display)
            screenshot_flash -= 1

        # Dashboard data (every 60 frames)
        if frame_index % 60 == 0:
            write_dashboard_data(
                frames_protected=frames_protected,
                faces_detected=total_faces,
                protection_level=shield.get_protection_level() if shield else "OFF",
                voice_shield_active=voice_shimmer.is_active if voice_shimmer else False,
                virtual_cam_active=vcam_active,
                session_start=session_start,
                threat_events=threat_events,
                recording_active=is_recording,
                privacy_mode=privacy.mode if privacy else "off",
            )

        # Show window — create lazily on first frame so no blank window appears.
        win_title = _win_title_demo if demo_mode else _win_title_live
        disp_w = display.shape[1]
        disp_h = display.shape[0]
        _ensure_window(win_title, disp_w, disp_h)
        cv2.imshow(win_title, display)

        # Key handling
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            print("[INFO] Quitting…")
            break

        elif key == ord("r"):
            sim_recording = not sim_recording
            state = "ON (simulated)" if sim_recording else "OFF"
            print(f"[INFO] Simulated recording: {state}")
            if sim_recording:
                threat_events.append({
                    "type": "recording_simulated",
                    "ts": datetime.now().isoformat(),
                    "action": "privacy_shield_test",
                })

        elif key == ord("p"):
            if privacy is not None:
                privacy_mode_idx = (privacy_mode_idx + 1) % len(_PRIVACY_MODES_CYCLE)
                new_mode = _PRIVACY_MODES_CYCLE[privacy_mode_idx]
                privacy.set_mode(new_mode)
                print(f"[INFO] Privacy mode: {new_mode}")

        elif key == ord("d"):
            demo_mode = not demo_mode
            cv2.destroyAllWindows()
            _windows_created.clear()  # force re-creation on next frame
            print(f"[INFO] Demo mode: {'ON' if demo_mode else 'OFF'}")

        elif key == ord("m"):
            if shield is not None:
                shield.demo_visual_mode = not shield.demo_visual_mode
                print(f"[INFO] Adv.shield: {'VISUAL' if shield.demo_visual_mode else 'STEALTH'}")

        elif key == ord("v"):
            if voice_shimmer is not None:
                if voice_shimmer.is_active:
                    voice_shimmer.stop_stream()
                    print("[INFO] Voice shield OFF")
                else:
                    voice_shimmer.start_stream()
                    print("[INFO] Voice shield ON")
            else:
                print("[WARN] VoiceShimmer not available.")

        elif key == ord("h"):
            hud_visible = not hud_visible

        elif key == ord("s"):
            path = save_screenshot(display)
            screenshot_flash = 8
            print(f"[INFO] Screenshot: {path}")

        elif key in (ord("+"), ord("=")):
            if shield is not None:
                shield.perturbation_strength = min(15, shield.perturbation_strength + 1)
                print(f"[INFO] Strength: {shield.perturbation_strength}")

        elif key == ord("-"):
            if shield is not None:
                shield.perturbation_strength = max(1, shield.perturbation_strength - 1)
                print(f"[INFO] Strength: {shield.perturbation_strength}")

    # Final dashboard write
    write_dashboard_data(
        frames_protected=frames_protected,
        faces_detected=total_faces,
        protection_level=shield.get_protection_level() if shield else "OFF",
        voice_shield_active=False,
        virtual_cam_active=False,
        session_start=session_start,
        threat_events=threat_events,
        recording_active=False,
        privacy_mode=privacy.mode if privacy else "off",
    )

    # Cleanup
    if rec_detector is not None:
        rec_detector.stop()
    if voice_shimmer is not None:
        voice_shimmer.stop_stream()
    if vcam is not None:
        try:
            vcam.close()   # type: ignore[union-attr]
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Digital Chameleon shutdown complete. Goodbye!")


if __name__ == "__main__":
    main()

