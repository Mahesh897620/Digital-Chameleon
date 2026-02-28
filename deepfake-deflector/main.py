"""
Deepfake Deflector - Main Entry Point
======================================
Opens the default webcam, displays a live feed with an FPS counter,
and exits cleanly when 'q' is pressed.

Keyboard shortcuts:
  q   — Quit
  d   — Toggle demo split-screen mode
  m   — Toggle shield mode (Stealth ↔ Visual Demo)
  v   — Toggle voice shield
  h   — Toggle HUD overlay
  s   — Save screenshot to screenshots/
  +/= — Increase perturbation strength
  -   — Decrease perturbation strength
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

# -----------------------------------------------------------------------
# Graceful imports — the app keeps running even if optional modules are
# missing.  The startup_check() function reports what's available.
# -----------------------------------------------------------------------
try:
    from modules.video_processor import FaceProcessor
except ImportError:
    FaceProcessor = None  # type: ignore[assignment,misc]

try:
    from modules.video_processor import AdversarialShield
except ImportError:
    AdversarialShield = None  # type: ignore[assignment,misc]

try:
    from modules.audio_processor import VoiceShimmer
except ImportError:
    VoiceShimmer = None  # type: ignore[assignment,misc]

# -----------------------------------------------------------------------
# Constants
# -----------------------------------------------------------------------
_VERSION = "1.0"
_SCREENSHOTS_DIR = os.path.join(os.path.dirname(__file__), "screenshots")


# -----------------------------------------------------------------------
# Startup dependency check
# -----------------------------------------------------------------------
def startup_check() -> Dict[str, bool]:
    """Probe each optional component and print a status table.

    Returns a dict mapping component name → availability bool.
    """
    status: Dict[str, bool] = {}

    # -- Webcam --
    try:
        cap = cv2.VideoCapture(0)
        ok = cap.isOpened()
        cap.release()
        status["Webcam"] = ok
    except Exception:
        status["Webcam"] = False

    # -- MediaPipe --
    try:
        import mediapipe  # noqa: F401
        status["MediaPipe"] = True
    except ImportError:
        status["MediaPipe"] = False

    # -- PyAudio --
    try:
        import pyaudio  # noqa: F401
        status["PyAudio"] = True
    except ImportError:
        status["PyAudio"] = False

    # -- pyvirtualcam --
    try:
        import pyvirtualcam  # noqa: F401
        status["pyvirtualcam"] = True
    except ImportError:
        status["pyvirtualcam"] = False

    # -- Streamlit --
    try:
        import streamlit  # noqa: F401
        status["Streamlit"] = True
    except ImportError:
        status["Streamlit"] = False

    # -- scipy --
    try:
        import scipy  # noqa: F401
        status["scipy"] = True
    except ImportError:
        status["scipy"] = False

    # Print table
    print()
    print("=" * 50)
    print("  Deepfake Deflector — Component Status")
    print("=" * 50)
    for name, available in status.items():
        icon = "\u2713" if available else "\u2717"
        colour_label = "ready" if available else "missing"
        print(f"  {icon}  {name:<16} {colour_label}")
    print("=" * 50)
    print()

    return status

# -----------------------------------------------------------------------
# Dashboard data writer
# -----------------------------------------------------------------------
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
) -> None:
    """Write current session state to *data/threat_log.json* for the dashboard."""
    os.makedirs(_DATA_DIR, exist_ok=True)
    payload = {
        "faces_detected": faces_detected,
        "frames_protected": frames_protected,
        "protection_level": protection_level,
        "voice_shield_active": voice_shield_active,
        "virtual_cam_active": virtual_cam_active,
        "session_start": session_start,
        "threat_events": threat_events[-200:],  # cap at 200
    }
    try:
        with open(_LOG_PATH, "w") as f:
            json.dump(payload, f, indent=2)
    except OSError as exc:
        print(f"[WARN] Could not write dashboard data: {exc}")


# -----------------------------------------------------------------------
# Screenshot helper
# -----------------------------------------------------------------------
def save_screenshot(frame: np.ndarray) -> str:
    """Save *frame* to the ``screenshots/`` folder. Returns the file path."""
    os.makedirs(_SCREENSHOTS_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = os.path.join(_SCREENSHOTS_DIR, f"deflector_{ts}.png")
    cv2.imwrite(path, frame)
    return path


# -----------------------------------------------------------------------
# Demo-mode drawing helpers
# -----------------------------------------------------------------------
def _draw_scanning_line(frame: np.ndarray, frame_index: int) -> None:
    """Draw a red horizontal scanning line that sweeps top→bottom."""
    h, w = frame.shape[:2]
    y = frame_index % h
    # Main line
    cv2.line(frame, (0, y), (w, y), (0, 0, 255), 2)
    # Fading trail above
    for offset in range(1, 20):
        alpha = max(0, 1.0 - offset / 20.0)
        yy = y - offset
        if 0 <= yy < h:
            overlay = frame[yy : yy + 1, :].copy()
            red_line = np.full_like(overlay, (0, 0, 255), dtype=np.uint8)
            cv2.addWeighted(red_line, alpha * 0.4, overlay, 1 - alpha * 0.4, 0, overlay)
            frame[yy : yy + 1, :] = overlay


def _draw_harvest_blocked(frame: np.ndarray, frame_index: int) -> None:
    """Flash 'HARVEST BLOCKED' on the frame every ~60 frames."""
    if (frame_index // 20) % 3 == 0:
        h, w = frame.shape[:2]
        text = "HARVEST BLOCKED"
        scale = 0.9
        thickness = 2
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
        cx, cy = (w - tw) // 2, h // 2
        # Dark background pill
        cv2.rectangle(frame, (cx - 10, cy - th - 10), (cx + tw + 10, cy + 10), (0, 0, 0), -1)
        cv2.putText(frame, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 255, 0), thickness, cv2.LINE_AA)


def build_demo_frame(
    raw_frame: np.ndarray,
    protected_frame: np.ndarray,
    frame_index: int,
    landmarks: Optional[list] = None,
    bounding_box: Optional[tuple] = None,
    demo_visual: bool = False,
) -> np.ndarray:
    """Create a side-by-side demo frame: UNPROTECTED (left) | PROTECTED (right).

    When *demo_visual* is True the left pane shows animated red dots
    crawling over detected face landmarks (simulating AI harvesting)
    and the right pane shows the same dots scattering / glitching away
    with "HARVEST FAILED ✗" flashing.
    """
    h, w = raw_frame.shape[:2]

    # --- Left side: unprotected with scanning + harvest animation ---
    left = raw_frame.copy()
    _draw_scanning_line(left, frame_index * 3)

    if landmarks and bounding_box:
        # Draw animated red dots crawling over the face (AI harvesting)
        n_dots = min(len(landmarks), (frame_index * 4) % (len(landmarks) + 40))
        for i in range(n_dots):
            if i < len(landmarks):
                lx, ly = landmarks[i]
                cv2.circle(left, (lx, ly), 2, (0, 0, 255), -1)
        # Connecting lines between key landmarks (extraction mesh)
        key_ids = [159, 386, 1, 61, 291]
        pts = [(landmarks[k][0], landmarks[k][1]) for k in key_ids if k < len(landmarks)]
        for i in range(len(pts)):
            for j in range(i + 1, len(pts)):
                cv2.line(left, pts[i], pts[j], (0, 0, 200), 1, cv2.LINE_AA)
        # Pulsing box around face
        bx, by, bw, bh = bounding_box
        pulse = int(4 * abs(np.sin(frame_index * 0.1)))
        cv2.rectangle(left, (bx - pulse, by - pulse),
                       (bx + bw + pulse, by + bh + pulse), (0, 0, 255), 2)
        cv2.putText(left, "EXTRACTING IDENTITY...", (bx, by - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1, cv2.LINE_AA)

    # Red border
    cv2.rectangle(left, (0, 0), (w - 1, h - 1), (0, 0, 255), 3)
    # Label
    cv2.rectangle(left, (0, 0), (280, 40), (0, 0, 180), -1)
    cv2.putText(left, "UNPROTECTED", (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    # Threat text
    cv2.putText(left, "AI Harvester Active", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

    # --- Right side: protected with glitching landmarks ---
    right = protected_frame.copy()

    if demo_visual and landmarks and bounding_box:
        # Draw *scattered / glitching* dots — same landmarks but with
        # random offsets that increase over time, simulating failure
        scatter = 5 + int(10 * abs(np.sin(frame_index * 0.15)))
        for i, (lx, ly) in enumerate(landmarks):
            dx = np.random.randint(-scatter, scatter + 1)
            dy = np.random.randint(-scatter, scatter + 1)
            # Alternate colours: green→yellow→cyan for glitch feel
            colours = [(0, 255, 0), (0, 255, 255), (255, 255, 0)]
            clr = colours[i % 3]
            cv2.circle(right, (lx + dx, ly + dy), 2, clr, -1)
        # Flash "HARVEST FAILED ✗"
        if (frame_index // 15) % 2 == 0:
            text = "HARVEST FAILED X"
            (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)
            cx = (w - tw) // 2
            cy = h // 2
            cv2.rectangle(right, (cx - 12, cy - th - 12), (cx + tw + 12, cy + 12), (0, 0, 0), -1)
            cv2.putText(right, text, (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2, cv2.LINE_AA)
        # "Shield layers active" badge
        cv2.rectangle(right, (0, h - 35), (260, h), (0, 80, 0), -1)
        cv2.putText(right, "5 SHIELD LAYERS ACTIVE", (8, h - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    else:
        _draw_harvest_blocked(right, frame_index)

    # Green border
    cv2.rectangle(right, (0, 0), (w - 1, h - 1), (0, 255, 0), 3)
    # Label
    mode_label = "PROTECTED [VISUAL]" if demo_visual else "PROTECTED"
    label_w = 310 if demo_visual else 250
    cv2.rectangle(right, (0, 0), (label_w, 40), (0, 140, 0), -1)
    cv2.putText(right, mode_label, (10, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)

    # --- Combine ---
    combined = np.hstack((left, right))
    return combined


# -----------------------------------------------------------------------
# HUD drawing
# -----------------------------------------------------------------------
def draw_hud(
    frame: np.ndarray,
    fps: float,
    shield: AdversarialShield,
    voice_active: bool,
    vcam_active: bool,
    demo_mode: bool,
) -> None:
    """Draw the Heads-Up Display (top bar, bottom bar, shortcuts)."""
    h, w = frame.shape[:2]

    # --- Top bar ---
    cv2.rectangle(frame, (0, 0), (w, 36), (30, 30, 30), -1)
    title = f"DEEPFAKE DEFLECTOR v{_VERSION}  --  DIGITAL CHAMELEON PROTOCOL ACTIVE"
    cv2.putText(frame, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 180), 1, cv2.LINE_AA)
    fps_txt = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_txt, (w - 130, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1, cv2.LINE_AA)

    # --- Status indicators (below top bar) ---
    y0 = 60
    level = shield.get_protection_level()
    level_color = {"LOW": (0, 255, 255), "MEDIUM": (0, 165, 255), "HIGH": (0, 0, 255)}[level]
    cv2.putText(frame, f"Shield: {level}", (10, y0), cv2.FONT_HERSHEY_SIMPLEX, 0.6, level_color, 2, cv2.LINE_AA)

    # Shield mode indicator
    mode_txt = "Mode: VISUAL" if shield.demo_visual_mode else "Mode: STEALTH"
    mode_clr = (0, 255, 255) if shield.demo_visual_mode else (0, 200, 0)
    cv2.putText(frame, mode_txt, (10, y0 + 28), cv2.FONT_HERSHEY_SIMPLEX, 0.6, mode_clr, 2, cv2.LINE_AA)

    vs_txt = "Voice: ON" if voice_active else "Voice: OFF"
    vs_clr = (0, 255, 0) if voice_active else (0, 0, 255)
    cv2.putText(frame, vs_txt, (10, y0 + 56), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vs_clr, 2, cv2.LINE_AA)

    vc_txt = "VCam: ON" if vcam_active else "VCam: OFF"
    vc_clr = (0, 255, 0) if vcam_active else (0, 0, 255)
    cv2.putText(frame, vc_txt, (10, y0 + 84), cv2.FONT_HERSHEY_SIMPLEX, 0.6, vc_clr, 2, cv2.LINE_AA)

    dm_txt = "Demo: ON" if demo_mode else "Demo: OFF"
    dm_clr = (0, 255, 255) if demo_mode else (128, 128, 128)
    cv2.putText(frame, dm_txt, (10, y0 + 112), cv2.FONT_HERSHEY_SIMPLEX, 0.6, dm_clr, 2, cv2.LINE_AA)

    # --- Demo visual layer indicators (right side) ---
    if shield.demo_visual_mode:
        layers = shield.get_demo_layers()
        lx = w - 260
        ly_start = 60
        cv2.rectangle(frame, (lx - 8, ly_start - 18), (w - 4, ly_start + len(layers) * 24 + 4), (20, 20, 20), -1)
        cv2.putText(frame, "ACTIVE LAYERS:", (lx, ly_start),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
        for i, layer_name in enumerate(layers):
            ly = ly_start + 22 + i * 24
            # Pulsing dot
            dot_clr = (0, 255, 0)
            cv2.circle(frame, (lx + 6, ly - 5), 5, dot_clr, -1)
            cv2.putText(frame, layer_name, (lx + 18, ly),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 180), 1, cv2.LINE_AA)

    # --- Bottom bar: perturbation strength slider + shortcuts ---
    bar_h = 50
    cv2.rectangle(frame, (0, h - bar_h), (w, h), (30, 30, 30), -1)

    # Strength slider
    strength = shield.perturbation_strength
    label = f"Strength: {strength}/15"
    cv2.putText(frame, label, (10, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (200, 200, 200), 1, cv2.LINE_AA)
    bar_x = 160
    bar_w = 200
    bar_y = h - 32
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (80, 80, 80), -1)
    filled = int(bar_w * strength / 15)
    bar_color = (0, 255, 0) if strength <= 5 else (0, 165, 255) if strength <= 10 else (0, 0, 255)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + filled, bar_y + 14), bar_color, -1)
    cv2.rectangle(frame, (bar_x, bar_y), (bar_x + bar_w, bar_y + 14), (200, 200, 200), 1)

    # Shortcuts
    shortcuts = "[Q]uit [D]emo [M]ode [V]oice [H]UD [S]creenshot [+/-]Strength"
    cv2.putText(frame, shortcuts, (bar_x + bar_w + 20, h - 18), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (160, 160, 160), 1, cv2.LINE_AA)


# ----------------------------------------------------------------------
# Virtual camera helpers
# ----------------------------------------------------------------------
_VCAM_SETUP_MSG = """
============================================================
  Virtual Camera Setup Instructions
============================================================
  macOS:
    1. Install OBS Studio  -> https://obsproject.com
    2. Start OBS and enable "Start Virtual Camera"
    3. pip3 install pyvirtualcam

  Windows:
    1. Install OBS Studio  -> https://obsproject.com
    2. OBS ships with a virtual camera driver (auto-installed)
    3. pip3 install pyvirtualcam

  Linux:
    1. sudo apt install v4l2loopback-dkms
    2. sudo modprobe v4l2loopback devices=1
    3. pip3 install pyvirtualcam
============================================================
"""


def setup_virtual_camera(
    width: int, height: int, fps: float
) -> Optional[object]:
    """Try to open a pyvirtualcam virtual camera.

    Returns a ``pyvirtualcam.Camera`` instance on success, or ``None``
    if the library or driver is unavailable.
    """
    try:
        import pyvirtualcam  # type: ignore[import-untyped]

        cam = pyvirtualcam.Camera(width=width, height=height, fps=fps)
        print(f"[INFO] Virtual camera started: {cam.device}  ({width}x{height} @ {fps:.0f} fps)")
        return cam
    except ImportError:
        print("[WARN] pyvirtualcam is not installed. Run: pip3 install pyvirtualcam")
        print(_VCAM_SETUP_MSG)
        return None
    except Exception as exc:
        print(f"[WARN] Could not start virtual camera: {exc}")
        print("[HINT] Make sure the virtual camera driver is installed (see instructions below).")
        print(_VCAM_SETUP_MSG)
        return None


# ======================================================================
# Main loop
# ======================================================================
def main():
    # --- CLI arguments ---
    parser = argparse.ArgumentParser(description="Deepfake Deflector -- live feed")
    parser.add_argument(
        "--no-vcam",
        action="store_true",
        help="Disable virtual camera output (useful for testing)",
    )
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Start in demo split-screen mode",
    )
    parser.add_argument(
        "--no-check",
        action="store_true",
        help="Skip the startup dependency check",
    )
    args = parser.parse_args()

    # --- Startup dependency check ---
    if not args.no_check:
        dep_status = startup_check()
        if not dep_status.get("Webcam", False):
            print("[ERROR] No webcam detected. Please connect a camera and try again.")
            return
    else:
        dep_status = {}

    # --- Open the default webcam (index 0) ---
    try:
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            raise RuntimeError("VideoCapture(0) returned a closed handle")
    except Exception as exc:
        print(f"[ERROR] Could not open webcam: {exc}")
        print("[HINT] Check your camera connection and permissions (System Preferences > Privacy > Camera).")
        return

    print("[INFO] Webcam opened. Controls: q=quit  d=demo  m=mode  v=voice  h=HUD  s=screenshot  +/-=strength")

    # --- Webcam resolution (used for virtual camera) ---
    cam_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    cam_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cam_fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    print(f"[INFO] Webcam resolution: {cam_w}x{cam_h} @ {cam_fps:.0f} fps")

    # --- Initialise face processor (graceful if MediaPipe missing) ---
    face_proc = None
    if FaceProcessor is not None:
        try:
            face_proc = FaceProcessor()
            print("[INFO] FaceProcessor initialised (MediaPipe Face Mesh).")
        except Exception as exc:
            print(f"[WARN] FaceProcessor init failed: {exc}")
    else:
        print("[WARN] FaceProcessor unavailable (mediapipe not installed).")

    # --- Initialise adversarial shield (always available — pure numpy) ---
    shield = None
    if AdversarialShield is not None:
        try:
            shield = AdversarialShield(perturbation_strength=8)
            print(f"[INFO] AdversarialShield active -- protection level: {shield.get_protection_level()}")
        except Exception as exc:
            print(f"[WARN] AdversarialShield init failed: {exc}")
    else:
        print("[WARN] AdversarialShield unavailable.")

    # --- Initialise voice shimmer (graceful if PyAudio/scipy missing) ---
    voice_shimmer = None
    if VoiceShimmer is not None:
        try:
            voice_shimmer = VoiceShimmer()
            print("[INFO] VoiceShimmer ready (press 'v' to toggle).")
        except Exception as exc:
            print(f"[WARN] VoiceShimmer init failed: {exc}")
    else:
        print("[WARN] VoiceShimmer unavailable (check scipy / pyaudio).")

    # --- Virtual camera ---
    vcam = None  # type: Optional[object]
    vcam_active = False
    if not args.no_vcam:
        vcam = setup_virtual_camera(cam_w, cam_h, cam_fps)
        vcam_active = vcam is not None
    else:
        print("[INFO] Virtual camera disabled (--no-vcam).")

    # --- UI state ---
    demo_mode: bool = args.demo
    hud_visible: bool = True
    screenshot_flash: int = 0  # countdown frames for flash overlay

    # --- Dashboard session state ---
    session_start = datetime.now().isoformat()
    frames_protected = 0
    total_faces_detected = 0
    threat_events: List[Dict[str, Any]] = []
    frame_index = 0

    # FPS tracking variables
    prev_time = time.time()
    fps = 0.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break

        # Calculate FPS
        current_time = time.time()
        elapsed = current_time - prev_time
        if elapsed > 0:
            fps = 1.0 / elapsed
        prev_time = current_time

        # Keep a raw copy BEFORE any processing (used in demo mode)
        raw_frame = frame.copy()

        # --- Face detection (skipped if FaceProcessor unavailable) ---
        face_detected = False
        landmarks: list = []
        bbox = None
        if face_proc is not None:
            try:
                face_detected, landmarks, bbox = face_proc.detect_face(frame)
            except Exception:
                pass  # transient detection error — skip this frame

        # --- Adversarial shield (skipped if AdversarialShield unavailable) ---
        if shield is not None:
            if shield.demo_visual_mode:
                # Visual demo mode: apply all 5 visible layers + stealth noise
                frame = shield.apply_demo_shield(frame, landmarks, bbox)
            elif face_detected and bbox is not None:
                frame = shield.apply_face_shield(frame, bbox)
            else:
                frame = shield.apply_full_shield(frame)

        if face_proc is not None:
            face_proc.draw_debug(frame, landmarks, bbox)

        # --- Session accounting ---
        frames_protected += 1
        if face_detected:
            total_faces_detected += 1
        frame_index += 1

        # --- Write dashboard data every 60 frames ---
        if frame_index % 60 == 0:
            write_dashboard_data(
                frames_protected=frames_protected,
                faces_detected=total_faces_detected,
                protection_level=shield.get_protection_level() if shield else "OFF",
                voice_shield_active=voice_shimmer.is_active if voice_shimmer else False,
                virtual_cam_active=vcam_active,
                session_start=session_start,
                threat_events=threat_events,
            )

        # =================================================================
        # Build display frame — either demo split-screen or normal view
        # =================================================================
        if demo_mode:
            display = build_demo_frame(
                raw_frame, frame, frame_index,
                landmarks=landmarks,
                bounding_box=bbox,
                demo_visual=shield.demo_visual_mode if shield else False,
            )
        else:
            display = frame

        # --- HUD overlay ---
        if hud_visible and shield is not None:
            draw_hud(
                display,
                fps,
                shield,
                voice_shimmer.is_active if voice_shimmer else False,
                vcam_active,
                demo_mode,
            )
        else:
            # Minimal FPS even when HUD is hidden or shield unavailable
            cv2.putText(display, f"FPS: {fps:.1f}", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)

        # --- Screenshot flash ---
        if screenshot_flash > 0:
            overlay = display.copy()
            cv2.rectangle(overlay, (0, 0), (display.shape[1], display.shape[0]), (255, 255, 255), -1)
            alpha = min(screenshot_flash / 8.0, 0.5)
            cv2.addWeighted(overlay, alpha, display, 1 - alpha, 0, display)
            screenshot_flash -= 1

        # --- Send protected frame to virtual camera ---
        if vcam_active and vcam is not None:
            try:
                # Always send the single protected frame (not split-screen)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                vcam.send(rgb_frame)      # type: ignore[union-attr]
                vcam.sleep_until_next_frame()  # type: ignore[union-attr]
            except Exception as exc:
                print(f"[WARN] Virtual camera send failed: {exc}")
                vcam_active = False

        # --- Display ---
        title = "Deepfake Deflector - DEMO" if demo_mode else "Deepfake Deflector - Live Feed"
        cv2.imshow(title, display)

        # ==============================================================
        # Key handling
        # ==============================================================
        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            print("[INFO] Exiting...")
            break
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
        elif key == ord("d"):
            demo_mode = not demo_mode
            cv2.destroyAllWindows()  # reset window for new size
            print(f"[INFO] Demo mode {'ON' if demo_mode else 'OFF'}")
        elif key == ord("m"):
            if shield is not None:
                shield.demo_visual_mode = not shield.demo_visual_mode
                mode_name = "VISUAL DEMO" if shield.demo_visual_mode else "STEALTH"
                print(f"[INFO] Shield mode: {mode_name}")
            else:
                print("[WARN] AdversarialShield not available.")
        elif key == ord("h"):
            hud_visible = not hud_visible
            print(f"[INFO] HUD {'visible' if hud_visible else 'hidden'}")
        elif key == ord("s"):
            path = save_screenshot(display)
            screenshot_flash = 8
            print(f"[INFO] Screenshot saved: {path}")
        elif key in (ord("+"), ord("=")):
            if shield is not None:
                shield.perturbation_strength = min(15, shield.perturbation_strength + 1)
                print(f"[INFO] Perturbation strength: {shield.perturbation_strength}")
        elif key == ord("-"):
            if shield is not None:
                shield.perturbation_strength = max(1, shield.perturbation_strength - 1)
                print(f"[INFO] Perturbation strength: {shield.perturbation_strength}")

    # Final dashboard write
    write_dashboard_data(
        frames_protected=frames_protected,
        faces_detected=total_faces_detected,
        protection_level=shield.get_protection_level() if shield else "OFF",
        voice_shield_active=False,
        virtual_cam_active=False,
        session_start=session_start,
        threat_events=threat_events,
    )

    # Cleanup
    if voice_shimmer is not None:
        voice_shimmer.stop_stream()
    if vcam is not None:
        try:
            vcam.close()  # type: ignore[union-attr]
            print("[INFO] Virtual camera closed.")
        except Exception:
            pass
    cap.release()
    cv2.destroyAllWindows()
    print("[INFO] Resources released. Goodbye!")


if __name__ == "__main__":
    main()
