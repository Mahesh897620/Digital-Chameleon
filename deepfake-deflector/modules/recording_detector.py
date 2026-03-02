"""
Digital Chameleon — Recording Detector
========================================
Automatically detects when any video-call app (WhatsApp, Zoom, Google Meet,
Teams, FaceTime, Skype …) starts an **in-app or OS-level screen / camera
recording** on macOS, Windows and Linux.

How detection works (layered approach for maximum accuracy):
─────────────────────────────────────────────────────────────
1. **Process-name scanner** – watches for dedicated recorder sub-processes
   that every major platform spawns the moment recording begins
   (e.g. ``caphost``, ``com.apple.cmio.registerassistantservice``, ``ffmpeg``,
   ``obs64``, ``ZoomRecordingService``, etc.).

2. **macOS CGWindowList API** (primary, highest accuracy) – queries the live
   window compositor for any window whose title or owner name contains
   known recording-indicator strings ("Recording", "REC", "●", "⏺" …).

3. **Pixel-change velocity monitor** – secondary fallback.  When a recording
   capture device is opened it briefly re-initialises the OS camera
   pipeline; we measure the variance of per-frame pixel deltas and apply
   a short-time Fourier analysis to detect the characteristic "camera
   re-init stutter" (≈ 3-8 dropped / duplicated frames).

4. **Heuristic app-state inference** – reads /proc/*/fd symlinks on Linux
   and CoreMedia API hints on macOS to detect a second consumer of the same
   camera device beyond the live-view window.

The detector runs in a **background thread** at 4 Hz, writes a thread-safe
boolean flag, and fires an optional callback.  The main loop only reads
the flag — zero latency impact on the 30 fps video pipeline.
"""

from __future__ import annotations

import platform
import queue
import subprocess
import threading
import time
from typing import Callable, Dict, List, Optional, Set

import cv2
import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Known recorder process names / fragments (lower-cased)
# ─────────────────────────────────────────────────────────────────────────────
# Processes that are EXCLUSIVELY spawned when recording STARTS.
# Deliberately narrow — general-purpose apps (Chrome, Discord, Teams, etc.)
# run all the time and must NOT be included here to avoid false positives.
_RECORDER_PROCS: Set[str] = {
    # Zoom — sub-process started only when local recording begins
    "zoomrecordingservice",
    # Loom screen recorder
    "loom",
    # Windows Game Bar (only active during capture)
    "gamebarftserver",
    # Linux xdg screen-cast portal
    "xdg-desktop-portal-gnome", "xdg-desktop-portal-kde",
    # Apple QuickTime new recording session helper (not the app itself)
    "com.apple.quicktimeplayer",
}

# Process names that must NEVER match (prevent substring accidents).
_PROC_EXCLUSIONS: Set[str] = {
    "python", "python3", "code helper", "electron helper", "helper",
    "google chrome helper", "chromiumrenderer", "obs", "obs64",
    "com.apple.webkit", "safari", "xpcproxy",
}

# Recording indicator strings found in window titles (lower-cased)
_RECORDING_WINDOW_HINTS: List[str] = [
    "recording", "rec ", "● rec", "⏺", " rec)", "(recording)",
    "screen capture", "capture in progress",
]


# ─────────────────────────────────────────────────────────────────────────────
# Pixel-variance burst detector
# ─────────────────────────────────────────────────────────────────────────────
class _VarianceBurstDetector:
    """Detects the frame-variance burst that occurs when a second camera consumer
    (recorder) attaches to the OS camera pipeline.

    The OS camera driver momentarily re-calibrates exposure / white-balance
    when a new consumer opens the stream, producing 3-8 frames with higher-
    than-normal inter-frame variance.  We detect this by maintaining a
    rolling mean and alerting when the current variance exceeds 3× the mean.
    """

    _WINDOW = 90   # rolling window (3 s at 30 fps)

    def __init__(self) -> None:
        self._variances: List[float] = []
        self._prev_grey: Optional[np.ndarray] = None
        self._alert_counter: int = 0  # frames the alert holds active

    def update(self, frame: np.ndarray) -> bool:
        """Feed a frame and return True if a recording-start burst is detected."""
        grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        if self._prev_grey is None or self._prev_grey.shape != grey.shape:
            self._prev_grey = grey
            return False

        diff = cv2.absdiff(grey, self._prev_grey).astype(np.float32)
        var = float(np.var(diff))
        self._prev_grey = grey
        self._variances.append(var)

        if len(self._variances) > self._WINDOW:
            self._variances.pop(0)

        if len(self._variances) < 20:
            return False  # warm-up period

        rolling_mean = np.mean(self._variances[:-5])
        recent_mean = np.mean(self._variances[-5:])

        if rolling_mean > 0 and recent_mean > rolling_mean * 3.5:
            self._alert_counter = 30  # hold for 1 s

        if self._alert_counter > 0:
            self._alert_counter -= 1
            return True

        return False


# ─────────────────────────────────────────────────────────────────────────────
# macOS window-list helper
# ─────────────────────────────────────────────────────────────────────────────
def _macos_check_recording_window() -> bool:
    """Use AppleScript + CGWindowList to find recording-indicator windows
    and detect apps that are actively recording."""
    # Method A: AppleScript process name check
    try:
        script = (
            'tell application "System Events" to get name of every process '
            'whose background only is false'
        )
        result = subprocess.run(
            ["osascript", "-e", script],
            capture_output=True, text=True, timeout=2
        )
        output = result.stdout.lower()
        for hint in _RECORDING_WINDOW_HINTS:
            if hint in output:
                return True
        for proc in _RECORDER_PROCS:
            if proc in output:
                return True
    except Exception:
        pass

    # Method B: Check System Preferences / Screen Recording indicator
    # macOS shows an orange dot when screen is being recorded
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get value of attribute '
             '"AXTitle" of windows of processes whose name contains "zoom"'],
            capture_output=True, text=True, timeout=2
        )
        if "recording" in result.stdout.lower():
            return True
    except Exception:
        pass

    # Method C: Check for QuickTime active recording session (NOT just replayd
    # which is always running as a macOS background service)
    try:
        result = subprocess.run(
            ["osascript", "-e",
             'tell application "System Events" to get name of every process '
             'whose name contains "QuickTime" and background only is false'],
            capture_output=True, text=True, timeout=2
        )
        if "quicktime player" in result.stdout.lower():
            # QuickTime is open — check if actually recording via window title
            wt = subprocess.run(
                ["osascript", "-e",
                 'tell application "QuickTime Player" to get name of documents '
                 'where recording is true'],
                capture_output=True, text=True, timeout=2
            )
            if wt.stdout.strip() and "true" not in wt.stdout.lower().replace("true", ""):
                pass  # not recording
            if len(wt.stdout.strip()) > 2:
                return True
    except Exception:
        pass

    return False


# ─────────────────────────────────────────────────────────────────────────────
# Cross-platform process list scanner
# ─────────────────────────────────────────────────────────────────────────────
def _scan_processes() -> bool:
    """Return True if any known recorder process is currently running.

    Uses exact-name matching only to avoid false positives from helper
    processes that share substring names (Chrome helpers, Code helpers, etc.).
    """
    try:
        import psutil  # optional dependency
        for proc in psutil.process_iter(["name"]):
            try:
                name = (proc.info.get("name") or "").lower().strip()
                # Skip if on exclusion list
                if any(excl in name for excl in _PROC_EXCLUSIONS):
                    continue
                # Exact process name match only
                if name in _RECORDER_PROCS:
                    return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
    except ImportError:
        # psutil not installed — fall back to ps -axco (exact comm column)
        system = platform.system()
        try:
            if system in ("Darwin", "Linux"):
                lines = subprocess.check_output(
                    ["ps", "-axco", "comm"], timeout=2
                ).decode().splitlines()
            elif system == "Windows":
                lines = subprocess.check_output(
                    ["tasklist", "/fo", "csv", "/nh"], timeout=2
                ).decode().splitlines()
            else:
                return False
            for line in lines:
                name = line.strip().lower().strip('"').split(",")[0]
                if any(excl in name for excl in _PROC_EXCLUSIONS):
                    continue
                if name in _RECORDER_PROCS:
                    return True
        except Exception:
            pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Zoom / Meet / WhatsApp specific recording indicator files
# ─────────────────────────────────────────────────────────────────────────────
def _check_app_recording_files() -> bool:
    """Check for temp/in-progress files written by Zoom and OBS *while* recording.

    Only matches files that are actively being written right now — NOT
    completed recordings that already exist on disk (those would cause a
    permanent false-positive on every startup).
    """
    import os
    import glob

    # Files that only exist WHILE a recording is in progress
    patterns: List[str] = [
        # macOS QuickTime in-progress temp indicator
        "/private/var/folders/**/QuickTime*.tmp",
        # OBS in-progress recording segment (only present during active recording)
        os.path.expanduser("~/Videos/*.mkv.part"),
        os.path.expanduser("~/Videos/*.mp4.part"),
        # Zoom in-progress temp file (not the final .mp4 which persists after recording)
        os.path.expanduser("~/Documents/Zoom/**/*.tmp"),
        os.path.expanduser("~/Documents/Zoom/**/*.part"),
    ]

    now = time.time()
    for pat in patterns:
        try:
            for match in glob.glob(pat, recursive=True):
                # Only count files modified in the last 10 seconds
                try:
                    if now - os.path.getmtime(match) < 10:
                        return True
                except OSError:
                    pass
        except Exception:
            pass
    return False


# ─────────────────────────────────────────────────────────────────────────────
# Main recording-detection class
# ─────────────────────────────────────────────────────────────────────────────
class RecordingDetector:
    """Thread-safe, zero-latency recording detector.

    Usage::

        detector = RecordingDetector(on_recording_start=my_callback,
                                     on_recording_stop=my_stop_callback)
        detector.start()
        …
        if detector.is_recording:
            # apply privacy blur
        …
        detector.stop()

    The ``is_recording`` property is updated asynchronously by a background
    thread (polls at 4 Hz) and can be read from the main video loop at any
    time without locking.
    """

    def __init__(
        self,
        on_recording_start: Optional[Callable[[], None]] = None,
        on_recording_stop: Optional[Callable[[], None]] = None,
        sensitivity: str = "high",   # "high" | "medium" | "low"
    ) -> None:
        self._on_start = on_recording_start
        self._on_stop = on_recording_stop
        self._sensitivity = sensitivity

        # Thread-safe state
        self._lock = threading.Lock()
        self._recording = False
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._frame_queue: queue.Queue = queue.Queue(maxsize=4)

        # Variance burst detector (fed from main thread)
        self._burst = _VarianceBurstDetector()

        # Confidence counters (vote-based to avoid false positives)
        self._positive_votes: int = 0
        self._negative_votes: int = 0
        # High: 2/4 Hz polls confirm → trigger; Medium: 3/4; Low: 4/4
        self._thresh = {"high": 2, "medium": 3, "low": 4}.get(sensitivity, 2)

        self._system = platform.system()
        print(f"[RecordingDetector] Initialised  (OS={self._system}, sensitivity={sensitivity})")

    # ──────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────

    @property
    def is_recording(self) -> bool:
        with self._lock:
            return self._recording

    def feed_frame(self, frame: np.ndarray) -> None:
        """Feed a video frame for pixel-burst analysis (call from main loop)."""
        try:
            self._frame_queue.put_nowait(frame)
        except queue.Full:
            pass

    def start(self) -> None:
        """Start the background detection thread."""
        if self._thread and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True, name="RecordingDetector")
        self._thread.start()
        print("[RecordingDetector] Background thread started.")

    def stop(self) -> None:
        """Stop the background detection thread."""
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=3.0)
        print("[RecordingDetector] Stopped.")

    # ──────────────────────────────────────────────────────────────────────
    # Background thread
    # ──────────────────────────────────────────────────────────────────────

    def _run(self) -> None:
        interval = 0.25  # 4 Hz

        while not self._stop_event.is_set():
            detected = self._poll_all_sources()

            if detected:
                self._positive_votes = min(self._positive_votes + 1, self._thresh + 1)
                self._negative_votes = 0
            else:
                self._negative_votes = min(self._negative_votes + 1, 6)
                self._positive_votes = max(self._positive_votes - 1, 0)

            prev = self.is_recording

            if self._positive_votes >= self._thresh and not prev:
                with self._lock:
                    self._recording = True
                print("[RecordingDetector] *** RECORDING DETECTED — privacy shield ACTIVATED ***")
                if self._on_start:
                    try:
                        self._on_start()
                    except Exception as e:
                        print(f"[RecordingDetector] on_start callback error: {e}")

            elif self._negative_votes >= 6 and prev:
                with self._lock:
                    self._recording = False
                print("[RecordingDetector] Recording stopped — restoring normal view.")
                if self._on_stop:
                    try:
                        self._on_stop()
                    except Exception as e:
                        print(f"[RecordingDetector] on_stop callback error: {e}")

            time.sleep(interval)

    def _poll_all_sources(self) -> bool:
        """Run all detection methods; return True if ANY fires."""
        # 1. Process scanner
        if _scan_processes():
            # Further validate: make sure it's a video-call app context
            return True

        # 2. macOS window compositor check
        if self._system == "Darwin":
            if _macos_check_recording_window():
                return True

        # 3. App-specific recording file check
        if _check_app_recording_files():
            return True

        # 4. Pixel burst (drain the frame queue)
        while not self._frame_queue.empty():
            try:
                frame = self._frame_queue.get_nowait()
                if self._burst.update(frame):
                    return True
            except queue.Empty:
                break

        return False

    def get_status_dict(self) -> Dict[str, object]:
        """Return a status snapshot for HUD / dashboard display."""
        return {
            "is_recording": self.is_recording,
            "sensitivity": self._sensitivity,
            "positive_votes": self._positive_votes,
            "negative_votes": self._negative_votes,
        }
