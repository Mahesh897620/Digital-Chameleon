"""
Digital Chameleon — New Module Tests
"""
import sys
sys.path.insert(0, ".")

import numpy as np
import time

from modules.privacy_shield import (
    PrivacyShield,
    COMBINED_MODE, BLUR_MODE, PIXELATE_MODE,
    NOISE_MODE, BLACKOUT_MODE, AVATAR_MODE,
)
from modules.recording_detector import RecordingDetector

dummy = np.random.randint(0, 255, (480, 640, 3), dtype="uint8")
bbox = (120, 80, 200, 220)
all_modes = [COMBINED_MODE, BLUR_MODE, PIXELATE_MODE, NOISE_MODE, BLACKOUT_MODE, AVATAR_MODE]

# ── Test 1: PrivacyShield — all modes fade in when recording ──────────
print("=== PrivacyShield Tests ===")
for m in all_modes:
    ps = PrivacyShield(mode=m)
    for _ in range(25):
        out = ps.apply(dummy.copy(), bbox, is_recording=True)
    s = ps.get_status()
    assert s["active"], f"Not active after 25 frames: {m}"
    assert out.shape == dummy.shape, f"Shape mismatch for mode {m}"
    # Fade out
    for _ in range(25):
        out = ps.apply(dummy.copy(), bbox, is_recording=False)
    s = ps.get_status()
    assert s["fade_alpha"] < 0.1, f"Not faded out after 25 frames off: {m}"
    print(f"  [OK] mode={m}")

# ── Test 2: Privacy transform actually changes pixels ─────────────────
print()
print("=== Pixel-Change Tests ===")
solid_face = np.zeros((480, 640, 3), dtype="uint8")
solid_face[80:300, 120:320] = 200  # white face block

for m in [COMBINED_MODE, BLUR_MODE, NOISE_MODE]:
    ps = PrivacyShield(mode=m)
    for _ in range(25):
        result = ps.apply(solid_face.copy(), bbox, is_recording=True)
    face_roi = result[80:300, 120:320]
    original_roi = solid_face[80:300, 120:320]
    diff = float(np.mean(np.abs(face_roi.astype(float) - original_roi.astype(float))))
    assert diff > 5.0, f"Face region unchanged (diff={diff:.2f}) for mode {m}"
    print(f"  [OK] mode={m}  pixel_diff={diff:.1f}")

# ── Test 3: No-recording → frame passes unchanged ─────────────────────
print()
print("=== No-Recording Passthrough ===")
fixed = np.ones((480, 640, 3), dtype="uint8") * 128
ps = PrivacyShield(mode=COMBINED_MODE)
# 0 frames of recording → fade_alpha stays 0
result = ps.apply(fixed.copy(), bbox, is_recording=False)
assert float(np.mean(np.abs(result.astype(float) - fixed.astype(float)))) < 1.0
print("  [OK] Frame unchanged when not recording")

# ── Test 4: RecordingDetector lifecycle ───────────────────────────────
print()
print("=== RecordingDetector Lifecycle ===")
started = []
stopped = []

rd = RecordingDetector(
    on_recording_start=lambda: started.append(1),
    on_recording_stop=lambda: stopped.append(1),
    sensitivity="high",
)
rd.start()
time.sleep(0.8)

status = rd.get_status_dict()
assert "is_recording" in status
assert "sensitivity" in status
print(f"  [OK] Running: is_recording={rd.is_recording}")
rd.stop()
print("  [OK] Stopped cleanly")

print()
print("=== ALL TESTS PASSED ===")
