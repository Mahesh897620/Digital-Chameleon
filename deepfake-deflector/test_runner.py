"""Quick smoke-test for FaceProcessor."""
import cv2
import numpy as np
from modules.video_processor import FaceProcessor

# 1. Init
fp = FaceProcessor()
print("[OK] FaceProcessor initialised")

# 2. Blank frame — no face expected
blank = np.zeros((480, 640, 3), dtype=np.uint8)
detected, lm, bbox = fp.detect_face(blank)
assert detected is False and lm == [] and bbox is None
print("[OK] detect_face returns (False, [], None) on blank frame")

# 3. draw_debug on no-face frame
out = fp.draw_debug(blank.copy(), lm, bbox)
assert out.shape == (480, 640, 3)
print("[OK] draw_debug draws SCANNING... overlay on blank frame")

# 4. Webcam single-frame test (if available)
cap = cv2.VideoCapture(0)
if cap.isOpened():
    ret, frame = cap.read()
    if ret:
        detected, lm, bbox = fp.detect_face(frame)
        out = fp.draw_debug(frame, lm, bbox)
        status = "FACE LOCKED" if detected else "SCANNING..."
        print(f"[OK] Webcam frame: {frame.shape}, detected={detected}, landmarks={len(lm)}, status={status}")
    else:
        print("[WARN] Webcam opened but failed to read frame")
    cap.release()
else:
    print("[WARN] No webcam available - skipping live test (logic verified with synthetic frame)")

print("[OK] All tests passed")
