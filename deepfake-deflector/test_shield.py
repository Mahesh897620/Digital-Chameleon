"""Smoke-test for AdversarialShield + FaceProcessor integration."""
import cv2
import numpy as np
from modules.video_processor import AdversarialShield, FaceProcessor

# --- AdversarialShield unit tests ---
shield = AdversarialShield(perturbation_strength=8)
print(f"[OK] AdversarialShield init: strength={shield.perturbation_strength}, "
      f"noise_pattern={shield.noise_pattern}, counter={shield.frame_counter}")

# generate_perturbation
shape = (480, 640, 3)
noise = shield.generate_perturbation(shape)
assert noise.shape == shape, f"Shape mismatch: {noise.shape}"
assert noise.dtype == np.float32, f"Wrong dtype: {noise.dtype}"
assert np.abs(noise).max() <= 8 + 0.01, f"Noise exceeded strength: {np.abs(noise).max()}"
print(f"[OK] generate_perturbation: shape={noise.shape}, max_abs={np.abs(noise).max():.2f}")

# temporal shift — noise should change every 3 frames
shield2 = AdversarialShield(perturbation_strength=8)
n1 = shield2.generate_perturbation(shape).copy()  # counter=1
n2 = shield2.generate_perturbation(shape).copy()  # counter=2 (same pattern)
n3 = shield2.generate_perturbation(shape).copy()  # counter=3 (new pattern)
assert np.array_equal(n1, n2), "Noise should be same within 3-frame window"
assert not np.array_equal(n2, n3), "Noise should change at 3-frame boundary"
print("[OK] Temporal shift works (changes every 3 frames)")

# apply_face_shield
frame = np.full((480, 640, 3), 128, dtype=np.uint8)
bbox = (100, 100, 200, 200)
protected = shield.apply_face_shield(frame.copy(), bbox)
assert protected.dtype == np.uint8
assert protected.shape == frame.shape
# ROI should differ from original
roi_orig = frame[100:300, 100:300]
roi_prot = protected[100:300, 100:300]
assert not np.array_equal(roi_orig, roi_prot), "Face ROI should be modified"
print("[OK] apply_face_shield modifies face ROI, preserves frame shape/dtype")

# apply_full_shield
full_prot = shield.apply_full_shield(frame.copy())
assert full_prot.dtype == np.uint8
assert full_prot.shape == frame.shape
assert not np.array_equal(frame, full_prot), "Full shield should modify frame"
print("[OK] apply_full_shield modifies entire frame")

# get_protection_level
for s, expected in [(1, "LOW"), (5, "LOW"), (6, "MEDIUM"), (10, "MEDIUM"), (11, "HIGH"), (15, "HIGH")]:
    sh = AdversarialShield(perturbation_strength=s)
    assert sh.get_protection_level() == expected, f"strength={s} => {sh.get_protection_level()}, expected {expected}"
print("[OK] get_protection_level: LOW/MEDIUM/HIGH thresholds correct")

# clamping
assert AdversarialShield(perturbation_strength=0).perturbation_strength == 1
assert AdversarialShield(perturbation_strength=99).perturbation_strength == 15
print("[OK] perturbation_strength clamped to [1, 15]")

# --- Integration: FaceProcessor + AdversarialShield on blank frame ---
fp = FaceProcessor()
blank = np.zeros((480, 640, 3), dtype=np.uint8)
detected, lm, bbox = fp.detect_face(blank)
if detected and bbox:
    result = shield.apply_face_shield(blank, bbox)
else:
    result = shield.apply_full_shield(blank)
fp.draw_debug(result, lm, bbox)
print("[OK] Full pipeline (detect -> shield -> draw_debug) works")

print("\n=== ALL TESTS PASSED ===")
