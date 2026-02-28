"""Test suite for the Dual Mode (Stealth / Visual Demo) system."""

import numpy as np
from modules.video_processor import AdversarialShield

def main():
    # Test dual-mode init
    s = AdversarialShield(perturbation_strength=8)
    assert s.demo_visual_mode is False, "default should be stealth"
    s.demo_visual_mode = True
    assert s.demo_visual_mode is True
    print("[OK] demo_visual_mode toggle")

    # Test get_demo_layers
    layers = s.get_demo_layers()
    assert len(layers) == 5, f"expected 5 layers, got {len(layers)}"
    print(f"[OK] get_demo_layers: {layers}")

    # Dummy data
    frame = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    bbox = (100, 80, 200, 200)
    face_roi = frame[80:280, 100:300].copy()

    # 1. skin tone shift
    shifted = s.subtle_skin_tone_shift(face_roi.copy())
    assert shifted.shape == face_roi.shape
    print("[OK] subtle_skin_tone_shift")

    # 2. micro geometry warp
    warped = s.micro_geometry_warp(face_roi.copy())
    assert warped.shape == face_roi.shape
    print("[OK] micro_geometry_warp")

    # 3. eye iris shimmer
    landmarks = [(320 + i % 50, 240 + i % 50) for i in range(478)]
    result = s.eye_iris_shimmer(frame.copy(), landmarks)
    assert result.shape == frame.shape
    print("[OK] eye_iris_shimmer")

    # 4. temporal face blend (first call stores prev, second blends)
    f1 = s.temporal_face_blend(frame.copy(), bbox)
    s._prev_frame = frame.copy()
    f2 = s.temporal_face_blend(frame.copy(), bbox)
    assert f2.shape == frame.shape
    print("[OK] temporal_face_blend")

    # 5. background noise injection
    bg = s.background_noise_injection(frame.copy(), bbox)
    assert bg.shape == frame.shape
    bg_none = s.background_noise_injection(frame.copy(), None)
    assert bg_none.shape == frame.shape
    print("[OK] background_noise_injection")

    # 6. Full demo shield pipeline
    s.frame_counter = 0
    demo_out = s.apply_demo_shield(frame.copy(), landmarks, bbox)
    assert demo_out.shape == frame.shape
    print("[OK] apply_demo_shield (with face)")

    # 7. Demo shield with no face
    demo_noface = s.apply_demo_shield(frame.copy(), [], None)
    assert demo_noface.shape == frame.shape
    print("[OK] apply_demo_shield (no face)")

    print()
    print("ALL DUAL-MODE TESTS PASSED")

if __name__ == "__main__":
    main()
