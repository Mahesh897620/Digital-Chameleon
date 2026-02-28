"""Smoke-test for VoiceShimmer class."""
import numpy as np
from modules.audio_processor import VoiceShimmer

# 1. Init defaults
vs = VoiceShimmer()
assert vs.sample_rate == 44100
assert vs.chunk_size == 1024
assert vs.shimmer_amount == 0.003
assert vs.is_active is False
print(f"[OK] __init__ defaults: sr={vs.sample_rate}, chunk={vs.chunk_size}, "
      f"shimmer={vs.shimmer_amount}, active={vs.is_active}")

# 2. get_status before any processing
status = vs.get_status()
assert status == {"is_active": False, "shimmer_level": 0.003, "chunks_processed": 0}
print(f"[OK] get_status initial: {status}")

# 3. apply_shimmer on a synthetic chunk
chunk = np.sin(2 * np.pi * 440 * np.arange(1024) / 44100).astype(np.float32)
processed = vs.apply_shimmer(chunk)
assert processed.shape == chunk.shape, f"Shape mismatch: {processed.shape} vs {chunk.shape}"
assert processed.dtype == np.float32, f"Wrong dtype: {processed.dtype}"
assert not np.array_equal(processed, chunk), "Processed should differ from original"
assert np.all(processed >= -1.0) and np.all(processed <= 1.0), "Output out of [-1, 1]"
print(f"[OK] apply_shimmer: shape={processed.shape}, dtype={processed.dtype}, "
      f"max_diff={np.abs(processed - chunk).max():.6f}")

# 4. Chunks counter increments
for _ in range(5):
    vs.apply_shimmer(chunk)
status2 = vs.get_status()
assert status2["chunks_processed"] == 6  # 1 + 5
print(f"[OK] chunks_processed counter: {status2['chunks_processed']}")

# 5. apply_shimmer preserves length for various sizes
for size in [512, 1024, 2048, 4096]:
    c = np.random.randn(size).astype(np.float32) * 0.5
    p = vs.apply_shimmer(c)
    assert len(p) == size, f"Length mismatch for size {size}: {len(p)}"
print("[OK] Output length preserved for chunk sizes 512, 1024, 2048, 4096")

# 6. Verify formant jitter introduces variation across calls
results = [vs.apply_shimmer(chunk.copy()) for _ in range(5)]
# At least some pairs should differ (randomness)
diffs = [not np.array_equal(results[i], results[i + 1]) for i in range(4)]
assert any(diffs), "Expected variation across consecutive calls"
print("[OK] Consecutive calls produce varied outputs (formant jitter)")

# 7. start_stream / stop_stream without pyaudio (just logic check)
#    We don't actually open the audio device in CI — just verify no crash
#    when pyaudio is missing or device unavailable
try:
    vs.start_stream()
    vs.stop_stream()
    print("[OK] start_stream/stop_stream completed (device may or may not be available)")
except Exception as e:
    print(f"[WARN] Audio device not available (expected in CI): {e}")

# 8. Syntax check of main.py
import ast
with open("main.py") as f:
    ast.parse(f.read())
print("[OK] main.py syntax valid")

print("\n=== ALL VOICE SHIMMER TESTS PASSED ===")
