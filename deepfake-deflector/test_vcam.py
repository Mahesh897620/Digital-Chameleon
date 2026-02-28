"""Test virtual camera integration in main.py."""
import ast
import importlib
import importlib.machinery
import importlib.util
import sys
import types

# 1. Syntax check
with open("main.py") as f:
    tree = ast.parse(f.read())
print("[OK] main.py syntax valid")

# 2. Verify setup_virtual_camera function exists and is callable
sys.path.insert(0, ".")
# We need to import main as a module without running it
loader = importlib.machinery.SourceFileLoader("main_mod", "main.py")
spec = importlib.util.spec_from_loader("main_mod", loader)
main_mod = importlib.util.module_from_spec(spec)

# Patch pyvirtualcam import to avoid needing the actual driver
fake_pyvirtualcam = types.ModuleType("pyvirtualcam")
class FakeCam:
    def __init__(self, **kw):
        self.device = "/dev/fake"
        self._kw = kw
    def send(self, frame): pass
    def sleep_until_next_frame(self): pass
    def close(self): pass
fake_pyvirtualcam.Camera = FakeCam
sys.modules["pyvirtualcam"] = fake_pyvirtualcam

loader.exec_module(main_mod)

# 3. setup_virtual_camera exists and works
assert hasattr(main_mod, "setup_virtual_camera"), "setup_virtual_camera not found"
cam = main_mod.setup_virtual_camera(1280, 720, 30.0)
assert cam is not None, "setup_virtual_camera returned None with fake driver"
assert hasattr(cam, "send"), "Camera object missing send()"
assert hasattr(cam, "sleep_until_next_frame"), "Camera object missing sleep_until_next_frame()"
assert hasattr(cam, "close"), "Camera object missing close()"
print(f"[OK] setup_virtual_camera(1280, 720, 30) returned camera: {cam.device}")

# 4. Verify argparse --no-vcam is defined
import argparse
found_no_vcam = False
for node in ast.walk(tree):
    if isinstance(node, ast.Constant) and node.value == "--no-vcam":
        found_no_vcam = True
        break
assert found_no_vcam, "--no-vcam argument not found in source"
print("[OK] --no-vcam CLI argument defined")

# 5. Verify virtual cam overlay text exists in source
source = open("main.py").read()
assert 'VCam: ON' in source, "Missing ACTIVE overlay text"
assert 'VCam: OFF' in source, "Missing INACTIVE overlay text"
print("[OK] Virtual camera overlay text present")

# 6. Verify cam.send() and COLOR_BGR2RGB conversion in source
assert 'cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)' in source, "Missing BGR→RGB conversion"
assert 'vcam.send(' in source, "Missing vcam.send() call"
assert 'vcam.sleep_until_next_frame()' in source, "Missing sleep_until_next_frame()"
print("[OK] BGR→RGB conversion + send + sleep_until_next_frame in loop")

# 7. Verify cleanup closes virtual camera
assert 'vcam.close()' in source, "Missing vcam.close() in cleanup"
print("[OK] vcam.close() in cleanup section")

# 8. Verify error handling with fallback
del sys.modules["pyvirtualcam"]
# Re-import with no pyvirtualcam available
loader2 = importlib.machinery.SourceFileLoader("main_mod2", "main.py")
spec2 = importlib.util.spec_from_loader("main_mod2", loader2)
main_mod2 = importlib.util.module_from_spec(spec2)
loader2.exec_module(main_mod2)
cam2 = main_mod2.setup_virtual_camera(640, 480, 30.0)
assert cam2 is None, "Should return None when pyvirtualcam unavailable"
print("[OK] Graceful fallback when pyvirtualcam not installed")

print("\n=== ALL VIRTUAL CAMERA TESTS PASSED ===")
