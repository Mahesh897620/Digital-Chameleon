"""Test dashboard data pipeline and dashboard.py structure."""
import ast
import json
import os
import sys
import importlib
import importlib.machinery
import importlib.util
from pathlib import Path

PROJECT = Path(__file__).resolve().parent

# -----------------------------------------------------------------------
# 1. Syntax check all files
# -----------------------------------------------------------------------
for name in ["main.py", "modules/dashboard.py"]:
    with open(PROJECT / name) as f:
        ast.parse(f.read(), filename=name)
print(f"[OK] Syntax valid: main.py, modules/dashboard.py")

# -----------------------------------------------------------------------
# 2. Import write_dashboard_data from main.py
# -----------------------------------------------------------------------
sys.path.insert(0, str(PROJECT))
loader = importlib.machinery.SourceFileLoader("main_mod", str(PROJECT / "main.py"))
spec = importlib.util.spec_from_loader("main_mod", loader)
main_mod = importlib.util.module_from_spec(spec)

# Stub pyvirtualcam so import doesn't fail
import types
fake_pyvirtualcam = types.ModuleType("pyvirtualcam")
class FakeCam:
    def __init__(self, **kw): self.device = "/dev/fake"
    def send(self, f): pass
    def sleep_until_next_frame(self): pass
    def close(self): pass
fake_pyvirtualcam.Camera = FakeCam
sys.modules["pyvirtualcam"] = fake_pyvirtualcam

loader.exec_module(main_mod)
assert hasattr(main_mod, "write_dashboard_data"), "write_dashboard_data not found"
print("[OK] write_dashboard_data function exists in main.py")

# -----------------------------------------------------------------------
# 3. Call write_dashboard_data and verify JSON output
# -----------------------------------------------------------------------
test_events = [
    {"time": "12:00:00", "type": "face_harvest_attempt", "severity": "high"},
    {"time": "12:00:05", "type": "voice_clone_probe", "severity": "medium"},
]
main_mod.write_dashboard_data(
    frames_protected=1234,
    faces_detected=42,
    protection_level="MEDIUM",
    voice_shield_active=True,
    virtual_cam_active=False,
    session_start="2026-03-01T12:00:00",
    threat_events=test_events,
)
log_path = PROJECT / "data" / "threat_log.json"
assert log_path.exists(), f"Expected {log_path} to exist"
with open(log_path) as f:
    data = json.load(f)

assert data["faces_detected"] == 42
assert data["frames_protected"] == 1234
assert data["protection_level"] == "MEDIUM"
assert data["voice_shield_active"] is True
assert data["virtual_cam_active"] is False
assert data["session_start"] == "2026-03-01T12:00:00"
assert len(data["threat_events"]) == 2
assert data["threat_events"][0]["severity"] == "high"
print(f"[OK] threat_log.json written and verified: {dict(data)}")

# -----------------------------------------------------------------------
# 4. Verify dashboard.py reads the JSON correctly
# -----------------------------------------------------------------------
dash_loader = importlib.machinery.SourceFileLoader(
    "dash_mod", str(PROJECT / "modules" / "dashboard.py")
)
dash_spec = importlib.util.spec_from_loader("dash_mod", dash_loader)
dash_mod = importlib.util.module_from_spec(dash_spec)
# We can't actually run Streamlit, but we can import helper functions
# by patching st minimally
fake_st = types.ModuleType("streamlit")
fake_st.set_page_config = lambda **kw: None
fake_st.session_state = {}
sys.modules["streamlit"] = fake_st
dash_loader.exec_module(dash_mod)

result = dash_mod._read_log()
assert result["frames_protected"] == 1234
assert result["protection_level"] == "MEDIUM"
print("[OK] dashboard._read_log() reads threat_log.json correctly")

dur = dash_mod._session_duration("2026-03-01T12:00:00")
assert dur != "00:00:00", f"Duration should be non-zero, got {dur}"
print(f"[OK] _session_duration works: {dur}")

dur_none = dash_mod._session_duration(None)
assert dur_none == "00:00:00"
print("[OK] _session_duration(None) returns 00:00:00")

# -----------------------------------------------------------------------
# 5. Verify dashboard source has all required components
# -----------------------------------------------------------------------
src = (PROJECT / "modules" / "dashboard.py").read_text()
checks = [
    ("Threat Monitor", "title"),
    ("Frames Protected", "metric"),
    ("Session Duration", "metric"),
    ("Protection Level", "metric"),
    ("Harvest Attempts Blocked", "metric"),
    ("line_chart", "chart"),
    ("Threat Event Log", "section"),
    ("st.rerun()", "auto-refresh"),
]
for text, desc in checks:
    assert text in src, f"Missing {desc}: '{text}'"
print(f"[OK] Dashboard source has all {len(checks)} required components")

# -----------------------------------------------------------------------
# 6. Verify run.sh and run.bat exist
# -----------------------------------------------------------------------
assert (PROJECT / "run.sh").exists(), "run.sh missing"
assert (PROJECT / "run.bat").exists(), "run.bat missing"
assert os.access(PROJECT / "run.sh", os.X_OK), "run.sh not executable"
print("[OK] run.sh (executable) and run.bat exist")

# -----------------------------------------------------------------------
# 7. Verify main.py writes every 60 frames
# -----------------------------------------------------------------------
main_src = (PROJECT / "main.py").read_text()
assert "frame_index % 60 == 0" in main_src, "Missing 60-frame write trigger"
assert "write_dashboard_data(" in main_src, "Missing write_dashboard_data call in loop"
print("[OK] main.py calls write_dashboard_data every 60 frames")

print("\n=== ALL DASHBOARD TESTS PASSED ===")
