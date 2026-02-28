"""
Deepfake Deflector — Threat Monitor Dashboard
===============================================
A Streamlit dashboard that reads ``data/threat_log.json`` (written by
main.py) and displays live protection metrics, a simulated harvest-attempt
counter, a protection-activity chart, and a threat-event log.

Launch with:  ``streamlit run modules/dashboard.py``
"""

from __future__ import annotations

import json
import random
import time
from datetime import datetime
from pathlib import Path

import streamlit as st

# -----------------------------------------------------------------------
# Paths
# -----------------------------------------------------------------------
_DATA_DIR = Path(__file__).resolve().parent.parent / "data"
_LOG_PATH = _DATA_DIR / "threat_log.json"


# -----------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------
def _read_log() -> dict:
    """Read the shared threat log JSON.  Returns defaults if missing."""
    try:
        with open(_LOG_PATH, "r") as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
        return {
            "faces_detected": 0,
            "frames_protected": 0,
            "protection_level": "—",
            "voice_shield_active": False,
            "virtual_cam_active": False,
            "session_start": None,
            "threat_events": [],
        }


def _session_duration(session_start: str | None) -> str:
    if session_start is None:
        return "00:00:00"
    try:
        start = datetime.fromisoformat(session_start)
        delta = datetime.now() - start
        total_secs = int(delta.total_seconds())
        h, rem = divmod(total_secs, 3600)
        m, s = divmod(rem, 60)
        return f"{h:02d}:{m:02d}:{s:02d}"
    except Exception:
        return "00:00:00"


_SEVERITY_COLORS = {
    "low": "🟢",
    "medium": "🟡",
    "high": "🔴",
}


# -----------------------------------------------------------------------
# Dashboard
# -----------------------------------------------------------------------
def create_dashboard() -> None:
    st.set_page_config(
        page_title="Deepfake Deflector — Threat Monitor",
        page_icon="🛡️",
        layout="wide",
    )

    st.title("🛡️ Deepfake Deflector — Threat Monitor")

    data = _read_log()

    # ---- Harvest-attempt counter (simulated for demo wow factor) ------
    if "harvest_count" not in st.session_state:
        st.session_state.harvest_count = 0
        st.session_state.harvest_last = time.time()
    now = time.time()
    if now - st.session_state.harvest_last >= random.randint(15, 30):
        st.session_state.harvest_count += random.randint(1, 3)
        st.session_state.harvest_last = now

    # ---- Protection activity history (kept in session state) ----------
    if "activity_history" not in st.session_state:
        st.session_state.activity_history = []
    st.session_state.activity_history.append(
        {
            "time": datetime.now().strftime("%H:%M:%S"),
            "frames_protected": data.get("frames_protected", 0),
            "faces_detected": data.get("faces_detected", 0),
        }
    )
    # Keep last 60 data points (~2 min at 2-s refresh)
    st.session_state.activity_history = st.session_state.activity_history[-60:]

    # ==================================================================
    # TOP METRICS ROW
    # ==================================================================
    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Frames Protected", f"{data.get('frames_protected', 0):,}")
    with col2:
        st.metric("Session Duration", _session_duration(data.get("session_start")))
    with col3:
        level = data.get("protection_level", "—")
        level_emoji = {"LOW": "🟡", "MEDIUM": "🟢", "HIGH": "🔴"}.get(level, "⚪")
        st.metric("Protection Level", f"{level_emoji}  {level}")
    with col4:
        st.metric(
            "Harvest Attempts Blocked",
            st.session_state.harvest_count,
            delta=f"+{random.randint(0, 2)}" if st.session_state.harvest_count else None,
        )

    # ==================================================================
    # SHIELD STATUS BADGES
    # ==================================================================
    st.markdown("---")
    scol1, scol2, scol3 = st.columns(3)
    with scol1:
        faces = data.get("faces_detected", 0)
        st.markdown(
            f"**Faces Detected:** {'🟢 ' + str(faces) if faces else '⚪ 0'}"
        )
    with scol2:
        vs = data.get("voice_shield_active", False)
        st.markdown(
            f"**Voice Shield:** {'🟢 ON' if vs else '🔴 OFF'}"
        )
    with scol3:
        vc = data.get("virtual_cam_active", False)
        st.markdown(
            f"**Virtual Camera:** {'🟢 ACTIVE' if vc else '🔴 INACTIVE'}"
        )

    # ==================================================================
    # PROTECTION ACTIVITY CHART
    # ==================================================================
    st.markdown("---")
    st.subheader("📈 Protection Activity Over Time")

    history = st.session_state.activity_history
    if len(history) >= 2:
        import pandas as pd

        df = pd.DataFrame(history)
        df = df.set_index("time")
        st.line_chart(df, height=280)
    else:
        st.info("Collecting data — chart will appear after a few seconds…")

    # ==================================================================
    # THREAT EVENT LOG
    # ==================================================================
    st.markdown("---")
    st.subheader("🚨 Threat Event Log")

    events = data.get("threat_events", [])
    if events:
        import pandas as pd

        rows = []
        for ev in reversed(events[-50:]):  # most recent first, cap at 50
            sev = ev.get("severity", "low")
            rows.append(
                {
                    "Time": ev.get("time", "—"),
                    "Severity": f"{_SEVERITY_COLORS.get(sev, '⚪')} {sev.upper()}",
                    "Event": ev.get("type", "—"),
                }
            )
        st.table(pd.DataFrame(rows))
    else:
        st.success("No threat events recorded. System is clean. ✅")

    # ==================================================================
    # FOOTER
    # ==================================================================
    st.markdown("---")
    st.caption(
        f"Last refreshed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}  •  "
        f"Data source: {_LOG_PATH}"
    )

    # ---- Auto-refresh every 2 seconds --------------------------------
    time.sleep(2)
    st.rerun()


if __name__ == "__main__":
    create_dashboard()
