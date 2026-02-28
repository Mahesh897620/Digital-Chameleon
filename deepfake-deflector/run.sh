#!/usr/bin/env bash
# ============================================================
#  Deepfake Deflector — Launch Script (macOS / Linux)
# ============================================================
#  Starts both the main webcam feed AND the Streamlit dashboard.
#  Usage:  ./run.sh [--no-vcam] [--demo]
# ============================================================

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

echo "============================================================"
echo "  Deepfake Deflector — Starting up"
echo "============================================================"

# --- Pre-flight checks -------------------------------------------
if ! command -v python3 &>/dev/null; then
  echo "[ERROR] python3 not found. Please install Python 3.9+."
  exit 1
fi

if ! python3 -c 'import cv2' &>/dev/null; then
  echo "[ERROR] opencv-python not installed. Run: pip3 install -r requirements.txt"
  exit 1
fi

# --- Launch Streamlit dashboard in background (optional) ----------
DASH_PID=""
if command -v streamlit &>/dev/null; then
  echo "[1/2] Starting Streamlit dashboard ..."
  streamlit run modules/dashboard.py \
      --server.headless true \
      --server.port 8501 \
      --browser.gatherUsageStats false \
      &>/dev/null &
  DASH_PID=$!
  echo "       Dashboard PID: $DASH_PID  ->  http://localhost:8501"
else
  echo "[1/2] Streamlit not found — dashboard skipped."
  echo "       Install with: pip3 install streamlit"
fi

# --- Launch main webcam feed (foreground) -------------------------
echo "[2/2] Starting main webcam feed ..."
echo ""
python3 main.py "$@"

# --- Cleanup on exit ---------------------------------------------
echo ""
if [ -n "$DASH_PID" ]; then
  echo "[INFO] Shutting down dashboard (PID $DASH_PID) ..."
  kill "$DASH_PID" 2>/dev/null || true
  wait "$DASH_PID" 2>/dev/null || true
fi
echo "[INFO] All processes stopped."
