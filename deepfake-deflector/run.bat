@echo off
REM ============================================================
REM  Deepfake Deflector — Launch Script (Windows)
REM ============================================================
REM  Starts both the main webcam feed AND the Streamlit dashboard.
REM  Usage:  run.bat [--no-vcam] [--demo]
REM ============================================================

echo ============================================================
echo   Deepfake Deflector — Starting up
echo ============================================================

cd /d "%~dp0"

REM --- Pre-flight checks ------------------------------------------
python --version >nul 2>&1
if errorlevel 1 (
    echo [ERROR] python not found. Please install Python 3.9+.
    exit /b 1
)

REM --- Launch Streamlit dashboard in background (optional) --------
echo [1/2] Starting Streamlit dashboard ...
streamlit --version >nul 2>&1
if errorlevel 1 (
    echo        Streamlit not found - dashboard skipped.
    echo        Install with: pip install streamlit
) else (
    start "Deepfake Deflector Dashboard" /B streamlit run modules/dashboard.py ^
        --server.headless true ^
        --server.port 8501 ^
        --browser.gatherUsageStats false
    echo        Dashboard  -^>  http://localhost:8501
)

REM --- Launch main webcam feed (foreground) -----------------------
echo [2/2] Starting main webcam feed ...
echo.
python main.py %*

REM --- Cleanup ----------------------------------------------------
echo.
echo [INFO] Stopping dashboard ...
taskkill /FI "WINDOWTITLE eq Deepfake Deflector Dashboard" /F >nul 2>&1
echo [INFO] All processes stopped.
