@echo off
REM GNNShield - Complete Setup and Launch Script
REM ==============================================

echo.
echo ================================================================
echo   GNNShield - DDoS Detection System
echo ================================================================
echo.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Administrator privileges required!
    echo Right-click and select "Run as administrator"
    pause
    exit /b 1
)

echo [1/5] Checking Python...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python not found!
    pause
    exit /b 1
)
python --version

echo.
echo [2/5] Checking dependencies...
python -c "import torch; import torch_geometric; import scapy; import flask" >nul 2>&1
if %errorLevel% neq 0 (
    echo [WARNING] Dependencies missing. Run install.bat first.
    pause
    exit /b 1
)
echo [OK] Dependencies installed

echo.
echo [3/5] Checking model...
if not exist "models\best_model.pth" (
    if not exist "models\latest_model.pth" (
        echo [WARNING] No trained model found!
        echo Run: python trainer.py
        pause
        exit /b 1
    )
)
echo [OK] Model found

echo.
echo [4/5] Creating required directories...
if not exist "logs" mkdir logs
if not exist "uploads" mkdir uploads
if not exist "processed_data" mkdir processed_data
echo [OK] Directories ready

echo.
echo [5/5] Starting GNNShield...
echo.
echo Dashboard: http://localhost:5000
echo Press Ctrl+C to stop
echo.

start /B python detector.py
timeout /t 2 /nobreak >nul
python dashboard.py

pause
