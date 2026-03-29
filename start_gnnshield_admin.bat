@echo off
REM GNNShield - Dashboard Launcher with Admin Privileges
REM ====================================================

echo ========================================
echo   GNNShield - Starting Dashboard
echo ========================================
echo.

net session >nul 2>&1
if %errorLevel% == 0 (
    echo [OK] Administrator privileges active
    echo.
    python dashboard.py
) else (
    echo [!] Requesting Administrator privileges...
    echo.
    powershell -Command "Start-Process cmd -ArgumentList '/c cd /d %~dp0 && python dashboard.py && pause' -Verb RunAs"
)

pause
