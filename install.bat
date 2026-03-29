@echo off
REM GNNShield - Installation Script
REM ================================

echo.
echo ================================================================
echo   GNNShield - Installation
echo ================================================================
echo.

net session >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Run as Administrator!
    pause
    exit /b 1
)

echo [1/6] Checking Python...
python --version >nul 2>&1
if %errorLevel% neq 0 (
    echo [ERROR] Python 3.8+ required!
    pause
    exit /b 1
)
python --version
echo.

echo [2/6] Upgrading pip...
python -m pip install --upgrade pip
echo.

echo [3/6] Installing PyTorch (CUDA 11.7)...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu117
if %errorLevel% neq 0 (
    echo [WARNING] Trying CPU version...
    pip install torch torchvision torchaudio
)
echo.

echo [4/6] Installing PyTorch Geometric...
pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cu117.html
if %errorLevel% neq 0 (
    pip install torch-geometric torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.0+cpu.html
)
echo.

echo [5/6] Installing dependencies...
pip install scapy pandas numpy flask flask-socketio python-socketio win10toast
echo.

echo [6/6] Verifying...
python -c "import torch; import torch_geometric; import scapy; import flask; print('[OK] All dependencies installed')"
if %errorLevel% neq 0 (
    echo [ERROR] Installation failed!
    pause
    exit /b 1
)
echo.

echo ================================================================
echo   Installation Complete!
echo ================================================================
echo.
echo Next steps:
echo   1. Place CICIDS dataset in data/cicids2017/
echo   2. Train: python trainer.py
echo   3. Run: run_gnnshield.bat
echo.
echo NOTE: Install Npcap from https://npcap.com
echo.
pause
