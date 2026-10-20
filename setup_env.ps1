# PowerShell Environment Setup Script for Seraph Zenith Mark 1 - Environmental Hazards Module
$ErrorActionPreference = "Stop"

Write-Host "==========================================================" -ForegroundColor Cyan
Write-Host "  Seraph Zenith Mark 1 - Hazards Module Environment Setup" -ForegroundColor Cyan
Write-Host "==========================================================" -ForegroundColor Cyan

# 1. Check Python installation
Try {
    $pythonVer = python --version
    Write-Host "[+] Found Python: $pythonVer" -ForegroundColor Green
} Catch {
    Write-Host "[!] Python not found in PATH. Please install Python 3.10 or 3.11." -ForegroundColor Red
    Exit 1
}

# 2. Create virtual environment if it does not exist
$VENV_PATH = "hazard-env"
If (Test-Path $VENV_PATH) {
    Write-Host "[+] Virtual environment '$VENV_PATH' already exists." -ForegroundColor Yellow
} Else {
    Write-Host "[*] Creating Python virtual environment '$VENV_PATH'..." -ForegroundColor Cyan
    python -m venv $VENV_PATH
    Write-Host "[+] Virtual environment created successfully." -ForegroundColor Green
}

# 3. Upgrade pip and setuptools inside venv
$PY_EXEC = "$VENV_PATH\Scripts\python.exe"
$PIP_EXEC = "$VENV_PATH\Scripts\pip.exe"

Write-Host "[*] Upgrading pip, setuptools, wheel..." -ForegroundColor Cyan
& $PY_EXEC -m pip install --upgrade pip setuptools wheel

# 4. Install PyTorch with CUDA 12.1 support
Write-Host "[*] Installing PyTorch + torchvision with CUDA 12.1 support..." -ForegroundColor Cyan
& $PIP_EXEC install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# 5. Install Ultralytics YOLOv8, OpenCV, and essential packages
Write-Host "[*] Installing Ultralytics YOLOv8, OpenCV, PyYAML, and imaging utilities..." -ForegroundColor Cyan
& $PIP_EXEC install ultralytics opencv-python-headless pyyaml tqdm pillow matplotlib roboflow requests pandas

Write-Host "==========================================================" -ForegroundColor Green
Write-Host "  Environment setup complete! Run check_environment.py" -ForegroundColor Green
Write-Host "==========================================================" -ForegroundColor Green
