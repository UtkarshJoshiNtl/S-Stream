# Build S-Stream Wind Tunnel for Windows x64 (onedir bundle).
#
#   Right-click -> Run with PowerShell, or:
#   powershell -ExecutionPolicy Bypass -File .\build_windows.ps1
#
# Prereqs: Python 3.11+ (py launcher), NVIDIA GPU for running the result.
# Output:  <repo>\simulation\dist\sstream\sstream.exe

$ErrorActionPreference = 'Stop'
$REPO = (Resolve-Path "$PSScriptRoot\..\..").Path
Set-Location $REPO

if (-not (Get-Command py -ErrorAction SilentlyContinue)) {
    throw "The 'py' launcher was not found. Install Python 3.11+ from python.org (check 'Add to PATH')."
}

Write-Host "==> Creating virtual environment .venv-win"
py -3.11 -m venv .venv-win
& .\.venv-win\Scripts\Activate.ps1

Write-Host "==> Installing dependencies"
python -m pip install --upgrade pip
pip install -e .\simulation
pip install pyinstaller

Write-Host "==> Running PyInstaller"
python -m PyInstaller --clean --noconfirm "$PSScriptRoot\sstream.spec"

Write-Host ""
Write-Host "Done. Bundle: $REPO\simulation\dist\sstream\"
Write-Host "Run:          $REPO\simulation\dist\sstream\sstream.exe"
Write-Host "Ship it by zipping the dist\sstream folder (keep _internal next to sstream.exe)."
