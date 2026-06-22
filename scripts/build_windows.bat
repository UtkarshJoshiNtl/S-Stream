@echo off
REM Build Windows executable for S-Stream
REM Requires: Python, PyInstaller, and all project deps installed

echo === Installing build dependencies ===
pip install pyinstaller

echo === Building S-Stream.exe ===
pyinstaller sstream.spec --onefile

echo === Build complete ===
echo Output: dist\S-Stream.exe
