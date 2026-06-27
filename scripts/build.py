#!/usr/bin/env python3
"""PyInstaller build script for S-Stream.

Usage:
    python scripts/build.py                  # build for current platform
    python scripts/build.py --onefile        # single executable
"""

from __future__ import annotations

import subprocess
import sys


def _data_separator() -> str:
    """PyInstaller uses : on Linux/macOS, ; on Windows."""
    return ";" if sys.platform == "win32" else ":"


def main() -> None:
    onefile = "--onefile" in sys.argv
    sep = _data_separator()

    args = [
        "pyinstaller",
        "--name",
        "S-Stream",
        "--windowed",
        "--add-data",
        f"presets/scenes{sep}presets/scenes",
        "--hidden-import",
        "PySide6.QtOpenGL",
        "--hidden-import",
        "PySide6.QtOpenGLWidgets",
        "--hidden-import",
        "OpenGL",
        "--hidden-import",
        "imageio",
        "--hidden-import",
        "imageio.plugins.ffmpeg",
        "--hidden-import",
        "pyqtgraph",
    ]

    if onefile:
        args.append("--onefile")
    else:
        args.append("--onedir")

    args.append("main.py")

    print(f"Running: {' '.join(args)}")
    result = subprocess.run(args, check=True)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
