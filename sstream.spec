# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec file for S-Stream fluid simulation workbench.

Usage:
    pyinstaller sstream.spec              # Build as directory (default)
    pyinstaller sstream.spec --onefile    # Build as single executable
"""

import os
import sys
from pathlib import Path

# Get the project root directory
block_cipher = None

# Analysis phase
a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[
        # Preset scenes
        ("presets/scenes", "presets/scenes"),
        # Resources (icons, etc.)
        ("resources", "resources"),
    ],
    hiddenimports=[
        # PySide6 Qt modules
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PySide6.QtGui",
        # OpenGL
        "OpenGL",
        "OpenGL.GL",
        "OpenGL.GLU",
        # Image/Video
        "imageio",
        "imageio.plugins.ffmpeg",
        "imageio.plugins.pillow",
        # Plotting
        "pyqtgraph",
        "numpy",
        "scipy",
        # Analysis modules
        "analysis.regimes",
        "analysis.sanity",
        "analysis.scorecard",
        "analysis.ai_context",
        "analysis.sweep",
        "analysis.physics",
        # Export modules
        "export.image",
        "export.report",
        "export.video",
        "export.data",
        # Scene modules
        "scene.scene",
        "scene.serializer",
        "scene.probe",
        # Engine modules
        "engines.base",
        "engines.lbm_common",
        "engines.lbm2d",
        # Workbench modules
        "workbench.app",
        "workbench.viewport",
        "workbench.panels.scene_panel",
        "workbench.panels.analysis_panel",
        "workbench.panels.outcome_panel",
        "workbench.dialogs.presets_dialog",
        "workbench.dialogs.recipes_dialog",
        "workbench.dialogs.sweep_dialog",
        "workbench.dialogs.export_dialog",
        # Presets
        "presets.loader",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=[
        # Exclude unnecessary modules to reduce size
        "tkinter",
        "matplotlib",
        "pandas",
        "pytest",
        "sphinx",
        "IPython",
        "jupyter",
        "notebook",
    ],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

# Collect all binaries and datas
pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="S-Stream",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,  # Windowed mode (no console)
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    # Icon placeholder - uncomment when icon is available
    # icon="resources/icon.icns" if sys.platform == "darwin" else "resources/icon.ico",
)

# coll = COLLECT(
#     exe,
#     a.binaries,
#     a.zipfiles,
#     a.datas,
#     strip=False,
#     upx=True,
#     upx_exclude=[],
#     name="S-Stream",
# )

# One-file build (single .exe)
exe = EXE(
    pyz,
    a.scripts,
    a.binaries,
    a.zipfiles,
    a.datas,
    [],
    name="S-Stream",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch=None,
    codesign_identity=None,
    entitlements_file=None,
    icon="resources/icon.ico",
)
