# -*- mode: python ; coding: utf-8 -*-
"""
PyInstaller spec for S-Stream fluid workbench.

Default: --onedir (directory with .exe + supporting files).
For a single-file build use the CLI directly:

    pyinstaller --onefile \\
        --add-data "presets/scenes:presets/scenes" \\
        --add-data "resources:resources" \\
        --hidden-import PySide6.QtOpenGL \\
        --hidden-import PySide6.QtOpenGLWidgets \\
        --hidden-import OpenGL \\
        --hidden-import numba \\
        --exclude cupy --exclude cupyx --exclude cupy_backends \\
        main.py
"""

import sys

block_cipher = None

a = Analysis(
    ["main.py"],
    pathex=[],
    binaries=[],
    datas=[
        ("presets/scenes", "presets/scenes"),
        ("resources", "resources"),
    ],
    hiddenimports=[
        "PySide6.QtOpenGL",
        "PySide6.QtOpenGLWidgets",
        "PySide6.QtWidgets",
        "PySide6.QtCore",
        "PySide6.QtGui",
        "OpenGL", "OpenGL.GL", "OpenGL.GLU",
        "imageio", "imageio.plugins.ffmpeg", "imageio.plugins.pillow",
        "pyqtgraph", "numba",
        "analysis.regimes", "analysis.sanity", "analysis.scorecard",
        "analysis.ai_context", "analysis.sweep", "analysis.physics",
        "export.image", "export.report", "export.video", "export.data",
        "scene.scene", "scene.serializer", "scene.probe",
        "engines.base", "engines.lbm_common", "engines.lbm2d",
        "workbench.app", "workbench.viewport",
        "workbench.panels.scene_panel", "workbench.panels.analysis_panel",
        "workbench.panels.outcome_panel",
        "workbench.dialogs.presets_dialog", "workbench.dialogs.recipes_dialog",
        "workbench.dialogs.sweep_dialog", "workbench.dialogs.export_dialog",
        "presets.loader",
        "resources.theme", "resources.colormaps",
    ],
    excludes=[
        "cupy", "cupyx", "cupy_backends",
        "tkinter", "matplotlib", "pandas", "pytest",
        "sphinx", "IPython", "jupyter", "notebook",
    ],
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    win_no_prefer_redirects=False,
    win_private_assemblies=False,
    cipher=block_cipher,
    noarchive=False,
)

pyz = PYZ(a.pure, a.zipped_data, cipher=block_cipher)

icon_path = None
if sys.platform == "win32":
    icon_path = "resources/icon.ico"

exe = EXE(
    pyz,
    a.scripts,
    exclude_binaries=True,
    name="S-Stream",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=True,
    console=False,
    disable_windowed_traceback=False,
    argv_emulation=False,
    icon=icon_path,
)

coll = COLLECT(
    exe,
    a.binaries,
    a.zipfiles,
    a.datas,
    strip=False,
    upx=True,
    upx_exclude=[],
    name="S-Stream",
)
