# -*- mode: python ; coding: utf-8 -*-
"""PyInstaller spec for the S-Stream headless solver (Windows x64, onedir).

Build it from anywhere with:

    pyinstaller simulation/packaging/sstream.spec

The resulting bundle lands in `simulation/dist/sstream/`:

    dist/sstream/sstream.exe        console launcher (progress prints)
    dist/sstream/_internal/         Python, VTK, Warp, assets/

Requires on the build machine: Python 3.11+, pip install pyinstaller,
plus the project dependencies (pip install -e ./simulation).
Requires at runtime (Windows): NVIDIA GPU + current driver; no CUDA
toolkit needed (Warp ships its own runtime; kernels JIT at startup).
"""

import os
from PyInstaller.utils.hooks import collect_all

# SPECPATH is injected by PyInstaller: directory that holds this spec file.
SPEC = SPECPATH
ROOT = os.path.abspath(os.path.join(SPEC, ".."))  # simulation/

datas = [
    # Default model and any future bundled assets.
    (os.path.join(ROOT, "assets"), "assets"),
]
binaries = []
hiddenimports = []

# Warp: native `warp` binaries + runtime data (JIT CUDA compiler).
# vtkmodules/pyvista: used by the voxelizer (VTK image-stencil path).
for package in ("warp", "vtkmodules", "pyvista"):
    d, b, h = collect_all(package)
    datas += d
    binaries += b
    hiddenimports += h

a = Analysis(
    [os.path.join(SPEC, "entry.py")],
    pathex=[ROOT],
    binaries=binaries,
    datas=datas,
    hiddenimports=hiddenimports,
    hookspath=[],
    hooksconfig={},
    runtime_hooks=[],
    excludes=["scipy", "tkinter"],
    noarchive=False,
    optimize=0,
)

pyz = PYZ(a.pure)

# onedir layout: fast startup, VTK/Warp stay unpacked.
exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="sstream",
    debug=False,
    bootloader_ignore_signals=False,
    strip=False,
    upx=False,
    upx_exclude=[],
    runtime_tmpdir=None,
    console=True,  # CLI: progress + result summary go to stdout
    disable_windowed_traceback=False,
    argv_emulation=False,
    target_arch="x86_64",
    codesign_identity=None,
    entitlements_file=None,
    icon=None,
)
coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    upx_exclude=[],
    name="sstream",
)
