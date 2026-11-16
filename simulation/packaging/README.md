# Packaging the S-Stream solver for Windows

S-Stream is a Python application; "compiling for Windows" means freezing it
into a native bundle with PyInstaller. Everything needed is in this folder:

| File | Purpose |
|---|---|
| `entry.py` | Frozen-bundle launcher; on startup failure writes `%TEMP%\sstream_error.log` + a message box |
| `sstream.spec` | PyInstaller spec (onedir, console). Bundles `assets/`, Warp, VTK/vtkmodules, PyVista |
| `build_windows.ps1` | One-shot Windows build: venv -> install -> PyInstaller -> `dist\sstream\sstream.exe` |

The bundle is a **console** app: it prints solve progress and the final
report. No Qt, no OpenGL window — the solver only needs the GPU (CUDA via
Warp; kernels JIT on first launch).

## Requirements

**Build machine**

- Windows 10/11 x64
- Python 3.11+ from python.org (with the `py` launcher)

**Target machine (the one that runs the .exe)**

- NVIDIA GPU with a current driver (tested: RTX 20xx+, sm_86)
- ~2 GB free disk (bundle is ~1 GB), ~4 GB VRAM for the Preview preset

## Build it yourself

```powershell
cd <repo>
powershell -ExecutionPolicy Bypass -File .\simulation\packaging\build_windows.ps1
```

Output: `simulation\dist\sstream\sstream.exe`. To distribute, zip the whole
`dist\sstream\` folder — `_internal\` must stay next to the exe.

## Build it with GitHub Actions

`.github/workflows/windows-build.yml` builds the same bundle on every push
to `main` (and on manual dispatch / version tags) and uploads
`sstream-windows-x64.zip` as an artifact. GitHub runners have no GPU, so
CI verifies packaging only — runtime is exercised on a real machine.

## Notes & caveats

- **onedir, not onefile**: PyInstaller unpacks onefile bundles (~1 GB) into
  `%TEMP%` on every launch. Onedir starts fast and keeps VTK stable.
- **First launch is slow**: Warp JITs its CUDA kernels into
  `%LOCALAPPDATA%\warp`; subsequent runs are fast.
- **Startup errors** land in `%TEMP%\sstream_error.log` and a message box
  (see `entry.py`). Common causes: missing NVIDIA driver.
- **Default model** is bundled (`assets/3d-files/ahmed_1.stl`, resolved
  via `sys._MEIPASS` when frozen); pass a `.obj`/`.stl` path to use your own.

## Invocation

```powershell
.\sstream.exe path\to\model.stl --preset preview --out results
.\sstream.exe --help
```

(sstream.exe --help works; see `sstream/cli.py`.)
