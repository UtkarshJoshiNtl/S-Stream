from __future__ import annotations

import argparse
import inspect
import os
import sys
from pathlib import Path

# Default to xcb on Wayland to avoid Qt6 buffer-size protocol errors
# (xdg_surface buffer does not match configured maximized state).
if "WAYLAND_DISPLAY" in os.environ and "QT_QPA_PLATFORM" not in os.environ:
    os.environ["QT_QPA_PLATFORM"] = "xcb"

from PySide6.QtGui import QIcon, QSurfaceFormat
from PySide6.QtWidgets import QApplication

from engines import LBM2D, LBM2DGPU, LBM2DLiquid, LBM2DMultiComponent
from resources.theme import APP_STYLESHEET


def _run_steps(sim, steps: int, physics_only: bool) -> None:
    """Call eng.run with physics_only when the engine supports it."""
    if not physics_only:
        sim.run(steps)
        return
    try:
        params = inspect.signature(sim.run).parameters
        if "physics_only" in params:
            sim.run(steps, physics_only=True)
            return
    except (TypeError, ValueError):
        pass
    try:
        sim.run(steps, physics_only=True)
    except TypeError:
        sim.run(steps)


def main() -> None:
    parser = argparse.ArgumentParser(description="S-Stream LBM fluid workbench")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) backend")
    parser.add_argument(
        "--liquid", action="store_true", help="Use liquid multiphase engine"
    )
    parser.add_argument(
        "--multicomponent",
        action="store_true",
        help="Use two-component Shan-Chen engine (oil-water separation)",
    )
    parser.add_argument("--width", type=int, default=128, help="Grid width")
    parser.add_argument("--height", type=int, default=128, help="Grid height")
    parser.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
    parser.add_argument("--steps", type=int, default=0, help="Steps if headless")
    parser.add_argument(
        "--physics-only",
        action="store_true",
        help="Skip smoke/particles in headless runs (when the engine supports it)",
    )
    parser.add_argument(
        "--serve",
        action="store_true",
        help="Start REST API server (requires: pip install sstream[api])",
    )
    parser.add_argument("--port", type=int, default=8080, help="API server port")
    args = parser.parse_args()

    if args.multicomponent:
        sim = LBM2DMultiComponent(width=args.width, height=args.height)
    elif args.liquid:
        sim = LBM2DLiquid(width=args.width, height=args.height)
    elif args.gpu:
        if LBM2DGPU is None:
            print(
                "Warning: CuPy not available (install with: pip install"
                " cupy-cuda12x). Falling back to CPU engine."
            )
            sim = LBM2D(width=args.width, height=args.height)
        else:
            sim = LBM2DGPU(width=args.width, height=args.height)
    else:
        sim = LBM2D(width=args.width, height=args.height)

    if args.serve:
        import uvicorn

        from engines.api import create_app

        app = create_app(sim)
        print(f"S-Stream API starting on http://0.0.0.0:{args.port}")
        print(f"Engine: {type(sim).__name__} | Grid: {sim.grid_shape}")
        uvicorn.run(app, host="0.0.0.0", port=args.port, log_level="info")
        return

    if args.headless:
        if args.steps > 0:
            _run_steps(sim, args.steps, physics_only=args.physics_only)
        print(f"Simulation: {sim.grid_shape[1]}x{sim.grid_shape[0]}")
        return

    fmt = QSurfaceFormat()
    fmt.setVersion(3, 3)
    fmt.setProfile(QSurfaceFormat.OpenGLContextProfile.CoreProfile)
    fmt.setSwapBehavior(QSurfaceFormat.SwapBehavior.DoubleBuffer)
    fmt.setDepthBufferSize(0)
    fmt.setSamples(0)
    QSurfaceFormat.setDefaultFormat(fmt)

    app = QApplication(sys.argv)
    app.setStyleSheet(APP_STYLESHEET)

    icon_path = Path(__file__).parent / "resources" / "icon.svg"
    if icon_path.exists():
        app.setWindowIcon(QIcon(str(icon_path)))

    from workbench.app import MainWindow

    window = MainWindow(sim)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
