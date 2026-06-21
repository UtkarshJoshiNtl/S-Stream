from __future__ import annotations

import argparse
import sys

from PySide6.QtWidgets import QApplication

from engines import LBM2D, LBM2DGPU, LBM2DLiquid


def main() -> None:
    parser = argparse.ArgumentParser(description="S-Stream LBM fluid workbench")
    parser.add_argument("--gpu", action="store_true", help="Use GPU (CuPy) backend")
    parser.add_argument(
        "--liquid", action="store_true", help="Use liquid multiphase engine"
    )
    parser.add_argument("--width", type=int, default=128, help="Grid width")
    parser.add_argument("--height", type=int, default=128, help="Grid height")
    parser.add_argument("--headless", action="store_true", help="Run headless (no GUI)")
    parser.add_argument("--steps", type=int, default=0, help="Steps if headless")
    args = parser.parse_args()

    if args.liquid:
        sim = LBM2DLiquid(width=args.width, height=args.height)
    elif args.gpu:
        if LBM2DGPU is None:
            print("CuPy not available, install with: pip install cupy-cuda12x")
            sys.exit(1)
        sim = LBM2DGPU(width=args.width, height=args.height)
    else:
        sim = LBM2D(width=args.width, height=args.height)

    if args.headless:
        if args.steps > 0:
            sim.run(args.steps)
        print(f"Simulation: {sim.grid_shape[1]}x{sim.grid_shape[0]}")
        return

    app = QApplication(sys.argv)
    from workbench.app import MainWindow

    window = MainWindow(sim)
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
