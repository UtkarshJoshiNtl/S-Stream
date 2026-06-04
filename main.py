from __future__ import annotations

import argparse
import time

from engines import LBM2D, LBM3DCPU, LBM3DGPU, SimEngine
from ui.app import CuFlodaApp


def run_headless(sim: SimEngine, steps: int) -> None:
    print(f"Running {steps} steps headless...")
    start = time.time()
    sim.run(steps)
    elapsed = time.time() - start
    print(f"Done: {steps} steps in {elapsed:.3f}s ({steps / elapsed:.1f} steps/s)")


def main() -> None:
    parser = argparse.ArgumentParser(description="CuFloda - Fluid Dynamics Simulation")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without visualization",
    )
    parser.add_argument("--width", type=int, default=128, help="Grid width")
    parser.add_argument("--height", type=int, default=128, help="Grid height")
    parser.add_argument("--steps", type=int, default=1000, help="Steps (headless only)")
    parser.add_argument("--viscosity", type=float, default=0.02, help="Fluid viscosity")
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="mode3d",
        help="Run in 3D mode (D3Q19 lattice)",
    )
    parser.add_argument("--depth", type=int, default=None, help="Grid depth (3D)")
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration (CuPy)",
    )
    args = parser.parse_args()

    if args.mode3d:
        depth = args.depth or (128 if args.gpu else 64)

        if args.gpu and LBM3DGPU is not None:
            sim: SimEngine = LBM3DGPU(
                width=args.width,
                height=args.height,
                depth=depth,
                viscosity=args.viscosity,
            )
        else:
            sim = LBM3DCPU(
                width=args.width,
                height=args.height,
                depth=depth,
                viscosity=args.viscosity,
            )

        sim.initialize(rho=1.0, u=0.15, v=0.0, w=0.0)
    else:
        sim = LBM2D(
            width=args.width,
            height=args.height,
            viscosity=args.viscosity,
        )
        sim.initialize(rho=1.0, u=0.15, v=0.0)

    if args.headless:
        run_headless(sim, args.steps)
    else:
        app = CuFlodaApp(sim)
        app.run()


if __name__ == "__main__":
    main()
