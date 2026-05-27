from __future__ import annotations

import argparse
import time
from cpu_lbm import CPULBM2D


def run_headless(sim, steps: int) -> None:
    """Run simulation steps without visualization, print timing at end."""
    print(f"Running {steps} steps headless...")
    start = time.time()
    sim.run(steps)
    elapsed = time.time() - start
    print(f"Done: {steps} steps in {elapsed:.3f}s ({steps / elapsed:.1f} steps/s)")


def run_visual_2d(sim, target_fps: int = 30) -> None:
    """2D event loop: handles input, runs 5 steps/frame, renders smoke via PyGame."""
    from visualizer import FluidVisualizer

    vis = FluidVisualizer(width=sim.width, height=sim.height, scale=5)

    last_frame_time = time.time()
    step_count = 0
    fps_timer = time.time()
    fps_frames = 0
    last_fps = 0.0

    print("Starting 2D simulation...")
    print("Controls: O=obstacle mode, E=emitter mode, R=reset, C=clear emitters")

    try:
        while vis.running:
            if not vis.handle_events(sim):
                break

            if not vis.paused:
                steps_per_frame = 5
                sim.run(steps_per_frame)
                step_count += steps_per_frame

            current_time = time.time()
            fps_frames += 1
            if current_time - fps_timer >= 0.5:
                fps = fps_frames / (current_time - fps_timer)
                last_fps = fps
                fps_timer = current_time
                fps_frames = 0
            else:
                fps = last_fps

            vis.update(
                sim.get_smoke(), sim.obstacles, sim.emitters, fps, step_count
            )

            frame_time = time.time() - last_frame_time
            if frame_time < 1.0 / target_fps:
                time.sleep(1.0 / target_fps - frame_time)
            last_frame_time = time.time()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        vis.close()
        print(f"Simulation ended. Total steps: {step_count}")


def run_visual_3d(sim, target_fps: int = 24) -> None:
    """3D event loop: runs 3 steps/frame, renders smoke via OpenGL volume rendering."""
    from visualizer3d import FluidVisualizer3D

    vis = FluidVisualizer3D(
        width=sim.width, height=sim.height, depth=sim.depth, scale=4
    )

    step_count = 0
    fps_timer = time.time()
    fps_frames = 0
    last_fps = 0.0
    steps_per_frame = 3
    last_frame_time = time.time()

    try:
        while vis.running:
            if not vis.handle_events(sim):
                break

            if not vis.paused:
                sim.run(steps_per_frame)
                step_count += steps_per_frame

            current_time = time.time()
            fps_frames += 1
            if current_time - fps_timer >= 0.5:
                fps = fps_frames / (current_time - fps_timer)
                last_fps = fps
                fps_timer = current_time
                fps_frames = 0
            else:
                fps = last_fps

            vis.update(sim.get_smoke(), fps, step_count)

            frame_time = time.time() - last_frame_time
            if frame_time < 1.0 / target_fps:
                time.sleep(1.0 / target_fps - frame_time)
            last_frame_time = time.time()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        vis.close()
        print(f"Simulation ended. Total steps: {step_count}")


def main() -> None:
    """Parse CLI args, construct the appropriate engine (2D/3D/CPU/GPU), and run."""
    parser = argparse.ArgumentParser(
        description="CuFloda - Fluid Dynamics Simulation"
    )
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Run without visualization (for benchmarking)",
    )
    parser.add_argument("--width", type=int, default=128, help="Grid width")
    parser.add_argument("--height", type=int, default=128, help="Grid height")
    parser.add_argument(
        "--steps", type=int, default=1000, help="Number of steps (headless only)"
    )
    parser.add_argument(
        "--viscosity", type=float, default=0.02, help="Fluid viscosity"
    )
    parser.add_argument(
        "--3d",
        action="store_true",
        dest="mode3d",
        help="Run in 3D mode (D3Q19 lattice)",
    )
    parser.add_argument(
        "--depth", type=int, default=None, help="Grid depth (3D only)"
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Use GPU acceleration (CuPy)",
    )
    args = parser.parse_args()

    if args.mode3d:
        if args.depth is None:
            depth = 128 if args.gpu else 64
        else:
            depth = args.depth

        if args.gpu:
            try:
                from gpu_lbm3d import GPULBM3D as Sim3D

                print("Using GPU backend (CuPy)")
            except ImportError as e:
                print(f"GPU backend unavailable: {e}")
                print("Falling back to CPU backend")
                from cpu_lbm3d import CPULBM3D as Sim3D
        else:
            from cpu_lbm3d import CPULBM3D as Sim3D

        sim = Sim3D(
            width=args.width,
            height=args.height,
            depth=depth,
            viscosity=args.viscosity,
        )
        sim.initialize(rho=1.0, u=0.15, v=0.0, w=0.0)

        if args.headless:
            run_headless(sim, args.steps)
        else:
            run_visual_3d(sim)
    else:
        sim = CPULBM2D(
            width=args.width,
            height=args.height,
            viscosity=args.viscosity,
        )
        sim.initialize(rho=1.0, u=0.15, v=0.0)

        if args.headless:
            run_headless(sim, args.steps)
        else:
            run_visual_2d(sim)


if __name__ == "__main__":
    main()
