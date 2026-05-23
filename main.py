import argparse
import time
from cpu_lbm import CPULBM2D


def run_headless(sim, steps):
    print(f"Running {steps} steps headless...")
    start = time.time()
    sim.run(steps)
    elapsed = time.time() - start
    print(f"Done: {steps} steps in {elapsed:.3f}s ({steps/elapsed:.1f} steps/s)")


def run_visual(sim, target_fps=30):
    from visualizer import FluidVisualizer

    vis = FluidVisualizer(width=sim.width, height=sim.height, scale=5)

    last_frame_time = time.time()
    step_count = 0
    fps_timer = time.time()
    fps_frames = 0
    last_fps = 0.0

    print("Starting simulation...")
    print("Controls: Space=Pause, R=Reset, ESC=Quit, Mouse=Draw obstacles")

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

            density = sim.get_density()
            velocity = sim.get_velocity()
            obstacles = sim.obstacles

            vis.update(density, velocity, obstacles, fps, step_count)

            frame_time = time.time() - last_frame_time
            if frame_time < 1.0 / target_fps:
                time.sleep(1.0 / target_fps - frame_time)
            last_frame_time = time.time()

    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    finally:
        vis.close()
        print(f"Simulation ended. Total steps: {step_count}")


def main():
    parser = argparse.ArgumentParser(description="CuFloda - Fluid Dynamics Simulation")
    parser.add_argument("--headless", action="store_true",
                        help="Run without visualization (for benchmarking)")
    parser.add_argument("--width", type=int, default=128, help="Grid width")
    parser.add_argument("--height", type=int, default=128, help="Grid height")
    parser.add_argument("--steps", type=int, default=1000,
                        help="Number of steps (headless only)")
    parser.add_argument("--viscosity", type=float, default=0.02,
                        help="Fluid viscosity")
    args = parser.parse_args()

    sim = CPULBM2D(width=args.width, height=args.height, viscosity=args.viscosity)
    sim.initialize(rho=1.0, u=0.15, v=0.0)

    if args.headless:
        run_headless(sim, args.steps)
    else:
        run_visual(sim)


if __name__ == "__main__":
    main()
