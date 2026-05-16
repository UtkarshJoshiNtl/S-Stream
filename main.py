import time
from cpu_lbm import CPULBM2D
from visualizer import FluidVisualizer

def main():
    # Simulation parameters
    width = 128
    height = 128
    viscosity = 0.02
    target_fps = 30
    
    # Initialize simulation
    sim = CPULBM2D(width=width, height=height, viscosity=viscosity)
    sim.initialize(rho=1.0, u=0.1, v=0.0)
    
    # Initialize visualizer
    vis = FluidVisualizer(width=width, height=height, scale=4)
    
    # Main loop
    clock = time.time()
    frame_count = 0
    step_count = 0
    fps_timer = time.time()
    fps_frames = 0
    
    print("Starting simulation...")
    print("Controls: Space=Pause, R=Reset, ESC=Quit, Mouse=Draw obstacles")
    
    try:
        while vis.running:
            # Handle events
            if not vis.handle_events(sim):
                break
            
            # Run simulation steps if not paused
            if not vis.paused:
                # Run multiple steps per frame for faster simulation
                steps_per_frame = max(1, int(target_fps / 10))
                sim.run(steps_per_frame)
                step_count += steps_per_frame
            
            # Calculate FPS
            current_time = time.time()
            fps_frames += 1
            if current_time - fps_timer >= 1.0:
                fps = fps_frames / (current_time - fps_timer)
                fps_timer = current_time
                fps_frames = 0
            else:
                fps = 0.0
            
            # Get simulation state
            density = sim.get_density()
            obstacles = sim.obstacles
            
            # Update visualization
            vis.update(density, obstacles, fps, step_count)
            
            # Frame rate control
            frame_time = time.time() - clock
            if frame_time < 1.0 / target_fps:
                time.sleep(1.0 / target_fps - frame_time)
            clock = time.time()
            
            frame_count += 1
    
    except KeyboardInterrupt:
        print("\nSimulation interrupted by user")
    
    finally:
        vis.close()
        print(f"Simulation ended. Total steps: {step_count}")

if __name__ == "__main__":
    main()
