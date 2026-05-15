import numpy as np
import time

# CPU LBM implementation for benchmarking
class CPULBM2D:
    def __init__(self, width, height, viscosity=0.02):
        self.width = width
        self.height = height
        self.omega = 1.0 / (3.0 * viscosity + 0.5)
        
        # D2Q9 lattice constants
        self.w = np.array([4/9, 1/9, 1/9, 1/9, 1/9, 1/36, 1/36, 1/36, 1/36])
        self.cx = np.array([0, 1, 0, -1, 0, 1, -1, -1, 1])
        self.cy = np.array([0, 0, 1, 0, -1, 1, 1, -1, -1])
        self.opp = np.array([0, 3, 4, 1, 2, 7, 8, 5, 6])
        
        # Initialize distribution functions
        self.f = np.zeros((9, height, width))
        self.rho = np.ones((height, width))
        self.u = np.zeros((height, width))
        self.v = np.zeros((height, width))
        self.obstacles = np.zeros((height, width), dtype=bool)
        
    def equilibrium(self, rho, u, v):
        feq = np.zeros((9, self.height, self.width))
        u2 = u**2 + v**2
        for i in range(9):
            cu = self.cx[i] * u + self.cy[i] * v
            feq[i] = self.w[i] * rho * (1 + 3*cu + 4.5*cu**2 - 1.5*u2)
        return feq
    
    def initialize(self, rho=1.0, u=0.1, v=0.0):
        self.f = self.equilibrium(self.rho, u, v)
    
    def collision(self):
        # Compute macroscopic quantities
        self.rho = np.sum(self.f, axis=0)
        self.u = np.sum(self.f * self.cx[:, np.newaxis, np.newaxis], axis=0) / self.rho
        self.v = np.sum(self.f * self.cy[:, np.newaxis, np.newaxis], axis=0) / self.rho
        
        # BGK collision
        feq = self.equilibrium(self.rho, self.u, self.v)
        self.f = self.f * (1 - self.omega) + feq * self.omega
    
    def streaming(self):
        # Stream particles
        for i in range(9):
            self.f[i] = np.roll(self.f[i], shift=(self.cx[i], self.cy[i]), axis=(1, 0))
    
    def apply_obstacles(self):
        # Bounce-back boundary condition
        for i in range(9):
            self.f[i][self.obstacles] = self.f[self.opp[i]][self.obstacles]
    
    def step(self):
        self.streaming()
        self.apply_obstacles()
        self.collision()
    
    def run(self, steps):
        for _ in range(steps):
            self.step()


def benchmark():
    print("=" * 60)
    print("CuFloda Performance Benchmark")
    print("=" * 60)
    
    sizes = [(64, 64), (128, 128), (256, 256)]
    steps = 100
    
    for width, height in sizes:
        print(f"\nBenchmarking {width}x{height} grid ({steps} steps):")
        print("-" * 60)
        
        # CPU benchmark
        cpu_sim = CPULBM2D(width, height, viscosity=0.02)
        cpu_sim.initialize(1.0, 0.1, 0.0)
        
        start = time.time()
        cpu_sim.run(steps)
        cpu_time = time.time() - start
        cpu_fps = steps / cpu_time
        
        print(f"CPU:  {cpu_time:.3f}s ({cpu_fps:.1f} FPS)")
        
        # GPU benchmark
        try:
            import cufloda
            gpu_sim = cufloda.LBM2D(width, height, 0.02)
            gpu_sim.initialize(1.0, 0.1, 0.0)
            
            start = time.time()
            gpu_sim.run(steps)
            gpu_time = time.time() - start
            gpu_fps = steps / gpu_time
            
            speedup = cpu_time / gpu_time
            print(f"GPU:  {gpu_time:.3f}s ({gpu_fps:.1f} FPS)")
            print(f"Speedup: {speedup:.1f}x")
        except ImportError:
            print("GPU:  Not available (CuFloda module not found)")
        except Exception as e:
            print(f"GPU:  Error - {e}")
    
    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark()
