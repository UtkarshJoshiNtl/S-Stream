import time
from cpu_lbm import CPULBM2D


def benchmark():
    print("=" * 60)
    print("CuFloda Performance Benchmark")
    print("=" * 60)

    sizes = [(64, 64), (128, 128), (256, 256)]
    steps = 100

    for width, height in sizes:
        print(f"\nBenchmarking {width}x{height} grid ({steps} steps):")
        print("-" * 60)

        cpu_sim = CPULBM2D(width, height, viscosity=0.02)
        cpu_sim.initialize(1.0, 0.1, 0.0)

        start = time.time()
        cpu_sim.run(steps)
        cpu_time = time.time() - start
        cpu_fps = steps / cpu_time

        print(f"CPU:  {cpu_time:.3f}s ({cpu_fps:.1f} FPS)")

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
