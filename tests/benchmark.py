import time
from engines.lbm2d import LBM2D


def benchmark() -> None:
    print("=" * 60)
    print("CuFloda Performance Benchmark")
    print("=" * 60)

    sizes = [(64, 64), (128, 128), (256, 256)]
    steps = 100

    for width, height in sizes:
        print(f"\nBenchmarking {width}x{height} grid ({steps} steps):")
        print("-" * 60)

        cpu_sim = LBM2D(width, height, viscosity=0.02)
        cpu_sim.initialize(1.0, 0.1, 0.0)

        start = time.time()
        cpu_sim.run(steps)
        cpu_time = time.time() - start
        cpu_fps = steps / cpu_time

        print(f"CPU:  {cpu_time:.3f}s ({cpu_fps:.1f} FPS)")

    print("\n" + "=" * 60)


if __name__ == "__main__":
    benchmark()
