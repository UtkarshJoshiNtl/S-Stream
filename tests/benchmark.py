"""S-Stream Performance Benchmark Suite.

Measures MLUPs/s (Million Lattice Updates Per Second) across different
engines and grid sizes. Reference: FluidX3D (MIT) benchmark methodology.

Legal Reference: FluidX3D (MIT) for MLUPs/s metric definition.
"""

import time
from dataclasses import dataclass

from engines.lbm2d import LBM2D


@dataclass
class BenchmarkResult:
    """Benchmark result for a single configuration."""

    engine_name: str
    grid_size: tuple[int, int]
    steps: int
    time_seconds: float
    mlups: float  # Million Lattice Updates Per Second
    fps: float


def benchmark_engine(
    engine_class,
    width: int,
    height: int,
    steps: int = 100,
    **engine_kwargs,
) -> BenchmarkResult:
    """Benchmark a single engine configuration."""
    try:
        sim = engine_class(width, height, **engine_kwargs)
    except Exception:
        # Engine not available (e.g., GPU not present)
        return BenchmarkResult(
            engine_name=engine_class.__name__,
            grid_size=(width, height),
            steps=steps,
            time_seconds=float("inf"),
            mlups=0.0,
            fps=0.0,
        )

    sim.initialize(1.0, 0.05, 0.0)

    # Warmup
    try:
        sim.step(physics_only=True)
    except TypeError:
        sim.step()
    sim.initialize(1.0, 0.05, 0.0)

    # Benchmark — physics-only (skip smoke/particles) for honest MLUPs
    start = time.perf_counter()
    try:
        sim.run(steps, physics_only=True)
    except TypeError:
        sim.run(steps)
    elapsed = time.perf_counter() - start

    mlups = (width * height * steps) / (elapsed * 1e6)
    fps = steps / elapsed

    return BenchmarkResult(
        engine_name=engine_class.__name__,
        grid_size=(width, height),
        steps=steps,
        time_seconds=elapsed,
        mlups=mlups,
        fps=fps,
    )


def run_benchmark_suite() -> list[BenchmarkResult]:
    """Run full benchmark suite across engines and grid sizes."""
    results = []

    sizes = [(64, 64), (128, 128), (256, 256), (512, 512)]
    steps = 100

    # CPU engines
    cpu_engines = [LBM2D]

    for engine_class in cpu_engines:
        for width, height in sizes:
            result = benchmark_engine(
                engine_class, width, height, steps, viscosity=0.02
            )
            results.append(result)

    # GPU engines (optional)
    try:
        from engines.lbm2d_gpu import LBM2DGPU

        gpu_engines = [LBM2DGPU]
        for engine_class in gpu_engines:
            for width, height in sizes:
                result = benchmark_engine(
                    engine_class, width, height, steps, viscosity=0.02
                )
                results.append(result)
    except ImportError:
        pass

    # Lettuce engine (optional)
    try:
        from engines.lbm2d_lettuce import LBM2DLettuce

        for width, height in sizes:
            result = benchmark_engine(
                LBM2DLettuce, width, height, steps, viscosity=0.02
            )
            results.append(result)
    except ImportError:
        pass

    return results


def print_results(results: list[BenchmarkResult]) -> None:
    """Print benchmark results in a formatted table."""
    print("=" * 80)
    print("S-Stream Performance Benchmark Results")
    print("Measurements are physics-only (smoke/particles skipped).")
    print("=" * 80)
    header = (
        f"{'Engine':<20} {'Grid':<12} {'Steps':<8}"
        f" {'Time (s)':<12} {'MLUPs/s':<12} {'FPS':<8}"
    )
    print(header)
    print("-" * 80)

    for result in results:
        grid_str = f"{result.grid_size[0]}x{result.grid_size[1]}"
        print(
            f"{result.engine_name:<20} {grid_str:<12} {result.steps:<8} "
            f"{result.time_seconds:<12.3f} {result.mlups:<12.1f} {result.fps:<8.1f}"
        )

    print("=" * 80)

    # Print MLUPs/s targets
    print("\nPerformance Targets:")
    print("  CPU (128x128):  >50 MLUPs/s")
    print("  GPU (128x128):  >500 MLUPs/s")
    print("  Lettuce (128x128): >1000 MLUPs/s")


def main() -> None:
    """Run benchmark suite."""
    results = run_benchmark_suite()
    print_results(results)


if __name__ == "__main__":
    main()
