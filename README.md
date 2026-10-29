# S-Stream — Live 3D Wind-Tunnel Experiments

S-Stream is pivoting to a small, local desktop wind-tunnel experience: load a 3D object, run airflow through the scene, and watch the flow evolve in real time.

## Current foundation

The project currently uses a C11/OpenGL compute-shader simulation with an SDL desktop window. It loads Wavefront OBJ meshes, offers an orbiting camera and several flow visualizations, and reports provisional drag/lift estimates during the run.

This is an experimental visualization and comparison tool. The included benchmark notes document current grid and drag-validation limits; treat force values as exploratory until those limits are addressed.

## Build

Linux dependencies: CMake, a C11 compiler, pkg-config, SDL2, SDL2_ttf, OpenGL, and an OpenGL 4.3-capable GPU/driver. See [docs/BUILDING.md](docs/BUILDING.md) for platform notes.

```sh
cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
```

Run the app from the repository root so it can find its bundled models and shaders:

```sh
./build/3d_fluid_simulation_car
```

The simulation CLI also supports headless runs and VTK output. Run `./build/3d_fluid_simulation_car --help` for options.

## What this project keeps small

The desktop C simulation and its tests are the active product. The upstream browser interface and ML training pipeline are not included in this working tree.

## License and attribution

S-Stream retains the upstream MIT license and notices. See [UPSTREAM.md](UPSTREAM.md) for the exact source revision and scope of this import.
