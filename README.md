# S-Stream

A small desktop wind-tunnel experiment built on a 3D lattice-fluid simulation.
The bundled car is the starting model; Wavefront OBJ files can be loaded with
`--model PATH`.

## Build and run

Requires CMake, a C11 compiler, pkg-config, SDL2, SDL2_ttf, OpenGL, and an
OpenGL 4.3-capable GPU/driver.

```sh
cmake -B build -S simulation -DCMAKE_BUILD_TYPE=Release
cmake --build build -j
./build/3d_fluid_simulation_car
```

Run from the repository root so the application can find its bundled model and
shaders. Use `./build/3d_fluid_simulation_car --help` to see the available
options.

Simulation and force estimates are exploratory and should not be treated as
validated engineering results.

## License

See [LICENSE](LICENSE) and [UPSTREAM.md](UPSTREAM.md) for the retained MIT
license and source attribution.
