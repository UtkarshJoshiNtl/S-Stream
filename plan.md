# S-Stream: Industry Takeover Plan

## Vision

Build the most accessible, accurate, and feature-rich Lattice Boltzmann fluid simulation platform in the world. Not the most powerful — the most **usable**. Engineers, students, and researchers should get a trustworthy fluid dynamics answer in under two minutes, without a PhD in computational physics.

## Design Philosophy

**Modular** — Every component (collision operator, boundary condition, lattice, turbulence model) is a swappable plugin. No monoliths.

**Intuitive** — Complexity is hidden by default, exposed on demand. A first-time user runs a simulation in 3 clicks. An expert tunes every relaxation parameter.

**Efficient** — CPU-first with optional GPU acceleration. Real-time feedback at 30+ fps is non-negotiable. No waiting for results.

## Priority Hierarchy

```
1. Accuracy        — If it's wrong, nothing else matters
2. Performance     — If it's slow, nobody uses it
3. Design & Viz    — If it's ugly/confusing, nobody understands it
4. Features        — If it doesn't exist, we build it
```

---

## Phase 0: Architectural Foundation
**Status**: ✅ Complete (commit `3237193`)

### 0.1 Fix ABC Contract Violations ✅

Added `get_obstacles_mut()`, `get_f()`, `get_pressure()` to `SimEngine` ABC. All analysis/viewport/probe code uses the abstract interface.

### 0.2 Extract Smoke Mixin ✅

`engines/smoke_mixin.py` — `SmokeMixin` class with `advect_smoke()`, `diffuse_smoke()`, `decay_smoke()`, `apply_emitters()`. Supports both 2D bilinear and 3D trilinear interpolation. Used by all 6 engines (LBM2D, LBM3D, LBM2DGPU, LBM2DLettuce, LBM2DLiquid, LBM2DMultiComponent).

### 0.3 Un-fuse the Step Kernel ✅

`LBM2D.step()` calls separate phases: streaming → obstacles → inflow → outflow → walls → collision → emitters → advect smoke → diffuse smoke → decay smoke. Each is its own Numba kernel.

### 0.4 Add Collision Operator Abstraction ✅

`engines/collision.py` — `CollisionOperator` ABC with `BGKCollision`, `TRTCollision`, `MRTCollision`, `SmagorinskyCollision`, `WaleCollision`. All accept lattice, viscosity, and omega parameters.

### 0.5 Add Boundary Condition Abstraction ✅

`engines/boundary_conditions.py` — Boundary condition module alongside inline BCs in engine files. Bounce-back, equilibrium inflow, open outflow implemented.

### 0.6 Clean-Room Algorithm Catalog ✅

All algorithm references documented. Clean-room reimplemented from published papers (Geller et al. 2013 for TRT, d'Humières et al. 2002 for MRT, Smagorinsky 1963, Lilly 1966 for SGS).

---

## Phase 1: Performance
**Status**: ✅ Complete (commit `c0e0591`)

### 1.1 Numba Kernel Optimization ✅

Separated kernels optimized with prange parallelism, fastmath, boundscheck=False. Memory layout `(9, H, W)` maintained for streaming compatibility.

### 1.2 GPU Kernel Improvement ✅

`engines/lbm2d_gpu.py` — Custom CUDA `RawKernel` with fused streaming+collision. Dual CUDA streams for compute/transfer overlap. 16x16 block size.

### 1.3 Lettuce Backend (PyTorch GPU) ✅

`engines/lbm2d_lettuce.py` — `LBM2DLettuce` using Lettuce's `LettuceD2Q9`, `LettuceBGK`, `LettuceStreaming`. Graceful fallback if PyTorch not installed.

### 1.4 Benchmark Suite ✅

`tests/benchmark.py` — MLUPs/s metrics, grid size scaling, engine comparison.

---

## Phase 2: Accuracy — Collision Operators & 3D
**Status**: ✅ Complete (commit `01440a2`)

### 2.1 Lattice3D Constants ✅

`engines/lbm_common.py` — `LATTICE_3D_Q19` with D3Q19 weights, velocity vectors, opposite indices, equilibrium, omega_from_viscosity.

### 2.2 TRT Collision Operator ✅

`TRTCollision` in `engines/collision.py`. Two relaxation rates (symmetric s+, antisymmetric s-). 2D only (raises `NotImplementedError` for 3D).

### 2.3 MRT Collision Operator ✅

`MRTCollision` in `engines/collision.py`. Full D2Q9 transformation matrices M and M_inv. 2D only.

### 2.4 2D Engine with Pluggable Collision ✅

`LBM2D` accepts `collision` parameter in constructor. Default `BGKCollision`.

### 2.5 3D Engine (CPU) ✅

`engines/lbm3d.py` — `LBM3D(SimEngine, SmokeMixin, ThermalMixin)`. D3Q19 lattice, BGK collision, bounce-back on all 6 faces, trilinear smoke advection. Integrates `ParticleTracer`.

### 2.6 Validation Benchmark Suite ✅

`tests/test_lbm.py` — 7 test classes: `TestInit`, `TestCollision`, `TestStreaming`, `TestBoundaries`, `TestEmitters`, `TestSmoke`, `TestPressure`. 133+ tests.

---

## Phase 3: Turbulence & Thermal
**Status**: ✅ Complete (commit `befb95f`)

### 3.1 Smagorinsky SGS Model ✅

`SmagorinskyCollision` in `engines/collision.py`. Strain rate tensor → turbulent viscosity → effective omega. Both 2D and 3D kernels.

### 3.2 WALE Model ✅

`WaleCollision` in `engines/collision.py`. Better near-wall behavior than Smagorinsky. Both 2D and 3D kernels.

### 3.3 Thermal LBM (Buoyancy) ✅

`engines/thermal_mixin.py` — `ThermalMixin` with Boussinesq buoyancy: `F = -beta * (T - T_ref) * g_hat`. Temperature distribution collision, 2D and 3D buoyancy kernels. Used by `LBM3D`.

### 3.4 Non-Newtonian Models ✅

`engines/non_newtonian.py` — `PowerLawModel`, `CarreauModel`, `BinghamModel` with `NonNewtonianCollision` wrapper. Variable-omega collision from local strain rate. 2D and 3D support.

### 3.5 Bonus: AI Context Builder ✅

`analysis/ai_context.py` — `build_ai_context()` generates structured text prompts from simulation state for Gemini integration.

---

## Phase 4: Geometry & Mesh
**Status**: ✅ Complete (commit `2447cae`)

> **Note**: Immersed Boundary Method (4.2) and Adaptive Mesh Refinement (4.3) are deferred to a future phase. They are independent features that can be added when needed.

### 4.1 STL Import ✅

`scene/scene.py` — `STLObstacle(ObstacleSpec)` with `trimesh` loading and 2D/3D rasterization. Optional `trimesh>=3.20.0` dependency.

### 4.2 Immersed Boundary Method — Future

Lagrangian markers on immersed surface, force spreading, velocity interpolation. Will enable moving boundaries without mesh regeneration.

### 4.3 Adaptive Mesh Refinement — Future

Quadtree/octree refinement based on velocity gradient or vorticity. Will be implemented as separate engine (`engines/lbm2d_amr.py`).

### 4.4 Image-to-Obstacle ✅

`scene/scene.py` — `ImageObstacle(ObstacleSpec)` with PIL loading, grayscale threshold, invert, scale. Optional `Pillow` dependency.

### 4.5 Geometry Primitives ✅

Seven types in `scene/scene.py`:
- `CircleObstacle` — click-drag from center
- `RectObstacle` — click-drag corners
- `PolygonObstacle` — freehand point-by-point
- `EllipseObstacle` — parametric with rotation
- `AirfoilObstacle` — full NACA 4-digit series, angle-of-attack, boundary + fill
- `ChannelObstacle` — variable inlet/outlet ratio (nozzle/diffuser)
- `LatticeObstacle` — porous media unit cell, configurable cell_size/wall_thickness

---

## Phase 5: Intuitive Design & Visualization
**Status**: ✅ Complete (commits `417f2a9`, `0df16a2`)

### 5.1 Engine-Agnostic Colormap System ✅

`resources/colormaps.py` — `FIELD_REGISTRY` with 10 fields (smoke, speed, vorticity, pressure, density, phase, temperature, component1, component2, color). 6 colormaps (viridis, plasma, inferno, coolwarm, blues, reds). Each engine declares fields via `get_field_names()`.

### 5.2 3D Viewport — Future

Volume rendering via ray marching (GLSL shader), orbit/pan/zoom camera, slice planes, isosurface extraction. Will be added when 3D interactive use cases mature.

### 5.3 Guided Setup Wizard ✅

`workbench/dialogs/wizard_dialog.py` — `WizardDialog` with 10 templates across 3 categories:
- **Study Flow Physics** (7): Vortex Shedding, Lid-Driven Cavity, Backward-Facing Step, Bluff Body Drag, Channel Flow, Nozzle & Diffuser, Porous Screen
- **Create & Experiment** (2): Blank Canvas, Two Cylinders
- **Learn Lattice Boltzmann** (1): What is LBM?

Auto-populates scene, keyboard shortcut Ctrl+W.

### 5.4 Parameter Presets with Explanation ✅

Every parameter spinbox in `workbench/panels/scene_panel.py` has tooltip with physical meaning, typical range, and stability guidance. 10 presets in `presets/scenes/`.

### 5.5 Real-Time Field Annotations ✅

`workbench/viewport.py` — Five overlay types with toggle buttons:
- Velocity vectors (quiver) with arrowheads
- Streamlines seeded from left boundary, RK2 traced
- Pressure contours via marching squares (11 levels)
- Force arrows on obstacles (momentum-exchange estimate)
- Particle trails with alpha gradient + dot rendering

### 5.6 Responsive Layout ✅

QMainWindow with dockable panels, toolbar, keyboard shortcuts (Space=pause, S=step, R=reset, Ctrl+O/L/S/W, 1-5=overlays).

### 5.7 Bonus: Outcome Panel ✅

`workbench/panels/outcome_panel.py` — Flow regime display, sanity warnings, design scorecard.

### 5.8 Bonus: Parameter Sweep ✅

`analysis/sweep.py` + `workbench/dialogs/sweep_dialog.py` — Parametric studies varying viscosity/inflow/diffusion/decay with threaded execution and pyqtgraph plotting.

### 5.9 Bonus: Export Dialog ✅

`workbench/dialogs/export_dialog.py` — Field selector, image/video/data export with resolution options.

### 5.10 Bonus: Recipes Dialog ✅

`workbench/dialogs/recipes_dialog.py` — Guided flow-story workflow recipes.

---

## Phase 6: Feature Expansion
**Status**: Partially complete

### 6.1 Multi-Phase Upgrades ✅

- **Color gradient model** ✅ — Color gradient perturbation force in `LBM2DMultiComponent` (sigma parameter)
- **Multi-component** ✅ — `engines/lbm2d_multicomponent.py`: Two-component Shan-Chen with intra (g11/g22) and inter (g12) forces. Fields: component1, component2, color. Scene panel controls for all parameters. `--multicomponent` CLI flag. Oil-water separation preset.
- **Free surface flow** — Future: Track liquid-gas interface explicitly (volume-of-fluid or level-set)
- **Phase change** — Future: Liquid-gas transition with latent heat coupling to thermal LBM

### 6.2 Fluid-Structure Interaction — Future

Combine IBM with structural solver: rigid body motion (falling sphere, fluttering flag), deformable bodies (spring-mass), two-way coupling.

### 6.3 Particle Tracking ✅

`engines/particle_tracer.py` — `ParticleTracer` with RK2 (Heun's) advection, bilinear/trilinear interpolation, trail buffer, `add_particles`/`add_particles_line`/`add_particles_random`. Integrated into all 6 engines. Viewport renders trails with alpha gradient.

### 6.4 Automated Design Optimization — Future

Adjoint LBM via Lettuce/PyTorch autograd. Define objective (minimize drag, maximize mixing), parameterize geometry, gradient-based optimization. "Optimize this shape" button.

### 6.5 Jupyter Integration ✅

`sstream/__init__.py` — Convenience package re-exporting all engines. `SimEngine` base class has `_repr_png_()`, `_repr_html_()` for inline display, plus `plot_field()`, `plot_velocity()`, `plot_pressure()`, `plot_smoke()`, `plot_vorticity()`.

### 6.6 REST API Mode ✅

`engines/api.py` — FastAPI app with 11 endpoints: `/`, `/engine`, `/run`, `/field`, `/field/array`, `/field/png`, `/velocity`, `/pressure`, `/obstacles`, `/obstacle`, `/emitters`, `/probe`. `python main.py --serve --port 8080`.

---

## Phase 7: Distribution & Ecosystem
**Status**: Not started

### 7.1 Packaging

`pyproject.toml` with optional dependency groups:
- `pip install sstream` — CPU only
- `pip install sstream[notebook]` — + matplotlib
- `pip install sstream[api]` — + FastAPI/uvicorn
- `pip install sstream[gpu]` — + CuPy
- `pip install sstream[lettuce]` — + PyTorch + Lettuce
- `pip install sstream[full]` — everything

### 7.2 Documentation

- **Quick start guide**: 5 minutes from install to first simulation
- **Physics guide**: What each model does, when to use it, expected accuracy
- **API reference**: Auto-generated from docstrings
- **Video tutorials**: 5-minute walkthroughs for common scenarios
- **Validation report**: Benchmark results vs published data

### 7.3 Community

- GitHub Discussions for Q&A
- Example gallery (user-contributed presets)
- Plugin registry for custom collision operators, BCs, visualizations

### 7.4 Web Version

Pyodide/WebAssembly for browser-based simulation. Subset of features (2D, BGK, basic visualization).

---

## Future Features (Backlog)

These are features we want to add but haven't scheduled yet:

| Feature | Complexity | Value | Dependencies |
|---------|-----------|-------|-------------|
| Free surface flow | High | Medium | None |
| Phase change (latent heat) | High | Medium | Thermal LBM (done) |
| Immersed boundary method | High | High | None |
| Adaptive mesh refinement | High | High | None |
| Fluid-structure interaction | Very high | High | IBM |
| Adjoint LBM optimization | Very high | Medium | Lettuce (done) |
| 3D viewport (volume rendering) | High | High | 3D engine (done) |
| TRT/MRT 3D support | Medium | Medium | 3D engine (done) |
| Zou-He boundary conditions | Medium | Medium | None |
| Moving wall boundary | Low | Medium | None |
| D3Q27 lattice | Low | Low | D3Q19 (done) |
| Conda package | Low | Medium | Packaging (7.1) |
| Validation benchmark suite | Medium | High | None |

---

## Implementation Roadmap

```
Phase 0: Foundation         ████████████████████████  ✅ Done
Phase 1: Performance        ████████████████████████  ✅ Done
Phase 2: Accuracy           ████████████████████████  ✅ Done
Phase 3: Turbulence/Thermal ████████████████████████  ✅ Done
Phase 4: Geometry/Mesh      ████████████████████████  ✅ Done (excl. IBM/AMR)
Phase 5: Design/Viz         ████████████████████████  ✅ Done
Phase 6: Features           ████████████░░░░░░░░░░░░  Partial (6.1-6.3, 6.5-6.6 done)
Phase 7: Distribution       ░░░░░░░░░░░░░░░░░░░░░░░░  Not started
```

## License Compliance Checklist

- [x] All incorporated code is MIT, BSD-3, or Apache-2.0
- [x] No GPL/AGPL source code copied — only algorithms reimplemented from papers
- [x] Apache-2.0 code retains original copyright notices and LICENSE text
- [x] Each clean-room reimplementation documented with paper reference
- [ ] Legal review before any public release

## Success Metrics

| Metric | Current | Target |
|--------|---------|--------|
| Grid size (interactive) | 128x128 | 256x256 |
| Collision operators | 5 (BGK, TRT, MRT, Smagorinsky, WALE) | 5+ |
| Boundary conditions | 4 (bounce-back, inflow, outflow, walls) | 7+ |
| Presets | 10 | 20+ |
| Geometry types | 7 (circle, rect, polygon, ellipse, airfoil, channel, lattice) | 7+ |
| Simulation engines | 6 (2D, 3D, GPU, Lettuce, Liquid, MultiComponent) | 6+ |
| Viewport overlays | 5 (quiver, streamlines, contours, force arrows, particles) | 5+ |
| Export formats | 4 (PNG, MP4/GIF, CSV, Markdown) | 4+ |
| Fields visualized | 10 | 10+ |
| Test count | 273+ | 300+ |
| Supported dimensions | 2D + 3D | 2D + 3D |
| Turbulence models | 2 (Smagorinsky, WALE) | 2+ |
| Non-Newtonian models | 3 (PowerLaw, Carreau, Bingham) | 3+ |
| Multi-phase models | 2 (single-component Shan-Chen, multi-component) | 3+ |
