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
**Duration**: 2-3 weeks | **Priority**: Prerequisite for everything

The current codebase has duck-typing leaks, fused kernels that prevent extensibility, and copy-pasted code across engines. This phase fixes the foundation so all subsequent phases plug in cleanly.

### 0.1 Fix ABC Contract Violations

The `SimEngine` ABC (`engines/base.py`) is the single most imported class in the codebase (15+ modules), but analysis code bypasses it:

| Leak | Location | Fix |
|------|----------|-----|
| `sim.obstacles` direct access | `analysis/physics.py:31`, `scene/scene.py:27,41,53` | Add `get_obstacles_mut() -> np.ndarray` to ABC |
| `sim.f` direct access | `analysis/physics.py:44` | Add `get_f() -> np.ndarray` to ABC |
| `sim.u`, `sim.v` direct access | `workbench/viewport.py:270`, `scene/probe.py` | Add `get_pressure() -> np.ndarray` to ABC |
| `rho - 1.0` pressure derivation | `probe.py:40`, `scorecard.py:44`, `viewport.py:293` | Centralize in `get_pressure()` |

**Verification**: `grep -r "type: ignore" analysis/ workbench/ scene/` returns zero hits.

### 0.2 Extract Smoke Mixin

Smoke advection, diffusion, and decay code is copy-pasted across three engines:
- `engines/lbm2d.py:320-361`
- `engines/lbm2d_liquid.py:303-349`
- `engines/lbm2d_gpu.py:239-285`

Create `engines/smoke_mixin.py` with a `SmokeMixin` class containing:
- `advect_smoke()` — bilinear interpolation
- `diffuse_smoke()` — Laplacian diffusion
- `decay_smoke()` — exponential decay
- `apply_emitters()` — point source injection

All three engines inherit from this mixin. Future engines get smoke for free.

**Verification**: All existing smoke tests pass unchanged.

### 0.3 Un-fuse the Step Kernel

Currently `_fused_step_nb` in `lbm2d.py:10-87` combines streaming, bounce-back, inflow, walls, and collision into one monolithic Numba kernel. This makes it impossible to:
- Swap collision operators (MRT, TRT)
- Add new boundary conditions
- Insert turbulence model computation between steps

Refactor into separate phases:

```python
class LBM2D(SimEngine):
    def step(self):
        self.streaming()        # _stream_nb
        self.apply_boundary_conditions()  # new BC dispatcher
        self.collision()        # pluggable collision operator
        self.apply_emitters()
        self.advect_smoke()
        self.diffuse_smoke()
        self.smoke[self.obstacles] = 0.0
        self.decay_smoke()
```

Each phase is its own Numba kernel. Performance impact: ~5-15% from lost fusion. Mitigated by Phase 1 (performance optimization).

**Verification**: `pytest tests/test_lbm.py` — all boundary and collision tests pass. Benchmark script shows <15% regression.

### 0.4 Add Collision Operator Abstraction

Create `engines/collision.py`:

```python
class CollisionOperator(ABC):
    @abstractmethod
    def collide(self, f, rho, u, v, lattice, viscosity) -> np.ndarray: ...

class BGKCollision(CollisionOperator):
    """Single-relaxation-time (existing behavior)."""

class TRTCollision(CollisionOperator):
    """Two-relaxation-time — better stability, minimal cost."""

class MRTCollision(CollisionOperator):
    """Multi-relaxation-time — most accurate, moderate cost."""
```

Remove `omega` from `SimEngine` ABC. Engines accept a `CollisionOperator` in their constructor. UI exposes collision model selection in expert mode.

**Verification**: Default engine with `BGKCollision` produces identical results to current code (bit-exact for same inputs).

### 0.5 Add Boundary Condition Abstraction

Create `engines/boundary_conditions.py`:

```python
class BoundaryCondition(Protocol):
    def apply(self, f, obstacles, lattice, **kwargs) -> None: ...

class BounceBack(BC): ...         # Existing wall/obstacle BC
class EquilibriumInflow(BC): ...  # Existing left-column inflow
class ZouHeBC(BC): ...            # Proper Zou-He velocity/pressure BC
class OpenOutflow(BC): ...        # Zero-gradient outflow
class MovingWall(BC): ...         # Lid-driven or moving boundary
class SymmetryBC(BC): ...         # Symmetry plane
```

Scene gains `bc_type` field on obstacle specs. Serializer registers new BC types.

**Verification**: Existing presets produce identical flow fields. New BC types work in unit tests.

### 0.6 Clean-Room Algorithm Catalog

For each GPL project we cannot incorporate, document the algorithms we will reimplement independently:

| Algorithm | Source Project | License | Our Implementation |
|-----------|---------------|---------|-------------------|
| MRT collision (D'Humières) | PyLBM, lbmpy | GPL | Phase 2 |
| Smagorinsky SGS model | OpenLB, waLBerla | GPL | Phase 3 |
| STL voxelization | OpenLB | GPL | Phase 4 |
| Immersed boundary method | LUMA | Apache-2.0 | Phase 4 (can reference directly) |
| Block-structured AMR | MARBLES | Apache-2.0 | Phase 4 (can reference directly) |
| D3Q19/D3Q27 lattice | All | Various | Phase 2 (trivial constants) |
| Zou-He boundary conditions | OpenFOAM, Palabos | GPL | Phase 0.5 (well-known algorithm) |

**Rule**: Read GPL source to understand the algorithm. Close the source. Implement from mathematical description and published papers. Document paper references for every implementation.

---

## Phase 1: Performance
**Duration**: 2-3 weeks | **Priority**: Must be fast enough for real-time

### 1.1 Numba Kernel Optimization

After un-fusing in Phase 0.3, optimize the separated kernels:
- **Memory layout**: Ensure `f` array is memory-layout friendly for cache lines (currently `(9, H, W)` — consider `(H, W, 9)` for AoS)
- **Loop ordering**: Tune prange/nested loop order for L1/L2 cache hits
- **Prefetching**: Add manual prefetch hints where Numba supports it
- **Target**: Recover the 5-15% lost from un-fusing, plus additional gains

### 1.2 GPU Kernel Improvement

Current `lbm2d_gpu.py` uses `cp.RawKernel` with hand-written CUDA C. Improvements:
- **Occupancy tuning**: Profile and adjust block size (currently 16x16)
- **Shared memory**: Use shared memory for streaming stencil (9-direction access pattern)
- **Stream overlap**: Overlap compute and host-device transfer for smoke/velocity readback
- **Target**: 2x speedup on existing GPU kernels

### 1.3 Lettuce Backend (PyTorch GPU)

Add `engines/lbm2d_lettuce.py` as an alternative GPU backend:
- Uses Lettuce's PyTorch-based LBM (MIT license)
- Automatic CUDA kernel optimization via PyTorch compiler
- Enables 3D GPU simulation immediately (Lettuce has D3Q19/D3Q27)
- **Tradeoff**: Adds PyTorch dependency (~2GB). Make it optional via `pip install sstream[gpu]`

### 1.4 Benchmark Suite

Expand `tests/benchmark.py` into a proper benchmark suite:
- **MLUPs/s** (Million Lattice Updates Per Second) as primary metric
- Grid sizes: 64x64, 128x128, 256x256, 512x512
- CPU (single-thread, multi-thread via Numba) and GPU
- Comparison table in documentation: S-Stream vs Lettuce vs FluidX3D (published numbers)
- **Target**: >50 MLUPs/s on CPU (128x128), >500 MLUPs/s on GPU

---

## Phase 2: Accuracy — Collision Operators & 3D
**Duration**: 3-4 weeks | **Priority**: Correct physics

### 2.1 Lattice3D Constants

Add `Lattice3D` to `engines/lbm_common.py`:
- **D3Q19**: 19 velocities, industry standard for 3D LBM
- **D3Q27**: 27 velocities, higher accuracy for complex flows
- Weights, velocity vectors, opposite-direction indices
- `equilibrium()` method for 3D distributions
- `omega_from_viscosity()` for 3D

Reference: PyLBM's symbolic lattice definitions (BSD-3, can reference directly).

### 2.2 TRT Collision Operator

Two-relaxation-time is the best bang-for-buck upgrade:
- Two relaxation rates: symmetric (`s_+`) and antisymmetric (`s_-`) modes
- `s_+` controls viscosity, `s_-` controls boundary stability
- Nearly zero additional cost over BGK
- Significantly better stability at high Re and near boundaries

Implementation based on Geller et al. (2013), "A simple and accurate scheme for the lattice Boltzmann method."

**Verification**: Lid-driven cavity Re=1000 benchmark against Ghia et al. (1982). Velocity profile error < 2%.

### 2.3 MRT Collision Operator

Multi-relaxation-time decouples all moment relaxation rates:
- Transform `f` to moment space: `m = M * f`
- Relax each moment independently: `m_new[i] = m_eq[i] + s_i * (m[i] - m_eq[i])`
- Transform back: `f_new = M_inv * m_new`

Requires transformation matrices `M` and `M_inv` for D2Q9 and D3Q19. Reference: d'Humières et al. (2002), "Multiple-relaxation-time Lattice Boltzmann models for 3D simulations."

**Verification**: Compare MRT vs BGK for backward-facing step at Re=1000. MRT should show less oscillation near reattachment.

### 2.4 2D Engine with Pluggable Collision

Refactor `LBM2D` to accept any `CollisionOperator`:
```python
sim = LBM2D(width=256, height=256, collision=TRTCollision())
```

Default remains BGK for backward compatibility.

### 2.5 3D Engine (CPU)

Create `engines/lbm3d.py` — `LBM3D(SimEngine)`:
- D3Q19 lattice, BGK collision initially
- Numba-accelerated kernels (3D parallel loops)
- Same boundary condition abstraction as 2D
- Same smoke mixin (3D bilinear advection)
- **Memory**: `f` array is `(19, D, H, W)` float32 — a 128^3 grid uses ~1GB for distributions alone

**Verification**: Poiseuille flow in 3D channel. Analytical solution match within 0.1%.

### 2.6 Validation Benchmark Suite

Create `tests/validation/` with automated comparisons against published data:

| Benchmark | Reference | What it proves |
|-----------|-----------|---------------|
| Lid-driven cavity Re=100, 400, 1000 | Ghia et al. (1982) | Basic accuracy |
| Cylinder wake Re=100-1000 | Sen et al. (2018) | Drag coefficient, Strouhal |
| Backward-facing step Re=100-800 | Armaly et al. (1983) | Separation/reattachment |
| Poiseuille flow (2D + 3D) | Analytical | Exact solution recovery |
| Couette flow | Analytical | Shear accuracy |
| Taylor-Green vortex decay | Analytical | Viscous dissipation |

Each benchmark produces a pass/fail report with error metrics. Run as part of CI.

---

## Phase 3: Turbulence & Thermal
**Duration**: 3-4 weeks | **Priority**: Real-world physics

### 3.1 Smagorinsky SGS Model

The single most impactful physics addition. Enables high-Re flows without prohibitive grid resolution.

Algorithm (clean-room from OpenLB/waLBerla papers):
1. Compute strain rate tensor: `S_ij = (1/2 * tau) * sum(c_i_alpha * c_i_beta * (f_i - f_eq_i))`
2. Compute magnitude: `|S| = sqrt(2 * S_ij * S_ij)`
3. Turbulent viscosity: `nu_t = (C_s * delta)^2 * |S|` (C_s ≈ 0.1-0.2)
4. Effective relaxation: `omega_eff = 1 / (3 * (nu + nu_t) + 0.5)`

Add `SmagorinskyCollision(CollisionOperator)` and `WaleCollision(CollisionOperator)` (WALE is better near walls).

Scene gains turbulence model selector (None / Smagorinsky / WALE) in expert mode.

**Verification**: Decaying isotropic turbulence energy spectrum matches Kolmogorov -5/3 scaling.

### 3.2 Thermal LBM (Buoyancy)

Add temperature as a second distribution function:
- `f_T` (temperature populations) streamed and collided alongside `f` (velocity populations)
- Boussinesq approximation: `F_buoyancy = -beta * (T - T_0) * g_hat` added as force term
- Creates natural convection, Rayleigh-Benard convection, mixed convection scenarios

Implementation reference: Lettuce thermal LBM (MIT, can reference directly).

**Verification**: Rayleigh-Benard convection onset at critical Rayleigh number (Ra_c ≈ 1708).

### 3.3 Non-Newtonian Models

Add shear-rate dependent viscosity:
- **Power-law**: `nu = nu_0 * gamma_dot^(n-1)` (shear-thinning n<1, shear-thickening n>1)
- **Carreau model**: More realistic polymer behavior
- Viscosity computed per-cell from strain rate tensor (reuse Smagorinsky infrastructure)

This opens biomedical (blood flow) and industrial (polymer processing) markets.

**Verification**: Power-law Poiseuille flow analytical solution.

### 3.4 OpenLB/Palabos Algorithm Reference Library

For algorithms we cannot incorporate due to GPL, create `docs/algorithms/` with:
- Mathematical description from published papers (not from GPL source code)
- Our implementation pseudocode
- Verification test cases
- Paper citation and DOI

This documents our clean-room reimplementation trail and protects against license concerns.

---

## Phase 4: Geometry & Mesh
**Duration**: 4-5 weeks | **Priority**: Real-world usability

### 4.1 STL Import

Create `scene/stl_obstacle.py`:
- Load triangle mesh via `trimesh` (MIT license)
- Rasterize onto Cartesian grid using ray casting (Amanatides & Woo algorithm)
- Anti-aliased boundary detection for accurate bounce-back placement
- Support binary and ASCII STL

Add `STLObstacle(ObstacleSpec)` to scene system. Register in serializer.

**Verification**: STL sphere rasterized onto grid. Compare drag coefficient against analytical sphere drag.

### 4.2 Immersed Boundary Method

Reference LUMA (Apache-2.0, can incorporate directly or reference):
- Lagrangian markers on the immersed surface
- Force spreading: `F_body = sum(f_k * delta_h(x - x_k))`
- Velocity interpolation: `u_boundary = sum(u(x) * delta_h(x - x_k))`
- Enables moving boundaries without mesh regeneration

Create `engines/ibm.py` as a mixin or wrapper.

**Verification**: Oscillating cylinder in free stream. Compare force coefficients against Blevins (1984).

### 4.3 Adaptive Mesh Refinement (AMR)

Reference MARBLES (Apache-2.0) for block-structured AMR:
- Quadtree (2D) / Octree (3D) refinement
- Refine regions with high velocity gradient or vorticity
- Coarsen regions with low flow activity
- Interpolation between refinement levels for streaming

**Tradeoff**: AMR adds significant complexity. Implement as a separate engine (`engines/lbm2d_amr.py`) rather than modifying existing engines. Users opt in explicitly.

**Verification**: Cylinder wake with AMR — fine mesh near cylinder, coarse far field. Compare drag against uniform fine-mesh result within 5%.

### 4.4 Image-to-Obstacle

Load PNG/BMP images as obstacle masks:
- Grayscale threshold → boolean obstacle mask
- Useful for complex 2D geometries (airfoils, channel networks)
- Simple, high-value educational feature

### 4.5 Geometry Primitives

Extend beyond circle/rect/polygon:
- **Ellipse**: Parametric ellipse with rotation angle
- **Airfoil**: NACA 4-digit series analytical profile
- **Channel**: Pre-built inlet/nozzle/diffuser geometry
- **Lattice/BG**: Porous media periodic unit cell

---

## Phase 5: Intuitive Design & Visualization
**Duration**: 3-4 weeks | **Priority**: User experience

### 5.1 Engine-Agnostic Colormap System

Current colormap modes (`viewport.py:_upload_smoke()`) are hardcoded. Refactor:
- Each engine declares available fields via `get_field_names() -> list[str]`
- Colormap registry maps field names to color LUTs
- New physics (temperature, turbulence viscosity) automatically appear in viewport
- Custom colormap import (ParaView .csv format)

### 5.2 3D Viewport

When 3D engine is selected, switch viewport to:
- Volume rendering via ray marching (GLSL shader)
- Orbit/pan/zoom camera controls (trackball)
- Slice planes for field inspection
- Isosurface extraction for threshold visualization

Reference: Lettuce's visualization (MIT) for rendering approaches.

### 5.3 Guided Setup Wizard

New users should never see a blank screen. Add:
- **Template selector**: "I want to study..." → [vortex shedding, channel flow, natural convection, ...]
- Auto-populates scene with correct geometry, parameters, and probes
- Explains what each parameter does in plain language
- Transitions seamlessly to full UI when ready

### 5.4 Parameter Presets with Explanation

Every parameter slider should have:
- Tooltip explaining the physical meaning
- Recommended range for current scenario
- Warning when value moves outside stable regime
- "Why?" link to documentation

### 5.5 Real-Time Field Annotations

Overlay on viewport:
- Velocity vectors (quiver) — already exists
- Streamlines — already exists
- **Pressure contours** (iso-lines) — new
- **Vorticity contours** — new
- **Temperature iso-surfaces** (3D) — new
- **Force arrows** on obstacles showing drag/lift direction and magnitude

### 5.6 Responsive Layout

- Panels resize gracefully
- Viewport maintains aspect ratio
- Mobile-friendly layout for tablet use (educational market)
- Keyboard shortcuts for all common actions (already partially exists)

---

## Phase 6: Feature Expansion
**Duration**: 4-5 weeks | **Priority**: Market reach

### 6.1 Multi-Phase Upgrades

Current Shan-Chen model is basic. Upgrade:
- **Color gradient model** — sharper interfaces, less spurious currents
- **Free surface flow** — reference Palabos algorithm (clean-room)
- **Phase change** — liquid-gas transition with latent heat (couples with thermal)
- **Multi-component** — two immiscible fluids (oil-water)

### 6.2 Fluid-Structure Interaction

Combine IBM (Phase 4.2) with structural solver:
- Rigid body motion in flow (falling sphere, fluttering flag)
- Deformable bodies via spring-mass model
- Two-way coupling: fluid forces structure, structure displaces fluid

### 6.3 Particle Tracking

Add Lagrangian particle advection:
- Passive tracer particles (follow flow)
- Heavy particles with inertia (sedimentation)
- Particle-fluid coupling (two-way)
- Useful for visualizing mixing, dispersion, transport

### 6.4 Automated Design Optimization

Leverage automatic differentiation (from Lettuce/PyTorch backend):
- Define objective: minimize drag, maximize mixing, etc.
- Parameterize geometry or flow conditions
- Gradient-based optimization using adjoint LBM
- "Optimize this shape" button in the UI

### 6.5 Jupyter Integration

```python
import sstream
sim = sstream.LBM2D(width=256, height=256)
sim.add_obstacle(128, 128, radius=20)
sim.run(1000)
sim.plot_velocity()  # inline matplotlib figure
```

Lowers barrier for researchers and students.

### 6.6 REST API Mode

```bash
python main.py --serve --port 8080
```

- POST /scene — load scene JSON
- POST /run — run N steps
- GET /field?type=velocity — get field as NumPy array
- GET /probe?id=0 — get probe data
- Enables web frontend, CI/CD integration, cloud deployment

---

## Phase 7: Distribution & Ecosystem
**Duration**: 2-3 weeks | **Priority**: Adoption

### 7.1 Packaging

- `pyproject.toml` with optional dependency groups:
  - `pip install sstream` — CPU only
  - `pip install sstream[gpu]` — + CuPy
  - `pip install sstream[lettuce]` — + PyTorch GPU backend
  - `pip install sstream[full]` — everything
- Wheel distribution on PyPI
- Conda package for scientific Python users

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
- Annual "S-Stream Challenge" — best simulation visualization

### 7.4 Web Version

- Pyodide/WebAssembly for browser-based simulation
- No install required — instant access for education
- Subset of features (2D, BGK, basic visualization)
- Upgrade path to desktop for full features

---

## Implementation Roadmap

```
Phase 0: Foundation         ████████████░░░░░░░░░░░░  Weeks 1-3
Phase 1: Performance        ░░░░░░████████░░░░░░░░░  Weeks 4-6
Phase 2: Accuracy           ░░░░░░░░░░░░████████░░░  Weeks 7-10
Phase 3: Turbulence/Thermal ░░░░░░░░░░░░░░░░████░░░  Weeks 11-14
Phase 4: Geometry/Mesh      ░░░░░░░░░░░░░░░░░░░████  Weeks 15-19
Phase 5: Design/Viz         ░░░░░░░░░░░░░░░░████░░░  Weeks 17-20
Phase 6: Features           ░░░░░░░░░░░░░░░░░░░████  Weeks 21-25
Phase 7: Distribution       ░░░░░░░░░░░░░░░░░░░░░██  Weeks 26-28
```

Phases 4 and 5 overlap (geometry and visualization are independent workstreams).

## License Compliance Checklist

- [ ] All incorporated code is MIT, BSD-3, or Apache-2.0
- [ ] No GPL/AGPL source code copied — only algorithms reimplemented from papers
- [ ] Apache-2.0 code retains original copyright notices and LICENSE text
- [ ] `NOTICE` files from Apache-2.0 projects included in distribution
- [ ] Each clean-room reimplementation documented with paper reference
- [ ] Legal review before any public release

## Success Metrics

| Metric | Current | 6 Months | 12 Months |
|--------|---------|----------|-----------|
| Grid size (interactive) | 128x128 | 256x256 | 512x512 |
| MLUPs/s (CPU) | ~10 | >50 | >100 |
| MLUPs/s (GPU) | ~100 | >500 | >1000 |
| Collision operators | 1 (BGK) | 3 (BGK/TRT/MRT) | 4 (+neural) |
| Boundary conditions | 4 | 7 | 10+ |
| Presets | 9 | 20 | 40 |
| Validation benchmarks | 0 | 6 | 12 |
| Supported dimensions | 2D | 2D + 3D | 2D + 3D |
| Turbulence models | 0 | 2 | 3 |
| Geometry import | None | STL | STL + images + primitives |
| Test count | 101 | 200+ | 400+ |
| PyPI downloads/month | 0 | 100 | 1000+ |
