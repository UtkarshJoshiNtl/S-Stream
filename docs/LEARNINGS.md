# LEARNINGS

Draft journal. Every entry below is either a fact measurable in this repo
(commit hash, file, test, command output) or is marked **to verify**.
Nothing here is a claim about physics the solver has not measured.

## What the original code got wrong

Audit facts, oldest first (see `git log` for the full messages):

1. The app was a raylib GUI first (`f433e2f` "Init: GUI: wrote baisc raylib
   gui"); the pivot to a headless wind tunnel (`b13d185`, `65865bb`) and
   the GUI removal (`cc1dee9`) left stale GUI references and junk files
   behind until `a50966b`/`72c1254` cleaned them.
2. `--preset` lied about resolution: `cli.py` always built the default
   `TunnelSpec` (256 cells streamwise), so `--preset preview` solved an
   8x larger grid than the report claimed (`811dbdb`).
3. The drag sign convention was inconsistent: `pipeline.py:149` flipped the
   sign (`cd = -drag * scale`) while `fields.py` and
   `tests/test_cylinder.py` used the positive convention, so the sign of
   the reported Cd depended on which code path produced it (`8f46eee`).
   The old `cd > 0` test only passed by accident of that flip, and a
   leftover DEBUG print sat in `test_cylinder.py`.
4. The bundled Koenigsegg OBJ has its longest axis along Z, so the default
   run simulated a car standing on its nose (placed extent
   0.593 x 0.305 x 1.2 m) and reported a meaningless Cd for it (`fd5a285`).
5. The voxelizer's `_fill_streamwise` closed every wind column from first
   to last solid cell: a leaky shell became a solid slab, concavities like
   the car underside were filled, and the mask lied for non-watertight
   meshes (`b523794`).
6. Three tau floors disagreed: `LbmConfig` rejected `tau <= 0.5`,
   `LatticeUnits` clamped to `max(0.5, min_tau)`, and the step kernel
   clamped Smagorinsky `tau_eff` to 0.5001 (`06380b7`).
7. `report.py` claimed "Poiseuille error 0.4%" and an Re-100 Strouhal
   number the run itself never measured; those are test-level facts, not
   run-level facts (`06380b7`).
8. Dead code (`fields.surface_pressure`, `fields.sample_nearest_fluid`,
   `DyeField.reset`, `MeshTransform.reset`) and misnamed trace fields
   (`drag_trace`/`lift_trace` held coefficients) (`0df7e0e`).
9. The DDF layout `f[x, y, z, q]` strided neighbouring threads 76 bytes
   apart; global loads did not coalesce (`26aa737`).

## What changed, with measured effect

- SoA populations `f[q, x, y, z]`, one register-vector pass per cell
  (`26aa737`): bench.py (256x102x77 empty tunnel, tau 0.56, Smagorinsky
  0.12, RTX 2050, Warp 1.18.0) went 41.3 -> 123.5 MLUPs
  (48.7 -> 16.3 ms/step, 3.0x). Car preview case 13.4 s -> 5.2 s.
  Numerics unchanged: 51/51 tests passed, car-run Cd bit-identical.
- One tau floor `TAU_MIN` everywhere; runtime Mach guard samples
  `max_speed()` every 64 steps and raises once above |u| = 0.3 lattice
  units, surfacing in `SolveResult.warnings` (`06380b7`).
- Placement: `orient/flip/yaw/pitch/fraction` with a fixed order
  (orient -> flip -> yaw -> pitch -> scale -> centre -> floor -> clamp),
  proper rotations only (det +1), placement PNG via `sstream-place`
  (`0ea42ae`, `d22c757`).
- `--quality low|medium|high|ultra` + `--cells` override; grid scan picks
  the largest whole-cell grid (998k / 2.94M / 8.91M / 16.9M cells); VRAM
  from the CUDA driver via `cuMemGetInfo_v2` on the primary context
  (3.22 GiB free measured on this RTX 2050, 4 GiB), refusal above 90% of
  free; warm-up MLUPs from the first 50 steps (`wp.synchronize()` before
  reading the timer) in `SolveResult.warmup_mlups` (`223ac75`).
- `--visual`: `u_lattice = 0.06`, tau pinned to [0.505, 0.52],
  Smagorinsky 0.12; report prints the measured effective Re (922 on the
  preview grid) and states "Cd not calibrated" (`2ddee38`).
- Snapshots: `--snapshots DIR --snapshot-count N`; window = last 2
  flow-throughs floored into the second half of the run; float16 npz per
  snapshot + `meta.json` + `solid.npy`; disk estimate printed before
  solving, refusal below 1.2x free (`0add72a`).
- `fields.q_criterion(u, dx)`: six gradients once, Frobenius norms summed
  in x-slabs of <= 4M cells, float32 (`fed920d`).
- Render (`2f58b1c`): `sstream-render` with independent
  `--streamlines/--smoke/--vortices/--model` toggles (>= 1 required).
  Bugs found and fixed while testing it:
  - stored velocities are lattice units; the renderer multiplies by
    dx/dt (= 500 here) at load so advection and colours are SI;
  - `GridSampler` subtracted a half cell on top of an origin that is
    already the first cell centre (half-cell shift of every sample);
  - the streamline rake was centred on tunnel mid-height, not the model;
  - `plotter.clear()` deletes the default light kit (flat, unlit model);
    `clear_actors()` keeps it;
  - the bundled car OBJ has no normals; `compute_normals(...)` is applied
    when missing, else shading stays flat;
  - smoke emitted before the kill-rebuild lost the newborns.

## Lessons

- Measure, then claim: the report's "Validation" section was removed
  rather than left asserting numbers the run never produced.
- One convention, enforced by a test: the drag sign is covered by a
  known-sign test in a viscous fluid where the sign is physical, because
  the shared air fixture sits at tau ~ 0.5000007 where a 640-step force
  window is noise.
- Layout is performance, but only trust it with a numerics guard: the SoA
  rewrite is accepted because Cd was bit-identical, not just because it
  was 3x faster.
- Keep the render code out of the solver: `pipeline.py` never imports
  `sstream.render`; the renderer reads `meta.json` + npz files, so the
  headless pipeline stays headless.
- Offscreen VTK on this stack has sharp edges: `PYVISTA_OFF_SCREEN=true`
  (with Xvfb), x-fastest point order in VTK ImageData, `pv.Plane` is a
  1x1 quad, and OBJ files may carry no normals at all.
- Honest failure modes beat silent ones: a leak at cell scale stays fluid
  in the flood-fill voxelizer, and a low-Mach violation becomes a warning
  in the report instead of garbage numbers.

- End-to-end item-9 run (`2f58b1c`, bundled car, `--quality low --visual`,
  202x81x61 = 998,082 cells, 184.6 MB est. vs 3.5 GB free, 2,200 steps in
  21.1 s, warm-up 107.7 MLUPs, effective Re 1,454): the first all-layers
  still hid the model inside the stream/smoke layers, so the renderer
  defaults were tuned for legibility on this model - rake tubes 12x9 ->
  9x7, tube radius 0.012 -> 0.007 m, smoke 80 -> 40 particles/frame,
  peak trail alpha 1.0 -> 0.55, smoke line_width 3 -> 2.

## Open questions (to verify)

- Warm-up MLUPs measured here (~100-108 at ~1M cells) are RTX 2050 /
  Warp 1.18.0 numbers; other GPUs and the CPU fallback backend are
  **to verify**.
- The `--visual` car run pins tau into [0.505, 0.52] and reports
  Cd = -9.65 / Cl = -38.6 (nominal Re 2.4M, effective Re 1,454), while a
  non-visual preview run of the same model reported Cd = +1.68 via the
  same force code path. The known-sign test (viscous fluid, tau ~ 0.58)
  still passes, so the sign machinery is consistent; the sign and
  magnitude of the force at the pinned visual parameters is **to verify**.
  The run prints "Cd not calibrated" and makes no validation claim.
- The Q-criterion isosurface at `--q-fraction 0.2` of the smoothed max
  shows shards, not coherent rolls, because the `--visual` field is an
  effective Re 922 flow; whether coherent structures appear at a
  calibrated Re is **to verify** (no validation claim is made).
- Smoke emitter placement is derived from the model's placed bounding
  box; whether it generalises to tall or tall+wide models (bull, person)
  is **to verify** if such assets are added.
- `compute_normals(auto_orient_normals=True)` on a mirrored
  (`--flip`) model is assumed outward; the orientation of the flipped
  case is **to verify**.
- FP16S DDF storage and esoteric pull/push streaming remain open from
  `credit.md`'s future-work list; neither is part of Phase 3.

## LBM Solver Optimization Research (2025)

Research conducted on FluidX3D and other production LBM solvers for weak hardware:

**FluidX3D optimizations (reference implementation):**
- **DDF-shifting**: Store `f_shifted = f - w_i` instead of raw populations. This reduces numerical error significantly, especially for FP16. The equilibrium computation uses a specific order of operations following Skordos (1993).
- **Esoteric-Pull in-place streaming**: Eliminates double-buffering by alternating memory access patterns based on time step parity, cutting VRAM by 50%.
- **FP16 support**: Half-precision DDF storage reduces memory bandwidth by ~50%.

**Current solver status (measured):**
- Correctness: All 101 tests pass (verified 2025-01)
- Memory layout: SoA `f[q, x, y, z]` already optimized for coalescing ✓
- Double-buffering: Uses 2 copies (f and f_new arrays)
- DDF-shifting: Not used in GPU kernels (CPU function added 2025-01)
- Precision: FP32 throughout
- Performance: 177.6 MLUPs on RTX 2050 (256x102x77 empty tunnel, tau 0.56, Smagorinsky 0.12) - after Phase 1 optimizations

**Phase 1 Optimizations Implemented (2025-01):**
- **Common Subexpression Elimination (CSE)**: Pre-computed `inv_cs2`, `inv_cs2_sq`, `inv_rho`, `cu_vals` array, `speed2`, and reused throughout collision and Smagorinsky calculations. Reduced redundant floating-point operations.
- **Register Reduction**: Reused variables (`w = weights[i] * rho`, `cxx = cx * cx`, etc.) instead of recomputing, pre-computed squares (`pxx2`, `pyy2`, etc.) and constants (`smag_const`, `half_speed2_e`).
- **Result**: Performance improved from 108.7 MLUPs to 177.6 MLUPs (63.4% improvement) on RTX 2050. All 101 tests pass.

**Implementation notes:**
- Added `equilibrium_shifted()` function to `lattice.py` following FluidX3D's operation order
- Added tests verifying `f_shifted = f_eq - w_i` and density recovery
- The CPU function is a foundation; GPU kernel integration would require:
  - Modifying `_step_kernel` to use shifted populations
  - Adjusting initialization and boundary condition kernels
  - Adding a config flag to enable/disable DDF-shifting

**Trade-offs for future optimization:**
- **DDF-shifting**: Improves numerical accuracy (especially for low tau/FP16), but requires kernel changes. Benefit depends on target precision.
- **Esoteric-Pull**: 50% VRAM reduction, but adds complexity to streaming logic. High value for memory-constrained GPUs.
- **FP16**: 50% memory bandwidth reduction, but may require careful handling of precision-sensitive operations. Best combined with DDF-shifting.

## Validation Framework (2026-11-20)

**Status**: Complete. Both validation cases pass.

**Validation cases:**
- **Poiseuille flow**: Exact solution comparison, 0.38% max error (threshold 1.5%). Grid 4×20×4, 12k steps, ~4 MLUPs (tiny grid, not perf-representative).
- **Cylinder Re=100**: Matches `tests/test_cylinder.py` setup. Grid 80×60×4, D=8, 30k steps. **Cd=1.583, St=0.184, Cl=0.324** — all within expected bands (Cd 1.35-1.85, St 0.16-0.21, Cl 0.15-0.60). ~100 MLUPs.

**CLI**: `python -m sstream.cli --validate --out DIR` runs both cases, saves JSON, prints summary.

**CSV export**: `--export-csv` flag writes `force_history.csv` with step, cd, cl, cl_amplitude.

## Profiling Findings (2026-11-20)

**Hardware**: RTX 2050 (4GB VRAM, sm_86, ~192 GB/s theoretical bandwidth)

**Baseline**: 176.7 MLUPs on 256×102×77 empty tunnel (tau=0.56, smag=0.12, inlet=0.06)

**Achievable bandwidth** (copy kernel): 105.2 GB/s

**LBM kernel bandwidth usage**: ~32 GB/s (30% of achievable)

**Bottleneck analysis:**
- **Memory-bound, not compute-bound**: Smagorinsky=0 vs 0.12 only 3% difference
- **Occupancy-limited**: `vec19` register vector (19 floats/thread) causes ~15% occupancy
- **Register limiting hurts**: cuda_max_registers=128/192/224 all worse (spills)
- **Inlet/outlet overhead**: ~9% (172.5 vs 188.7 MLUPs periodic)
- **VRAM not limiting**: 364 MB / 4 GB for benchmark grid

**Kernel is fused**: Single launch per step (streaming + collision). No loop fusion gain available.

**Optimization candidates:**
1. **FP16 storage**: TESTED AND REJECTED (see below).
2. **Esoteric-pull**: NOT needed (VRAM not limiting).
3. **Occupancy improvement**: Hard without restructuring vec19 or using shared memory.
4. **Boundary check optimization**: Minor gain, complex refactor.

## FP16 Experiment: Failed (2026-11-20)

**Status**: Reverted. FP32 is the only supported precision.

**What was tried**: Store populations as float16, compute in float32. Halves memory traffic.

**Results**:
- Performance: Only 7% gain (184 vs 173 MLUPs) — kernel is occupancy-limited, not bandwidth-bound
- Poiseuille: 100% error (body force ~5e-5 lost in FP16 precision at f~0.01)
- Cylinder Re=100: Cd=1.42 vs 1.58 (10% low), St=0.008 vs 0.184 (no vortex shedding), Cl=0.000

**Root cause**: LBM populations need ~4-5 decimal digits of precision; FP16 has ~3. Small forces and instabilities are lost.

**Path forward** (if ever needed): DDF-shifting (store f_i - f_eq_i) plus a custom 16-bit format like FluidX3D's FP16C (Lehmann et al., Phys. Rev. E 2022). Not worth the complexity for 7% gain on this hardware.
