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

## Open questions (to verify)

- Warm-up MLUPs measured here (~100-108 at ~1M cells) are RTX 2050 /
  Warp 1.18.0 numbers; other GPUs and the CPU fallback backend are
  **to verify**.
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
