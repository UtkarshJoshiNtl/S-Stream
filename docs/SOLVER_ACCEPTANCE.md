# Solver acceptance (Stage 1)

Scope: headless solver acceptance checks (no rendering changes). Tests/results below are for the current D3Q19 BGK+Smagorinsky LBM pipeline as exercised by automated tests and one scripted stability run.

## Summary

| Item | Test | Expected | Measured | Pass/Fail |
|---|---|---|---|---|
| 1 | Sphere drag at Re ≈ 100 with force-window convergence check (slow, `simulation/tests/test_sphere_re100.py`) | Mean Cd within 20% of reference **Cd = 1.087** (Johnson & Patel 1999, JFM 378, https://doi.org/10.1017/S0022112098003206, as recorded in `docs/REFERENCES.md`; value marked `to verify` against primary text) and late force window converged (relative change below tolerance). **No blockage correction applied** (blockage recorded). | Cd = **0.9084** (16.4% below reference), Re = 100.00, tau = 0.514369, Mach(u_lattice/√(1/3)) = 0.0866, bbox blockage = 2.7689%, silhouette blockage = 2.2003%, cd_trace samples = 163, late-window relative change = 0.04874 < tol 0.06872. | **PASS** |
| 2 | Outlet reflection sensitivity: same sphere case with outlet 1× and 2× farther downstream (slow, `simulation/tests/test_outlet_reflection.py`) | Drag differs by < 5% with existing outlet BC (same dx, model size/position relative to inlet, fluid/numerics). If fail: replace zero-gradient outlet with convective outlet or sponge layer and re-run. | Cd(1× outlet, L=4.0 m) = **1.25096**, Cd(2× outlet, L=6.0 m) = **1.26137**, relative difference = **0.833%**, sphere diameter = 0.3991 m. Existing zero-gradient outlet retained (no replacement needed). | **PASS** |
| 3 | Stability: 30,000 steps at `--quality medium` on Ahmed body run 1 (decimated) in `--visual` mode | No NaN/Inf; max \|u\| below Mach limit throughout. Report wall time and MLUPs. | Completed 30,000 steps on `cuda` (NVIDIA GeForce RTX 2050). **finite_vel = True, finite_rho = True**, max lattice speed \|u\| = **0.076022 < 0.3** (Mach limit), u_lattice_inlet = 0.06, tau = 0.505 (visual, pinned). Elapsed = **772.476 s**, wall = **773.025 s**, warmup = **111.218 MLUPs**. Grid = **317×119×79** (2,980,117 cells), solid = 16,033, dx ≈ 0.015142 m (cubic), tunnel = 4.8×1.8×1.2 m. Cd = 0.7587, Cl = -819.20 (visual/demo only, see notes). Warnings: mesh not watertight after closing small holes; unreachable cells filled; surface mask approximate; **Visual mode: tau pinned and parameters chosen for stability/visuals; Cd not calibrated.** | **PASS** (stability/finite/Mach criteria) |
| 4 | Convergence detector: stop when drag window changes by less than a tolerance; say clearly if it never converged | Detects convergence by relative change between consecutive drag sample windows and stops early when below tolerance; reports clearly when it never converged. | Implemented in `simulation/sstream/pipeline.py` (`SolvePlan.convergence_tol`, `convergence_window`, `SolveResult.converged/convergence_change/convergence_tol`) and exposed in CLI (`--convergence-tol`, `--convergence-window`). Result warnings state either "Convergence: reached at step ..." or "Convergence: never converged..." (including insufficient-samples case). Automated check: `test_convergence_detector_reports_never_converged_when_tolerance_unmet` (pipeline test) asserts `converged is False` and warning contains "Convergence: never converged". | **PASS** |

## Item 3 stability run details (exact setup)

- Case: `simulation/assets/3d-files/ahmed_1.stl` (AhmedML run 1 morphed variant), decimated to 100,000 faces, `--quality medium` (`resolve_target_cells('medium')`), `--visual` mode, 30,000 steps.
- Orientation/placement: `MeshSpec(orient='auto-vehicle')` with default transform; tunnel sized from model length via `TunnelSpec.from_model_length(...)` then resolved to medium target (`cells_streamwise_for_target`).
- Fluid: `FluidSpec(speed=30.0 m/s, kinematic_viscosity=1.5e-5 m^2/s, reference_length=1.2 m)` (scripted acceptance run).
- Device: `cuda` (NVIDIA GeForce RTX 2050, 4 GiB, sm_86), Warp 1.18.0.
- Performance/stability: warmup 111.21803139778237 MLUPs, elapsed 772.4763619600017 s, wall 773.0246979259973 s, `finite_vel True`, `finite_rho True`, `max_u_lattice 0.07602221518754959` (< 0.3), `u_lattice_inlet 0.06`, `tau 0.505`.
- Notes: visual mode pins tau/u_lattice/Smagorinsky for stability/visuals and is **not Cd-calibrated**; forces/Cd/Cl from this run are not treated as validated drag results.

## Explicit statements on demo/validation shapes

- **Ahmed and DrivAer are visual demo shapes** in this project context (used for plausible wake/separation visuals and solver robustness/stability exercises), not quantitative drag-validation targets.
- **Drag is not compared with published high-Re Ahmed/DrivAer values.** Published Ahmed-body data are at much higher Reynolds numbers/conditions than those reachable in the present setup; reported forces from Ahmed/DrivAer demo runs must not be treated as matches to literature Cd.
- `ahmed_1.stl` is an AhmedML run 1 morphed/dataset variant (not the canonical textbook Ahmed body); `ahmed_box.stl` is a plain rectangular prism with reference dimensions (not an Ahmed body). Any comparison must account for the exact variant, conditions, and solver regime.
- Low-Re quantitative validation in this stage uses the **sphere at Re ≈ 100** (and cylinder tests exist elsewhere in the suite) with appropriate domain/blockage caveats stated.

## What was NOT tested

- No comparison of Ahmed or DrivAer drag/lift against published high-Re wind-tunnel/CFD reference values (and no claim of calibrated Cd in visual mode).
- No blockage-corrected drag (sphere item 1 reports raw Cd and blockage ratio only; no correction applied).
- No grid/time-step independence/convergence study beyond the specific force-window checks reported (item 1 window check; item 4 detector capability demonstrated by automated test and available for runs).
- No outlet boundary-condition replacement study (convective outlet/sponge layer not implemented) because the zero-gradient outlet passed the item 2 sensitivity criterion (<5% Cd change).
- No quantitative validation at high Reynolds numbers or for production-vehicle aerodynamics accuracy; no turbulence-model calibration/certification.
- No rendering/visualization pipeline validation (out of scope; no rendering code changed).
- No multi-GPU/distributed performance scaling tests; performance reported for a single GPU (RTX 2050) run only.
- No systematic uncertainty quantification (numerical, geometric, BC, or blockage uncertainty) beyond reported checks/warnings.
- Stability evidence is limited to the single item 3 configuration/run reported above (30,000 steps, medium quality, visual mode, Ahmed run 1 decimated).

## References

- See `docs/REFERENCES.md` for low-Re sphere validation reference (Johnson & Patel 1999), LBM/external-aero notes, and verification-status annotations.
