# S-Stream Trust Report

Capability status as of the v2 trust-first roadmap. Labels:

| Label | Meaning |
|-------|---------|
| **Verified** | Named CI test with analytic/literature tolerance |
| **Experimental** | Implemented; may be inaccurate; off/badged by default |
| **Hidden** | Broken or unwired; not advertised |

See [README.md](README.md) for installation and usage.

## Verified (CI)

Run: `pytest tests/validation/ -v` — 9 tests, all currently passing.

| Capability | Test | Criterion |
|------------|------|-----------|
| Lattice pressure `p = ρ/3` | `test_pressure_definition` | Matches `rho/3` |
| BGK force-driven Poiseuille | `test_poiseuille_l2` | L2(`u`) < 2% vs analytic |
| BGK force-driven Poiseuille (fine grid) | `test_poiseuille_l2_fine` | L2(`u`) < 2% vs analytic |
| BGK lid-driven cavity Re=100 | `test_lid_cavity_ghia_re100` | Midplane Ghia within 5% |
| TRT lid-driven cavity Re=100 | `test_trt_lid_cavity` | Midplane Ghia within 5% |
| Periodic TGV KE decay | `test_tgv_ke_decay` | Relative KE error < 10% |
| Cylinder Cd Re≈40 | `test_cylinder_cd_re40` | Cd in [1.4, 2.8] (confined) |
| Closed-domain mass | `test_mass_conservation_closed` | Drift < 0.1% |
| Thermal buoyancy produces convection | `test_thermal_buoyancy_convection` | Non-zero `v`; `T(row 0) > T(last row)` |

"Verified" here means *the named test exists and passes*. It does not mean the
capability is free of untested parameter ranges — see the coverage gaps below.

## Boundary-condition contracts (CI, non-benchmark)

These are invariants rather than accuracy benchmarks, but they are what stop a
silent physics regression, so they are tracked here too.

| Contract | Test |
|----------|------|
| `use_fused` never changes obstacle physics | `test_lbm.py::TestFusedPathContract` |
| Obstacles apply in periodic mode | `test_lbm.py::TestPeriodicDomain` |
| Halfway bounce-back does not alias on a double-sided wall | `test_lbm.py::TestHalfwayBounceBack` |
| No domain BC writes into a solid | `test_lbm.py::TestBoundaryConditionsRespectObstacles` |
| Each engine owns its initial condition | `test_scene_reset.py` |
| Boussinesq force delivers exactly `F`; correct sign; all directions | `test_thermal.py` |
| `top`/`bottom` temperature regions match the moving-wall convention | `test_thermal.py::TestTemperatureRegionNaming` |

## Experimental

| Capability | Notes |
|------------|-------|
| TRT collision (2D) | Better walls than BGK; no 3D. Verified at Re=100 cavity only |
| Smagorinsky SGS | `‖S‖` from corrected `Π_neq` formula |
| WALE | ∇u form incomplete |
| MRT (2D) | `s_pxy=ω` fix applied; no 9-moment verification |
| Zou-He BC | Implemented; EQ inflow is the default (more stable) |
| Halfway bounce-back | Default obstacle BC; now genuinely honoured on every path |
| Thermal buoyancy (2D + 3D) | Runs in the default `step()` and is reachable from the scene panel. One convection benchmark only; no Nusselt/Rayleigh number verification |
| Rigid IBM + FSI | `engines/ibm.py`, `engines/fsi.py` |
| Free-surface tracker | Volume fraction advection scaffolding |
| Adjoint FD hooks | `engines/adjoint.py` finite differences |
| Shan-Chen liquid / multi-component | Visual phase separation; no Laplace-Young or contact-angle test |
| Non-Newtonian models | Variable-ω BGK. Bingham / Carreau / power-law wired from the UI |
| CuPy 2D / 3D GPU, Lettuce | Throughput paths; feature parity incomplete |
| 3D viewport midplane slice | Slice of 3D fields in OpenGL viewport |

## Hidden (do not advertise)

| Capability | Why |
|------------|-----|
| Gemini / live AI tutor | Coming soon — Outcome panel owns storytelling |
| MPI / multi-GPU | Deferred until single-GPU saturation |

## Known coverage gaps

Stated so that "Verified" is not read as "finished".

- **Fused BGK fast path.** Only used when the obstacle mask is empty, the
  operator is plain BGK and no force term is required. Any obstacle scene takes
  the slower path so `obstacle_bc` is honoured.
- **3D thermal coupling** has one convection benchmark. Buoyancy magnitude,
  Prandtl number and the Rayleigh–Bénard onset are unverified.
- **Multiphase** has no quantitative test at all — no Laplace-Young, no contact
  angle, no droplet-mass drift bound.
- **Non-Newtonian** models are checked for correct wiring, not for rheological
  accuracy (no pressure-drop-per-shear-rate comparison against a power law).
- **Obstacle BC in 3D** is not covered by the boundary-contract tests above,
  which exercise the 2D engine only.
- **Thermal buoyancy is not applied inside obstacles**; solids carry no
  temperature. That is intentional but unvalidated against a heated blockage.

## Mach / Re operating envelope

Hydro presets should use `U ≲ 0.05` (Ma ≲ 0.09) and `ν = U L / Re`.
Sanity warnings fire when `u_inflow > 0.1` or `ω` leaves `(0, 2)`.

## Lattice units

| Symbol | Meaning |
|--------|---------|
| `Δx = Δt = 1` | Lattice spacing and timestep |
| `c_s² = 1/3` | Speed of sound squared |
| `p = ρ/3` | Lattice pressure (display often uses gauge `p − 1/3`) |
| `ν = (τ − 1/2) / 3` | Kinematic viscosity; `ω = 1/τ` |

## Coordinate conventions

Row 0 is the **bottom** wall and row `height-1` is the **top** wall, matching
`apply_moving_wall_bottom` / `_top` and `set_temperature_boundary(region=...)`.
`v > 0` points from the bottom wall toward the top wall. The viewport flips the
texture on upload, so the rendered image is not in array order.