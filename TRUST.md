# S-Stream Trust Report

Capability status as of the v2 trust-first roadmap. Labels:

| Label | Meaning |
|-------|---------|
| **Verified** | Named CI test with analytic/literature tolerance |
| **Experimental** | Implemented; may be inaccurate; off/badged by default |
| **Hidden** | Broken or unwired; not advertised |

See [README.md](README.md) for installation and usage.

## Verified (CI)

| Capability | Test | Criterion |
|------------|------|-----------|
| Lattice pressure `p = ρ/3` | `test_pressure_definition` | Matches `rho/3` |
| BGK force-driven Poiseuille | `test_poiseuille_l2` | L2(`u`) < 2% vs analytic |
| BGK lid-driven cavity Re=100 | `test_lid_cavity_ghia_re100` | Midplane Ghia within 5% |
| Periodic TGV KE decay | `test_tgv_ke_decay` | Relative KE error < 10% |
| Cylinder Cd Re≈40 | `test_cylinder_cd_re40` | Cd in [1.4, 2.8] (confined) |
| Closed-domain mass | `test_mass_conservation_closed` | Drift < 0.1% |

Run: `pytest tests/validation/ -v`

## Experimental

| Capability | Notes |
|------------|-------|
| TRT collision (2D) | Better walls than BGK; no 3D |
| Smagorinsky SGS | `‖S‖` from corrected `Π_neq` formula |
| WALE | Marked experimental — ∇u form incomplete |
| MRT (2D) | `s_pxy=ω` fix applied; still Experimental |
| Zou-He BC | Implemented; EQ inflow is default (more stable) |
| Halfway bounce-back | Default obstacle BC |
| Rigid IBM + FSI | `engines/ibm.py`, `engines/fsi.py` |
| Free-surface tracker | Volume fraction advection scaffolding |
| Adjoint FD hooks | `engines/adjoint.py` finite differences |
| Shan-Chen liquid / multi-component | Visual phase separation; Laplace pending |
| Non-Newtonian models | Variable-ω BGK |
| CuPy 2D / 3D GPU, Lettuce | Throughput paths; feature parity incomplete |
| 3D viewport midplane slice | Slice of 3D fields in OpenGL viewport |

## Hidden (do not advertise)

| Capability | Why |
|------------|-----|
| Thermal buoyancy in product UI | Opt-in via `thermal_enabled`; not default `step()` |
| Gemini / live AI tutor | Coming soon — Outcome panel owns storytelling |
| MPI / multi-GPU | Deferred until single-GPU saturation |

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
