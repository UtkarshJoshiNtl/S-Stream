# References: external aerodynamics and LBM notes

This document summarizes reference material and open-source LBM implementation notes relevant to S-Stream's external-aerodynamics direction. It was compiled for design ideas only; **no code was copied** from the referenced projects.

**Verification status:** claims below are tied to the cited sources/links where possible. **Every claim not directly confirmed from a cited source during this pass is marked `to verify`.** Numerical values reproduced from secondary summaries (rather than the primary paper text itself) are also marked `to verify` where the primary text was not inspected here.

## Credits and learnings from FluidX3D

[ProjectPhysX/FluidX3D](https://github.com/ProjectPhysX/FluidX3D) is a
single-file C++ LBM wind-tunnel solver by Dr. Moritz Lehmann: D3Q19 SRT
collision, half-way bounce-back walls, Guo forcing, Smagorinsky-Lilly LES,
and GPU voxelization. It is the reference implementation for what S-Stream's
solver is trying to be.

### License compliance (read this before touching FluidX3D source)

`LICENSE.md` in FluidX3D is *source-available, no-cost, non-commercial*:

1. You must not misrepresent the origin; altered versions must be marked.
2. No commercial use.
3. No military use.
4. **The source must not be used to train AI models.**
5. Altered source must be published if you publish binaries or results.
6. The papers below must be cited.

What S-Stream does instead: **no FluidX3D source code is copied, ported, or
transcribed into this project.** The ideas were studied (papers, docs,
`defines.hpp` constants), then re-implemented from scratch in Python/Warp,
with different data structures and different kernel code. This file is the
attribution record; alterations to S-Stream that were inspired by FluidX3D
are marked `FluidX3D-style` in comments.

### Computational lessons (idea → where it landed)

| FluidX3D idea | S-Stream status |
|---|---|
| FP16S DDF storage: range-shifted float16; all arithmetic stays FP32 | **Not done** (future work). Would halve VRAM and double effective bandwidth. |
| Esoteric Pull/Push: in-place single-buffer streaming, AA pattern (doi 10.3390/computation10060092) | **Not done** (future work). S-Stream currently double-buffers `f`/`f_new` in FP32. |
| Reduce on device, ship only scalars: `max_speed`, `total_mass` as single floats | **Partially done**: `LbmSolver.total_wall_force` sums per-cell forces on the GPU via `lbm._force_sum_kernel` and copies back three floats. `max_speed`/`total_mass` still copy whole fields to the host. |
| Re-voxelize on geometry change, keep the solver resident otherwise | Done: `pipeline.run_case` builds the mask from the placed mesh and constructs the solver from it. |
| Stability discipline: `u < 0.57`, `rho ≈ 1`, `nu ≈ 1/6` | Partially done: `LatticeUnits.from_physical` clamps `u_lattice`; a runtime warning fires when `max|u| > 0.3`. A hard auto-pause guard is still future work. |
| Flag bitfield for walls/inlet/outlet | **Not done**: S-Stream uses a boolean solid mask. Works, but one flag array would unify inlet/outlet/moving-solid handling later. |
| `UPDATE_FIELDS` recompute rho/velocity every step | Already the case: `lbm.py` updates fields every step. |

### Citations required by the FluidX3D license

- M. Lehmann et al., *Esoteric Pull and Esoteric Push: Two Streamings
  In-place for 2D and 3D Lattice Boltzmann on GPUs*, Computation 10(6):92,
  2022. doi:10.3390/computation10060092
- M. Lehmann, *Accuracy and performance of the lattice Boltzmann
  method with half-way bounce-back and with force feedback*,
  Phys. Rev. E 106, 015308 (2022).
- M. Lehmann et al., *Combined scientific CFD simulation and interactive
  raytracing of free-surface flows using a single GPU*, IWOCL 2022.
- M. Lehmann, *High-performance free-surface flow simulation with
  raytraced visualization on a single GPU* (MSc thesis).

### Future work (ranked by value)

1. SoA (`f[q, x, y, z]`) population layout in Warp; drop the duplicate
   `feq` evaluation in the Smagorinsky branch of `_step_kernel`.
2. Device-side reduction for `max_speed` / `total_mass` and skip host
   readbacks when nothing consumes them.
3. FP16S DDF storage (`wp.float16` + per-array scale factor).
4. Esoteric pull streaming (single buffer, in-place).
5. Hard stability guard: abort + status warning when `max_speed` exceeds
   ~0.3 or density drifts from 1.

## 1. Ahmed body reference drag values

| Source | Slant angle | Reynolds number | Reported drag (C<sub>D</sub>) / notes | Relevance | Status |
|---|---:|---:|---|---|---|
| Ahmed, S. R., Ramm, G., & Faltin, G. (1984). *Some Salient Features Of The Time-Averaged Ground Vehicle Wake*. SAE Technical Paper 840300. https://doi.org/10.4271/840300 | 25° and 35° are the commonly cited critical slant configurations in the Ahmed-body literature (`to verify` exact values/conditions from the SAE paper text) | Wind-tunnel experiments on a bluff body in ground proximity; commonly cited around Re ≈ 4.29×10<sup>6</sup> based on body length (`to verify` exact Re definition/value from paper text) | Canonical result: low-drag/high-drag wake regimes separated by a critical rear slant angle near ~30°; strong C<sub>D</sub> change between ~25° and ~35° slant (`to verify` exact C<sub>D</sub> numbers from paper text) | Defines the standard bluff-body benchmark geometry/wake regimes S-Stream's bundled `ahmed_1.stl` validation shape is meant to relate to. | `to verify` (paper metadata/citation confirmed via DOI page; detailed numbers/conditions not extracted from full text in this pass) |
| Lienhart, H., & Becker, S. (2003). *Flow and Turbulence Structure in the Wake of a Simplified Car Model*. SAE Technical Paper 2003-01-0656. https://doi.org/10.4271/2003-01-0656 | 25° and 35° backlight/slant angles (LDA measurements) (`to verify`) | Re = 4.25×10<sup>6</sup> based on model length (commonly cited for the Lienhart/Becker experiments) (`to verify`) | Widely used experimental reference for mean flow/turbulence and force data for 25°/35° Ahmed bodies (`to verify` exact C<sub>D</sub> values reported) | High-quality benchmark data for validating separation, wake structure, and drag trends on the Ahmed body. | `to verify` (citation/DOI known; specific values not directly extracted here) |
| AhmedML dataset: Ashton, N., Maddix, D., Gundry, S., Shabestari, P. (2024). *AhmedML: High-Fidelity Computational Fluid Dynamics dataset for incompressible, low-speed bluff body aerodynamics*. arXiv:2407.20801. https://arxiv.org/abs/2407.20801 ; dataset https://huggingface.co/datasets/neashton/ahmedml (CC BY-SA 4.0) | 500 geometric variations of the Ahmed car body (slant/geometry parameterized per dataset) (`to verify` per-case slant angles without inspecting `geo_parameters` files) | Hybrid RANS–LES (OpenFOAM) incompressible, low-speed bluff-body cases; exact Re/conditions per run documented in dataset/paper (`to verify` per-run Re without inspecting files) | Provides per-run force/moment outputs (e.g. `force_mom_i.csv`, Cd/Cl, time-averaged) rather than a single canonical Cd; useful as a modern CFD reference envelope. | Bundled `ahmed_1.stl` in this repo comes from AhmedML run 1; dataset is a directly relevant geometry/force reference source. | Partly confirmed (dataset contents, solver description, license, and presence of force files confirmed from dataset card); per-run drag numbers/slant/Re `to verify` without opening case files. |

**Notes:** the Ahmed body is the standard simplified bluff-body benchmark; the key validation signal is the drag/wake transition around the critical rear slant (~25° vs ~35° regimes). Any exact C<sub>D</sub> value used for regression should be taken from the specific primary experiment/configuration (slant angle, Reynolds number definition, ground clearance, yaw, blockage) because published values vary with setup (`to verify` for any value not read from the primary source).

## 2. Open-source LBM codes (BCs, outlet treatment, tracers, license)

| Project | External-flow / wind-tunnel BCs | Outlet treatment | Passive smoke / scalar tracers | License | Relevance | Status |
|---|---|---|---|---|---|---|
| FluidX3D — https://github.com/ProjectPhysX/FluidX3D (README: https://github.com/ProjectPhysX/FluidX3D/blob/master/README.md, license: https://github.com/ProjectPhysX/FluidX3D/blob/master/LICENSE.md) | README describes LBM boundary types including equilibrium boundaries (non-reflective inflow/outflow), moving/stationary bounce-back solids, temperature boundaries, and force/force-field support; suitable primitives for external-flow setups (`to verify` exact wind-tunnel setup patterns/best practices). | Equilibrium inflow/outflow boundaries are listed; detailed pressure/outlet stabilization behavior for external aero `to verify` from docs/source. | No explicit passive smoke/scalar tracer feature stated in the inspected README excerpt; particles via immersed-boundary method are supported (`to verify` whether a scalar/advection field exists). | **Custom, source-available, non-commercial** license (FluidX3D `LICENSE.md`, © Dr. Moritz Lehmann): permits public research/education/personal use with restrictions; **commercial use not allowed**, military use not allowed, and other conditions apply. Not an OSI open-source license. | High-performance GPU LBM reference (memory-efficient FP16/Esoteric-Pull, LES Smagorinsky mentioned) for implementation ideas on performance/stability. | Partly confirmed from README and `LICENSE.md`; detailed BC/outlet/tracer capabilities beyond README `to verify`. |
| XLB — https://github.com/Autodesk/XLB | JAX/Python differentiable LBM framework aimed at scalable CFD; supports standard LBM boundary-condition constructs in examples/docs (`to verify` specific external-flow/wind-tunnel BC set and recommended configuration from current docs). | `to verify` (outlet/non-reflecting treatment details not confirmed in this pass). | `to verify` (passive scalar/tracer support not confirmed in this pass). | **Apache License 2.0** (confirmed from `https://raw.githubusercontent.com/Autodesk/XLB/main/LICENSE`). | Modern, differentiable/GPU-oriented LBM codebase; useful reference for Python/JAX workflows and extensibility patterns. | License confirmed; feature details `to verify`. |
| Palabos — https://palabos.unige.ch/ | Mature C++ LBM library widely used for CFD; provides boundary-condition infrastructure commonly used for channel/external flows (`to verify` exact wind-tunnel/characteristic BC classes and recommended external-aero usage from docs). | `to verify` (specific outlet/convective/pressure treatments and stability guidance not confirmed in this pass). | Passive-scalar/advection-diffusion capabilities are commonly associated with Palabos in the LBM ecosystem (`to verify` exact API/module names and usage from current documentation). | **GNU AGPL v3** stated by the Palabos project (project site states AGPLv3; `to verify` exact license text/file for the version in use). | Established reference implementation for complex BCs, coupling, and validation workflows in LBM. | Project-stated license noted; technical details and exact license file `to verify`. |
| waLBerla — https://www.walberla.net/ (docs: https://www.walberla.net/doxygen/index.html) | Massively parallel C++ multiphysics/LBM framework; supports configurable collision models, link-wise boundary conditions, and coupled multiphysics setups (confirmed from project site). Wind-tunnel/external-flow application patterns `to verify` from examples/docs. | `to verify` (specific outlet/non-reflecting/pressure boundary treatments not confirmed in this pass). | Framework supports multiphysics/coupled simulations; passive scalar/tracer transport support `to verify` from docs. | **GNU GPL v3** stated on project site ("published as free software under the GNU GPLv3"). | Strong reference for scalable block-structured LBM, boundary-condition composition, and multiphysics coupling. | License and high-level features confirmed from site; detailed BC/outlet/tracer specifics `to verify`. |

## 3. LBM external aerodynamics at high Re, Smagorinsky/LES closure, stability near τ = 0.5

| Source | Key points (relevance) | Status |
|---|---|---|
| Smagorinsky, J. (1963). *General Circulation Experiments with the Primitive Equations*. Monthly Weather Review. https://doi.org/10.1175/1520-0493(1963)091<0099:GCEWTP>2.3.CO;2 | Original Smagorinsky subgrid-scale viscosity concept used in LES closures; in LBM this is commonly implemented as an eddy-viscosity correction that increases the effective relaxation time locally. Relevant as the foundational LES closure model. | Citation confirmed as canonical reference; equation/application details in LBM context `to verify` against implementation notes. |
| Hou, S., Sterling, J., Chen, S., & Doolen, G. D. (1996). *A lattice Boltzmann subgrid model for high Reynolds number flows*. (commonly cited; e.g. Fields Institute/arXiv-era literature) https://doi.org/10.1090/fic/006/13 (`to verify` canonical accessible link) | Early LBM Smagorinsky subgrid formulation for high-Re flows; relates strain-rate-dependent eddy viscosity to modified relaxation time, improving stability at high Re. | `to verify` (widely cited; not directly retrieved/verified here). |
| Krafczyk, M., Tölke, J., & Luo, L.-S. (2003). *Large-eddy simulations with a multiple-relaxation-time LBE model*. International Journal of Modern Physics B. https://doi.org/10.1142/S0217979203017059 (`to verify`) | MRT-based LES is often reported to improve stability/accuracy over BGK at high Re and low viscosity (τ → 0.5). Relevant if S-Stream considers MRT/TRT for external aero. | `to verify`. |
| Yu, H., Luo, L.-S., & Girimaji, S. S. (2006). *LES of turbulent square jet flow using an MRT lattice Boltzmann model*. Computers & Fluids. https://doi.org/10.1016/j.compfluid.2005.04.009 (`to verify`) | Example of MRT+LES for turbulent jets; illustrates stability/quality benefits of advanced collision models with subgrid closure. | `to verify`. |
| General LBM stability note (BGK, low viscosity): as Re increases for fixed grid/velocity, kinematic viscosity ν decreases and BGK relaxation time τ = ν/c_s² + Δt/2 approaches 0.5 from above; near τ ≈ 0.5 BGK becomes prone to numerical instability/oscillations. Common mitigations: Smagorinsky/LES eddy viscosity (raises effective τ locally), TRT/MRT/cascaded/entropic/cumulant collisions, higher resolution, lower Ma, and robust/non-reflecting inlet/outlet BCs. | Directly relevant to S-Stream external-aero robustness targets and to interpreting "stability near tau = 0.5" in solver experiments. | Core τ-relation and qualitative instability/mitigation statements are standard LBM theory but **specific recommendations should be treated as `to verify` against the chosen model/implementation and cited primary sources before use as design constraints.** |
| FluidX3D README/features (https://github.com/ProjectPhysX/FluidX3D) — lists "Smagorinsky-Lilly subgrid turbulence LES model to keep simulations with very large Reynolds number stable" and documents collision/operator/precision options. | Practical example of an open GPU LBM code exposing Smagorinsky LES for high-Re stability; useful implementation reference (ideas only). | Feature presence confirmed from README; implementation details/effectiveness for a given setup `to verify`. |

## 4. Low-Re quantitative validation references (sphere/cylinder)

These are appropriate for solver acceptance at Reynolds numbers reachable by the present setup (unlike published high-Re Ahmed-body Cd values).

| Case | Source | Reynolds number | Reference drag / notes | Relevance | Status |
|---|---|---:|---|---|---|
| Smooth sphere, steady laminar wake | Johnson, T. A., & Patel, V. C. (1999). *Flow past a sphere up to a Reynolds number of 300*. Journal of Fluid Mechanics, 378, 19–70. https://doi.org/10.1017/S0022112098003206 | Re = 100 (based on sphere diameter and freestream velocity) | Reported total drag coefficient **C<sub>D</sub> = 1.087** at Re = 100 (steady, axisymmetric regime; `to verify` exact figure/table location in the primary text during this pass, but value/DOI widely cited) | Standard low-Re benchmark for sphere drag/convergence checks in external-flow solvers. | `to verify` (DOI/citation recorded; numeric value taken from commonly cited result for this paper, not re-extracted from the full text here) |
| Smooth sphere (correlations, context) | Clift, R., Grace, J. R., & Weber, M. E. (1978). *Bubbles, Drops, and Particles*. Academic Press. (see also standard drag correlations/compilations) | Re = 100 | Standard correlations give C<sub>D</sub> ≈ 1.087–1.10 at Re = 100 depending on correlation/data (`to verify` exact recommended value from source) | Context/cross-check for sphere drag at Re ≈ 100. | `to verify` |

**Acceptance note:** comparisons must account for domain/blockage, inlet/outlet placement, numerical resolution, and whether any blockage correction is applied (state explicitly in tests/reports). No blockage correction is implied unless documented.

## 5. Lookups / follow-ups to verify (actionable)

- Extract exact Ahmed-body reference C<sub>D</sub> values (with slant angle, Reynolds number definition/value, ground clearance, blockage, yaw) from the primary texts: Ahmed et al. SAE 840300 and Lienhart & Becker SAE 2003-01-0656.
- For `ahmed_1.stl` (AhmedML run 1), record run-specific slant/geometry parameters and reported Cd/Cl from dataset files (`geo_parameters*.csv`, `force_mom_1.csv`) to tie the bundled validation mesh to a concrete condition.
- Confirm from official docs/source: Palabos outlet/characteristic BC classes and passive-scalar modules; waLBerla external-flow BC examples and scalar transport facilities; XLB boundary-condition and tracer/scalar support.
- Collect primary references for LBM high-Re external-aero setups (inlet turbulence, sponge/convective outlets, grid resolution/y⁺ analogs in LBM units, Mach/τ limits) and validate the τ→0.5 stability guidance against MRT/TRT/entropic/cumulant variants.
- If using FluidX3D as an ideas reference, re-check license terms for the exact version/date before any closer study beyond public docs (custom non-commercial restrictions apply).

*Compiled 2026-11-16; updated 2026-11-17 with low-Re sphere validation reference notes. Web research performed for ideas only (no code copied).*