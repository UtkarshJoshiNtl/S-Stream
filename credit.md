# Credits and learnings from FluidX3D

## What FluidX3D is

[ProjectPhysX/FluidX3D](https://github.com/ProjectPhysX/FluidX3D) is a
single-file C++ LBM wind-tunnel solver by Dr. Moritz Lehmann: D3Q19 SRT
collision, half-way bounce-back walls, Guo forcing, Smagorinsky-Lilly LES,
and GPU voxelization. It is the reference implementation for what S-Stream's
solver is trying to be.

## License compliance (read this before touching FluidX3D source)

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

## Computational lessons (idea → where it landed)

| FluidX3D idea | S-Stream status |
|---|---|
| FP16S DDF storage: range-shifted float16; all arithmetic stays FP32 | **Not done** (future work). Would halve VRAM and double effective bandwidth. |
| Esoteric Pull/Push: in-place single-buffer streaming, AA pattern (doi 10.3390/computation10060092) | **Not done** (future work). S-Stream currently double-buffers `f`/`f_new` in FP32. |
| Reduce on device, ship only scalars: `max_speed`, `total_mass` as single floats | **Partially done**: `LbmSolver.total_wall_force` sums per-cell forces on the GPU via `lbm._force_sum_kernel` and copies back three floats. `max_speed`/`total_mass` still copy whole fields to the host. |
| Re-voxelize on geometry change, keep the solver resident otherwise | Done: `pipeline.run_case` builds the mask from the placed mesh and constructs the solver from it. |
| Stability discipline: `u < 0.57`, `rho ≈ 1`, `nu ≈ 1/6` | Partially done: `LatticeUnits.from_physical` clamps `u_lattice`; a runtime warning fires when `max|u| > 0.3`. A hard auto-pause guard is still future work. |
| Flag bitfield for walls/inlet/outlet | **Not done**: S-Stream uses a boolean solid mask. Works, but one flag array would unify inlet/outlet/moving-solid handling later. |
| `UPDATE_FIELDS` recompute rho/velocity every step | Already the case: `lbm.py` updates fields every step. |

## Citations required by the FluidX3D license

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

## Future work (ranked by value)

1. SoA (`f[q, x, y, z]`) population layout in Warp; drop the duplicate
   `feq` evaluation in the Smagorinsky branch of `_step_kernel`.
2. Device-side reduction for `max_speed` / `total_mass` and skip host
   readbacks when nothing consumes them.
3. FP16S DDF storage (`wp.float16` + per-array scale factor).
4. Esoteric pull streaming (single buffer, in-place).
5. Hard stability guard: abort + status warning when `max_speed` exceeds
   ~0.3 or density drifts from 1.
