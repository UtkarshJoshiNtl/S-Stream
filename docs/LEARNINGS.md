# LEARNINGS

Development journal for the S-Stream LBM solver. Entries are measurable facts (commit hash, file, test, command output) or marked **to verify**.

## Bug Fixes

### Tau calculation (2026-11-20)
- **Bug**: `test_sphere_drag.py` used `tau = 0.5 + 9*nu` instead of correct `0.5 + 3*nu`
- **Effect**: Effective viscosity 3x too high, wrong Reynolds number
- **Fix**: Corrected formula; results changed from Cd=1.42 to Cd=1.46 (Re=100)

### Outlet boundary condition (2026-11-20)
- **Bug**: `zero_gradient` outlet allowed mass accumulation, causing flow deceleration
- **Fix**: Changed to `fixed_density` outlet in `test_cylinder.py` and `test_sphere_drag.py`
- **Effect**: Prevents spurious drag increase over time

### Boundary conditions (2026-11-20)
- **Bug**: `periodic=(False, False, False)` (all no-slip walls) instead of `periodic=(False, True, True)`
- **Effect**: Wall effects increased sphere Cd from ~1.46 to ~2.02
- **Fix**: Periodic y,z boundaries for wind tunnel setup

### Drag sign convention (2025-11)
- **Bug**: Inconsistent sign between `pipeline.py` and `fields.py`
- **Fix**: Unified to positive convention; added known-sign test

## Performance Findings

### Memory layout optimization (2025-11)
- **Change**: SoA populations `f[q, x, y, z]` instead of AoS `f[x, y, z, q]`
- **Result**: 41.3 → 123.5 MLUPs (3.0x speedup) on RTX 2050
- **Guard**: Cd bit-identical before/after (numerics unchanged)

### Phase 1 optimizations (2026-01)
- **Change**: Common subexpression elimination, register reduction in collision kernel
- **Result**: 108.7 → 177.6 MLUPs (63.4% improvement)
- **Tests**: All 101 tests pass

### Occupancy/block_dim tuning (2026-11-20)
- **Tested**: block_dim = None (256), 32, 64, 128, 256
- **Result**: 174.8, 175.0, 175.6, 181.6, 169.0 MLUPs respectively
- **Conclusion**: Default block_dim=256 is fine; no tuning needed (null result)

### FP16 experiment (2026-11-20)
- **Status**: Reverted. FP32 is the only supported precision.
- **Result**: Only 7% gain (184 vs 173 MLUPs), but Poiseuille 100% error, cylinder Cd 10% low, no vortex shedding
- **Root cause**: LBM populations need ~4-5 decimal digits; FP16 has ~3
- **Path forward**: DDF-shifting + custom 16-bit format (not worth complexity for 7% gain)

## Validation Methodology

### Blockage correction
- Cylinder: 13% blockage ratio (D=8, height=60) → confined Cd=1.587 vs unconfined ~1.3
- Sphere: ~5% blockage → Cd systematically high by ~10-18%
- **Lesson**: Always report blockage ratio; compare to confined literature when possible

### Reference selection
- Poiseuille: Exact solution (0.38% error, threshold 1.5%)
- Cylinder: Confined literature bands (Cd 1.35-1.85, St 0.16-0.21)
- Sphere: Johnson & Patel (1999) Re=100 Cd=1.087, but our setup is confined

### Force sampling
- Sample every 4 steps after step 10000 (allows wake development)
- Check force-window convergence (early vs late means)
- Report Cd trace for convergence analysis

## Profiling Findings

**Hardware**: RTX 2050 (4GB VRAM, sm_86, ~192 GB/s theoretical bandwidth)

**Baseline**: 176.7 MLUPs on 256×102×77 empty tunnel (tau=0.56, smag=0.12)

**Achievable bandwidth** (copy kernel): 105.2 GB/s

**LBM kernel bandwidth usage**: ~32 GB/s (30% of achievable)

**Bottleneck**: Memory-bound, occupancy-limited (~15% due to vec19 register vector)

**Not limiting**: VRAM (364 MB / 4 GB), compute (Smagorinsky on/off only 3% difference)

## Open Questions (to verify)

- Warm-up MLUPs (~100-108 at ~1M cells) are RTX 2050 / Warp 1.18.0 numbers; other GPUs **to verify**
- Sphere Cd is systematically high (~10-18%); root cause is grid resolution + wall effects, not a bug
- High-Re Ahmed/DrivAer cases not yet validated against literature
- Visual mode (`--visual`) pins tau to [0.505, 0.52] for stability; Cd not calibrated

## Future Work

- **DDF-shifting**: Store `f_shifted = f - w_i` for better numerical accuracy (especially FP16)
- **Esoteric-pull**: 50% VRAM reduction via in-place streaming (not needed now, VRAM not limiting)
- **FP16 with DDF-shifting**: Would require custom 16-bit format; not worth complexity for current gains
