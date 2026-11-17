# Step 4 — sphere matrix with the steady tunnel: GATE FAILED

**The run was stopped after the first four cases** because the placement
target had already failed by more than an order of magnitude.  D=20 and D=30
were not run, so there is no valid Cd-vs-D trend or D=30 acceptance number.

Setup: Re = 100, `u_lattice = 0.05`, sphere bounding-box D ≈ 10 cells,
tunnel `L = 10D`, `W = H = 6D`, sphere resting on the floor; grid
`100 x 60 x 60`; 10,000 steps for every case (the two placements at a given
D therefore have exactly equal step counts). Cd is normalised by the measured
probe-plane freestream; the tunnel uses slip sides/roof, a no-slip floor and
the fixed-density outlet.

| D | Cs | x/L | Cd | late-window std | late-window change | Cy | Cz | u_probe/U |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 10 | 0 | 0.25 | 1.3550 | 0.0001 | 0.0001 | 0.0000 | -53.7362 | 0.9914 |
| 10 | 0 | 0.75 | 0.9296 | 0.0002 | 0.0000 | 0.0000 | -53.9666 | 0.9914 |
| 10 | 0.12 | 0.25 | 1.3859 | 0.0002 | 0.0001 | -0.0000 | -53.7421 | 0.9912 |
| 10 | 0.12 | 0.75 | 0.9449 | 0.0001 | 0.0000 | 0.0000 | -53.9800 | 0.9912 |

## How the targets failed

- **Placement agreement (target < 3 % at equal steps): FAILED.**
  - Cs = 0: 1.3550 vs 0.9296 → **37.2 %** difference.
  - Cs = 0.12: 1.3859 vs 0.9449 → **37.8 %** difference.
  This is *larger* than the drift-era gap once the tunnel is made steady,
  because the late-window change is now essentially zero (≤ 0.0001) and the
  two values are genuinely different steady states, not two points on a
  decaying trace.
- **Cd → 1.087 with increasing D and within 5 % at D = 30: NOT ASSESSED.**
  D = 20 and D = 30 were deliberately not started after the first gate
  failed.  The D = 10 values bracket the reference (0.93–1.39), so no
  conclusion about the D trend can be drawn from them.
- The boundary fix itself worked: the probe freestream is 0.991 U at both
  placements and the late-window change is ~0.0001, versus the old tunnel's
  continuous 26 % drift.  What remains is a **placement/floor-boundary-layer
  effect**, not a tunnel steadiness effect: the sphere at x/L = 0.75 sits in
  a much thicker floor boundary layer (Cz ≈ -54 at both placements, because
  both touch the floor) and sees a lower local dynamic pressure than the
  cross-plane-average probe measures.  A probe plane 1.5 D ahead of the body
  averaged over the full cross-section does not correct that.

## GATE

**FAILED.** Stop here. Step 5 is not run; `docs/SOLVER_ACCEPTANCE.md` and the
`--visual` printouts are left unchanged.  Recovering the placement target
needs a different fix (e.g. a local near-floor freestream definition, a
free-slip floor, or matching the boundary-layer state at both stations),
which is outside the agreed five-step workflow.
