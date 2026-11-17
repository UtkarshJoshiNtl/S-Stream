# Step 1 — empty-tunnel boundary variants

Setup: grid `96 x 58 x 58`, `tau = 0.514369`, `Cs = 0.12`,
`u_lattice = 0.05`, 20,000 steps, state logged every 100 steps.  All state is
recomputed from the live populations (the solver's cached fields are written
before the inlet/outlet kernel overwrites the x boundary planes).

| Variant | Wall / outlet setup | Final mean density | Final mean \|u\| / U | Max drift, steps 2000–20000 |
|---|---|---:|---:|---:|
| A | no-slip walls, zero-gradient outlet | 1.047071 | 0.6641 | 26.52 % |
| B | y/z periodic (no transverse walls), zero-gradient outlet | 1.000000 | 1.0000 | 0.000 % |
| C | free-slip y/z walls, zero-gradient outlet | 1.000241 | 0.9983 | 0.150 % |
| D | no-slip walls, fixed-density outlet | 1.001198 | 0.9681 | 0.378 % |
| E | free-slip y/z walls, fixed-density outlet | 1.000001 | 1.0000 | 0.002 % |

"Max drift" is the largest change from the step-2000 value over steps
2000–20000 among mean density (%, normalised to rho=1), mean speed,
centreline u_x, and the cross-plane speed at 25/50/75 % of the tunnel length
(speeds normalised to the nominal U=0.05).  Per-metric columns are in
`tunnel_variant_summary.csv`; full traces are in `tunnel_variant_A.csv` …
`tunnel_variant_E.csv`; the time series are plotted in `tunnel_variants.png`.

## Which of walls or outlet causes the drift?

**Both matter, but the outlet is the direct mass-conservation failure and the
no-slip walls are what trigger it.**

- A → D (same no-slip walls, change only the outlet): the density rise falls
  from +4.71 % to +0.12 % and the speed drift from 24.75 % to 0.03 %.  The
  zero-gradient outlet copies the accumulated density and the decelerated
  velocity out of the domain, so mass piles up and the flow keeps slowing.
  Resetting the outlet density to 1 with the upstream velocity closes the mass
  balance.
- A → C (same zero-gradient outlet, change only the walls): drift falls from
  26.52 % to 0.15 %.  No-slip transverse walls build a growing boundary layer;
  that streamwise velocity profile is what makes the zero-gradient outlet
  non-conservative.  Specular walls keep the profile plug-like, so copying the
  neighbour is almost exact.
- E combines both fixes and is steady to 0.002 % for the whole 18,000-step
  measurement window.  B (doubly periodic, no walls at all) is also exactly
  steady, which confirms the drift is a wall/outlet artefact rather than a
  bulk time step or forcing problem.

**Winner for Step 2:** free-slip transverse walls plus the fixed-density
equilibrium outlet (variant E).  The no-slip floor will be kept for the
tunnel defaults because the validation bodies rest on it; variant D shows the
fixed-density outlet alone holds it within 0.38 % drift, and Step 2 will test
the combined side/top-free-slip, floor-no-slip configuration directly.

## GATE

Only variant A drifts; B–E are steady.  GATE passed — continue to Step 2.
