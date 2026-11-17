# Diagnosis: why the two Re=100 sphere tests disagree on Cd

**Status: diagnosis only.** No solver, test, rendering, or pipeline code was changed.
All new code is in `simulation/scripts/diagnostics/`; all generated data is in
`docs/diag_data/`. The reference value used throughout is
Johnson & Patel (1999), Cd = 1.087 at Re = 100 (see `docs/REFERENCES.md`, where
the exact figure is flagged `to verify`). No blockage correction is applied;
the 2.2% silhouette blockage is recorded instead.

The runner used for the controlled cases calls `LbmSolver` directly and samples
the wall force on every `steps // 400` steps (≈400 samples per run), instead of
the ≈160 samples used by `pipeline.run_case` in pytest. It reproduces the pytest
window means (0.9083 for test 1 and 1.2509 for test 2 at 8,000 steps, versus the
tests' 0.908 and 1.251), so the denser sampling is not itself the explanation.

---

## 1. Setup audit

`docs/diag_data/setup_audit.csv`. Side by side for the two sphere tests (with
the 2x-outlet variant and the cylinder for context):

| quantity | test_sphere_re100 (x/L=0.75) | test_outlet_reflection 1x (x/L=0.25) | outlet 2x variant | cylinder |
|---|---|---|---|---|
| grid (nx, ny, nz) | (96, 58, 58) | (96, 58, 58) | (144, 58, 58) | (80, 60, 4) |
| dx (m) | 0.0417 | 0.0417 | 0.0417 | 1.0000 |
| sphere D bbox (cells) | 9.5795 | 9.5795 | 9.5795 | 8.0000 |
| u_lattice | 0.0500 | 0.0500 | 0.0500 | 0.0500 |
| tau (molecular) | 0.514369 | 0.514369 | 0.514369 | 0.5120 |
| Smagorinsky Cs | 0.12 | 0.12 | 0.12 | 0 |
| tunnel L×W×H | 4×2.4×2.4 m | 4×2.4×2.4 m | 6×2.4×2.4 m | 80×60×4 cells |
| silhouette blockage | 2.20% | 2.20% | 2.20% | 13.3% |
| inlet speed SI (m/s) | 0.0500 | 0.0500 | 0.0500 | 0.0500 |
| viscosity SI (m²/s) | 1.9957e-4 | 1.9957e-4 | 1.9957e-4 | 0.0040 |
| steps | 12,000 | 8,000 | 8,000 | 30,000 |
| force-averaging window | step 6,000–11,994 (163 samples) | step 4,000–8,000 (161 samples) | step 4,000–8,000 | step 10,000–30,000 |
| body x/L | 0.750 | 0.250 | 0.167 | 0.280 |
| upstream fetch (D) | 7.016 | 2.005 | 2.005 | 2.300 |
| downstream fetch (D) | 2.005 | 7.016 | 12.027 | 6.700 |
| solid cells | 464 | 464 | 464 | 200 |

**Re agrees exactly.** Re = u_lattice·D_cells/ν_lattice = 0.05×9.5795/0.004790
= 100.000 in both sphere tests (and in the 2x variant); using the bbox x-extent
gives 99.893 in both. The SI speed/diameter/viscosity calculation also gives
100.000. Re is not the cause of the disagreement.

The *setup* is not identical, however: the grid, tau, Cs, diameter, blockage,
inlet speed and viscosity are the same, but the streamwise **placement** and the
**number of steps** differ.

---

## 2. Geometry audit

`docs/diag_data/geometry_audit.csv`. The two pytest cases use byte-identical
voxel masks (same 464 solid cells, same 73-cell silhouette):

| case | solid cells | analytic volume (cells) | solid/analytic | effective D / bbox D | silhouette ratio |
|---|---|---|---|---|---|
| both pytest sphere cases | 464 | 460.29 | 1.0081 | 1.00268 | 1.0129 |
| sweep D=10 | 552 | 520.26 | 1.0610 | 1.01994 | 1.0230 |
| sweep D=20 | 4,224 | 4,162.06 | 1.0149 | 1.00494 | 1.0102 |
| sweep D=30 | 14,328 | 14,046.96 | 1.0200 | 1.00663 | 1.0173 |
| sweep D=40 (not run) | 33,552 | 33,296.49 | 1.0077 | 1.00255 | 1.0102 |

The voxelisation is identical for the two tests; at D=10 it over-fills by 6.1%
(effective D +2.0%), while D=20/30/40 are within 0.8–2.0% volume and ≤0.7%
effective diameter. Geometry is not the cause of the test-to-test disagreement,
although the D=10 staircase is worth remembering when reading the resolution
trend.

---

## 3. Controlled runs

`docs/diag_data/sweep_summary.csv`, with per-step traces in
`sweep_<key>.csv` / `sweep_<key>.png`. Controlled proportions for the base sweep
are L=10D, W=H=6D, body at x=2.5D, Re=100, u_lattice=0.05, τ>0.5001 in every
case; Cs and D are varied one at a time. The "exact" cases use the pytest STL
mask and grid so the pytest windows are reproduced exactly.

| key | group | status | grid | D bbox | Cs | tau | Re | blockage | steps | window steps | Cd (pytest window) | Cd (last window) | std | change | Cy | Cz | Lr/D | u_ref/u_nom | u_c/u_ref | q_c/q_nom | delta*/D |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| exact_x3p0 | pytest geometry | ok | (96, 58, 58) | 9.5795 | 0.12 | 0.51437 | 100.0000 | 0.0220 | 12,000 | 2,371 | 0.9083 | 0.8636 | 0.0162 | 0.06315 | -0.00 | -52.38 | 1.1483 | 0.7648 | 0.7920 | 0.3669 | 0.3038 |
| exact_x1p0_L4 | pytest geometry | ok | (96, 58, 58) | 9.5795 | 0.12 | 0.51437 | 100.0000 | 0.0220 | 12,000 | 2,371 | 1.1546 | 1.0990 | 0.0202 | 0.06185 | -0.00 | -52.26 | 0.7307 | 0.7618 | 1.0915 | 0.6914 | 0.1282 |
| exact_x1p0_L6 | pytest geometry | ok | (144, 58, 58) | 9.5795 | 0.12 | 0.51437 | 100.0000 | 0.0220 | 12,000 | 2,371 | 1.1748 | 1.1246 | 0.0184 | 0.05510 | -0.00 | -52.16 | 0.7307 | 0.7742 | 1.0907 | 0.7131 | 0.1276 |
| exact_x3p0_8000 | pytest timing | ok | (96, 58, 58) | 9.5795 | 0.12 | 0.51437 | 100.0000 | 0.0220 | 8,000 | 1,581 | 0.9859 | 0.9528 | 0.0122 | 0.04346 | -0.00 | -51.95 | 1.2527 | 0.8210 | 0.8025 | 0.4341 | 0.2997 |
| exact_x1p0_L4_8000 | pytest timing | ok | (96, 58, 58) | 9.5795 | 0.12 | 0.51437 | 100.0000 | 0.0220 | 8,000 | 1,581 | 1.2509 | 1.2099 | 0.0152 | 0.04245 | -0.01 | -51.82 | 0.8351 | 0.8170 | 1.0880 | 0.7901 | 0.1258 |
| sweep_D10_Cs0 | base sweep | ok | (100, 60, 60) | 9.9787 | 0.00 | 0.51497 | 100.0000 | 0.0222 | 10,000 | 1,976 | 1.1795 | 1.1337 | 0.0169 | 0.05011 | 3.95e-07 | -54.26 | 0.8017 | 0.7921 | 1.0881 | 0.7427 | 0.1306 |
| sweep_D10_Cs0.12 | base sweep | ok | (100, 60, 60) | 9.9787 | 0.12 | 0.51497 | 100.0000 | 0.0222 | 10,000 | 1,976 | 1.2003 | 1.1527 | 0.0175 | 0.05130 | 6.40e-06 | -54.28 | 0.8017 | 0.7895 | 1.0918 | 0.7430 | 0.1335 |
| sweep_D20_Cs0 | base sweep | ok | (200, 120, 120) | 19.9574 | 0.00 | 0.52994 | 100.0000 | 0.0219 | 20,000 | 3,951 | 1.1207 | 1.0766 | 0.0163 | 0.05089 | 1.15e-06 | -27.29 | 0.6514 | 0.7930 | 1.0832 | 0.7379 | 0.1231 |
| sweep_D20_Cs0.12 | base sweep | ok | (200, 120, 120) | 19.9574 | 0.12 | 0.52994 | 100.0000 | 0.0219 | 20,000 | 3,951 | 1.1274 | 1.0826 | 0.0165 | 0.05131 | -4.35e-06 | -27.29 | 0.6514 | 0.7923 | 1.0848 | 0.7387 | 0.1241 |
| sweep_D30_Cs0 | base sweep | ok | (300, 180, 180) | 29.9361 | 0.00 | 0.54490 | 100.0000 | 0.0221 | 30,000 | 5,926 | 1.0955 | 1.0517 | 0.0161 | 0.05163 | -1.89e-06 | -19.46 | 0.6013 | 0.7920 | 1.0934 | 0.7498 | 0.1215 |
| sweep_D30_Cs0.12 | base sweep | ok | (300, 180, 180) | 29.9361 | 0.12 | 0.54490 | 100.0000 | 0.0221 | 30,000 | 5,926 | 1.0986 | 1.0545 | 0.0162 | 0.05188 | -3.60e-06 | -19.46 | 0.6013 | 0.7916 | 1.0941 | 0.7502 | 0.1219 |
| sweep_D40_Cs0 | base sweep | impossible: solver arrays exceed 90% of device memory | (400, 240, 240) | 39.9147 | n/a | 0.55987 | 100.0000 | n/a | - | - | - | - | - | - | - | - | - | - | - | - | - |
| sweep_D40_Cs0.12 | base sweep | impossible: solver arrays exceed 90% of device memory | (400, 240, 240) | 39.9147 | n/a | 0.55987 | 100.0000 | n/a | - | - | - | - | - | - | - | - | - | - | - | - | - |
| place_D10_x7p5 | placement | ok | (100, 60, 60) | 9.9787 | 0.12 | 0.51497 | 100.0000 | 0.0222 | 12,000 | 2,371 | 0.9100 | 0.8660 | 0.0160 | 0.06219 | -2.07e-06 | -54.61 | 1.2026 | 0.7607 | 0.7562 | 0.3309 | 0.3077 |
| place_D20_x7p5 | placement | ok | (200, 120, 120) | 19.9574 | 0.12 | 0.52994 | 100.0000 | 0.0219 | 20,000 | 3,951 | 0.9007 | 0.8646 | 0.0133 | 0.05179 | -1.12e-06 | -27.41 | 0.9520 | 0.7920 | 0.8974 | 0.5052 | 0.2919 |
| place_D10_x7p5_L20 | placement | ok | (200, 60, 60) | 9.9787 | 0.12 | 0.51497 | 100.0000 | 0.0222 | 20,000 | 3,951 | 0.8633 | 0.8091 | 0.0194 | 0.08000 | -4.29e-06 | -55.08 | 0.8017 | 0.7016 | 0.7504 | 0.2772 | 0.3091 |
| place_D10_x2p5_L20 | placement | ok | (200, 60, 60) | 9.9787 | 0.12 | 0.51497 | 100.0000 | 0.0222 | 20,000 | 3,951 | 1.0497 | 0.9850 | 0.0232 | 0.07862 | -5.53e-06 | -55.01 | 0.7015 | 0.7029 | 1.0928 | 0.5901 | 0.1374 |
| place_D10_x12p5_L20 | placement | ok | (200, 60, 60) | 9.9787 | 0.12 | 0.51497 | 100.0000 | 0.0222 | 20,000 | 3,951 | 0.8125 | 0.7621 | 0.0181 | 0.07919 | -2.21e-06 | -55.07 | 0.8017 | 0.6991 | 0.6716 | 0.2205 | 0.3481 |
| free_D20_Cs0.12 | free-sphere control | ok | (200, 120, 120) | 19.9574 | 0.12 | 0.52994 | 100.0000 | 0.0219 | 20,000 | 3,951 | 1.1486 | 1.1070 | 0.0154 | 0.04696 | 2.95e-06 | -1.67e-06 | 0.6013 | 0.7908 | 0.9802 | 0.6008 | 0.1030 |

### Common-time comparison

The pytest windows end at different physical times: t* = u·t/D ≈ 62.6 at
12,000 steps and ≈41.8 at 8,000 steps. Reducing each trace to the common
window t* = 50–60 (D=9.5795) removes the run-length difference:

| case | steps | Cd (pytest window) | Cd (t*=50–60) |
|---|---|---|---|
| exact x=3.0 (test 1) | 12,000 | 0.9083 | 0.8699 |
| exact x=1.0 (test 2 geometry) | 12,000 | 1.1546 | 1.1068 |
| exact x=3.0, 8,000 steps | 8,000 | 0.9859 | — (ends before t*=50) |
| exact x=1.0, 8,000 steps | 8,000 | 1.2509 | — (ends before t*=50) |

At the **same 8,000 steps** the two placements give 0.9859 vs 1.2509 (Δ=0.265);
at the **same 12,000 steps** they give 0.9083 vs 1.1546 (Δ=0.246). Increasing
test 1's run length from 8,000 to 12,000 steps lowers its Cd by 0.078. Both
placement and elapsed time matter, and they act in the same direction for the
reported pair (test 1 is both farther downstream of the inlet and run longer).

For the D-sweep the common window is t*=40–50 (each run ends at t*≈50). At
Cs=0.12 the mean Cd falls 1.153 (D=10) → 1.083 (D=20) → 1.055 (D=30); the
Cs=0 series gives 1.134 → 1.077 → 1.052. The finest case is already slightly
below the 1.087 reference, but it is still a floor-contacting sphere in a
decaying tunnel, so this is a trend, not a calibration result.

### Placement and outlet controls

- At fixed x=1.0D and 12,000 steps, lengthening the outlet from 4D to 6D raises
  Cd from 1.1546 to 1.1748. Outlet proximity therefore cannot explain the low
  Cd of test 1; if anything, a nearer outlet lowers Cd slightly here.
- At fixed D=10, L=20D and 20,000 steps, moving the sphere from x=2.5D to 7.5D
  to 12.5D lowers Cd monotonically 1.0497 → 0.8633 → 0.8125 while δ*/D grows
  0.137 → 0.309 → 0.348 and q/q_nominal falls 0.590 → 0.277 → 0.221.
- At D=20 the same 2.5D → 7.5D move at 20,000 steps lowers Cd 1.1274 → 0.9007.
  The fetch effect is large and persists at both resolutions.
- Extending the outlet at x=7.5D from L=10D to L=20D lowers Cd 0.9100 → 0.8633,
  but that control also doubles the step count (12k → 20k), and the empty-tunnel
  control shows exactly this kind of downward drift; it is not evidence of an
  outlet-reflection mechanism.

---

## 4. The tunnel never reaches steady state

`docs/diag_data/empty_tunnel_drift.csv` / `.png`. This control uses the *empty*
grid with the same tau, Cs, inlet speed and step count as the pytest cases:

| step | mean rho | mean u | flux at inlet x=0 (populations) | flux at mid tunnel | q/q_nominal at mid |
|---:|---:|---:|---:|---:|---:|
| 200 | 1.00249 | 0.04649 | 0.0500 | 0.04761 | 0.904 |
| 1,000 | 1.00621 | 0.04556 | 0.0500 | 0.04685 | 0.873 |
| 4,000 | 1.01433 | 0.04284 | 0.0500 | 0.04445 | 0.779 |
| 8,000 | 1.02359 | 0.03978 | 0.0500 | 0.04168 | 0.679 |
| 12,000 | 1.03208 | 0.03700 | 0.0500 | 0.03913 | 0.593 |

The velocity-inlet / zero-gradient-outlet tunnel has no mass sink, so the mean
density rises monotonically and the entire flow decelerates for the whole run.
The inlet populations keep imposing `sum c_x f = 0.05`, but the actual
streamwise mass flux at x=1 has fallen to 0.0391 by step 12,000. Because Cd is
normalised by the **nominal** `u_lattice = 0.05`, the reported Cd decays with
time even for a fixed geometry. This is visible in every trace: the x=3.0 case
goes 0.9859 (8k window) → 0.9083 (12k window) → 0.8636 (final 20% window).

The local free-stream used in the boundary-layer metrics below is therefore the
velocity at the top of the measurement plane, not the nominal inlet value.

---

## 5. Floor boundary layer and placement

The sphere in both tests touches the no-slip floor (its lowest solid cells are
at z=0). One diameter upstream of the sphere front:

| case (12,000 steps) | u_ref/u_nominal | u_centre/u_ref | q_centre/q_nominal | δ*/D |
|---|---|---|---|---|
| exact x=3.0 (test 1, 7.0D upstream fetch) | 0.765 | 0.792 | 0.367 | 0.304 |
| exact x=1.0 (test 2, 2.0D upstream fetch) | 0.762 | 1.092 | 0.691 | 0.128 |
| x=12.5D, L=20D, D=10, 20,000 steps | 0.699 | 0.672 | 0.221 | 0.348 |
| x=2.5D, L=20D, D=10, 20,000 steps | 0.703 | 1.093 | 0.590 | 0.137 |

At the sphere centre height the local dynamic pressure is only ~37% of the
nominal inlet value in test 1, versus ~69% in test 2. The `u_centre/u_ref`
values above 1 at x=1.0D/2.5D are a local inlet-development effect at the
measurement plane two diameters upstream of the front; the robust comparison
is the displacement thickness δ*/D and q_centre/q_nominal. The longer upstream
fetch grows a floor boundary layer roughly 2.4× thicker (0.304D vs 0.128D),
so the sphere in test 1 is immersed in much stronger shear and a much lower
effective free stream, while Cd is still normalised by the nominal speed.
This is the main reason its Cd is lower. The L=20D placement sweep makes the
same point monotonically: x=2.5D → 7.5D → 12.5D gives Cd 1.050 → 0.863 →
0.813 and q/q_nominal 0.590 → 0.277 → 0.221.

A related artifact: the vertical force coefficient is **Cz ≈ −52** in the
floor-contact pytest cases (and −54 at D=10, −27 at D=20, −19 at D=30; it falls
with resolution but is still large). The D=20 control with the sphere centred
in Z gives **Cz ≈ 0** and Cd 1.149, confirming that the floor-contact momentum
exchange is not a physically clean wall force. The huge Cz is common to both
pytest cases, so it does not explain their *difference*, but it makes both
absolute Cd values suspect. Note that the free sphere still has Cd 1.149 after
20,000 steps, higher than the floor-contact D=20 value of 1.127: lifting the
sphere changes both the wall artifact and the local inflow, so it is not a
simple additive correction.

---

## 6. Independent force check

`docs/diag_data/force_balance.csv`, produced by
`force_balance.py` from the saved final velocity/density/f-population crops of
the finest run (D=30, Cs=0.12), the free-sphere control and the two exact pytest
cases. The check
integrates steady x-momentum over a box containing the sphere,
F_body = ∮[ρu(u·n) + p n − τ·n] dS, with the floor shear taken from the actual
bounce-back populations. Cd = −F_body·2/(u²A). A "tight" box ending at the rear
of the sphere is also reported because the wide box's outlet-side face is
clipped to nx−2 and sits in a distorted gradient region.

| case | floor | Cd (momentum exchange) | Cd CV wide | residual | Cd CV tight | residual | Cd pressure-only |
|---|---|---:|---:|---:|---:|---:|---:|
| D=30, Cs=0.12 (finest) | yes | 1.0277 | 0.6088 | −40.8% | 0.8332 | −18.9% | 0.6169 |
| D=20 free sphere | no | 1.0809 | 0.3626 | −66.4% | 1.5182 | +40.5% | 0.5393 |
| test_sphere_re100 crop | yes | 0.8367 | −1.6235 | −294% | 1.3421 | +60.4% | 0.4851 |
| test_outlet_reflection crop | yes | 1.0650 | 0.5441 | −48.9% | 1.1547 | +8.4% | 0.6694 |

The momentum-exchange and control-volume estimates disagree by 8% to 294%
depending on the case and box, and the wide/tight spread is equally large.
The reason is visible in the terms: the inlet and outlet pressure integrals are
≈±343 lattice force
units and cancel to leave ≈0.5, while the tunnel is still draining (a large
unsteady term that the steady balance omits) and the velocity is cell-centred
on a staircase body. **The CV check therefore does not independently confirm a
precise Cd for this setup; it confirms the order of magnitude and documents the
conditioning problem.** The pressure-only integral over the sphere surface is
about 50–63% of the momentum-exchange drag, which is a plausible pressure-drag
share at Re=100, but it is not an independent force law.

---

## 7. Why the cylinder test accepts Cd ≈ 1.587

`docs/diag_data/cylinder_summary.csv`. The cylinder case is a different
configuration: D=8 cells, ny=60 (13.3% periodic blockage), Cs=0, and periodic
side walls. Refining the mesh and reducing blockage moves Cd steadily toward
the classical value:

| D (cells) | ny | blockage | Cd | St |
|---:|---:|---:|---:|---:|
| 8 | 60 | 13.3% | 1.587 | 0.184 |
| 8 | 240 | 3.3% | 1.558 | 0.184 |
| 16 | 120 | 13.3% | 1.485 | 0.176 |
| 16 | 240 | 6.7% | 1.386 | 0.160 |
| 16 | 480 | 3.3% | 1.349 | 0.144 |
| 24 | 180 | 13.3% | 1.436 | (wake nearly steady) |
| 24 | 360 | 6.7% | 1.338 | (wake nearly steady) |

The accepted 1.587 is the **coarsest, most confined** member of this family.
Confinement and under-resolution both raise Cd; the test's 1.35–1.85 band is
deliberately wide enough to accept that coarse/confined value as well as the
finer, less confined ones. It is a vortex-shedding/stability regression check,
not a high-accuracy drag validation, and it is not comparable with the 2.2%
blockage sphere benchmark.

---

## 8. Explicit answer

**The two sphere tests do not disagree because of Re or geometry**: both are
Re=100 with the same 464-cell sphere, the same grid, tau, Cs and blockage.
`test_sphere_re100` reports 0.908 because its sphere sits at x/L=0.75 after
7.0D of upstream floor boundary-layer growth (local q/q_nominal ≈ 0.37) and
because its 12,000-step run samples a tunnel that has already decelerated by
~22% from the nominal inlet speed. `test_outlet_reflection` reports 1.25
because its sphere sits only 2.0D downstream of the inlet (local q/q_nominal ≈
0.69) and is averaged at 8,000 steps, earlier in the same decay. Controlled
runs isolate both effects: at equal 8,000 steps, placement alone gives 0.9859
vs 1.2509; at equal 12,000 steps it gives 0.9083 vs 1.1546. Moving the outlet farther
away at x=1.0D slightly *increases* Cd (1.154 → 1.175 with L=6), so outlet
reflection is not the low-Cd cause. The cylinder's 1.587 is accepted because
that test's deliberately wide band covers a coarse, 13.3%-confined case; the
same setup refined and unconfined tends to 1.35–1.39.

---

## 9. Ranked likely causes

1. **Upstream floor boundary layer / inlet shear (dominant).** Evidence:
   placement sweep at equal steps (0.9083 vs 1.1546 at 12k; 0.9859 vs 1.2509
   at 8k), δ*/D = 0.304 vs 0.128, centre-height q/q_nominal = 0.367 vs 0.691.
   The L=20D sweep gives 1.050 → 0.863 → 0.813 for x=2.5D → 7.5D → 12.5D.
   The sphere is not in a uniform free stream.
2. **Unsteady tunnel drift and nominal-speed normalisation (major).** Evidence:
   empty-tunnel control rho 1→1.032, u 0.05→0.037, flux 0.05→0.039,
   q/q_nominal 1→0.59 over 12,000 steps; every Cd trace falls with time.
3. **Different run length / force window (contributes to 2).** Test 1 averages
   6,000–12,000, test 2 averages 4,000–8,000; the same case loses ~0.08 Cd
   between the 8k and 12k windows and another ~0.04 by the final window.
4. **Floor-contact artifact (not the difference, but limits absolute trust).**
   Cz ≈ −52 in the pytest floor cases (−54/−27/−19 at D=10/20/30) versus ≈0
   for the D=20 centred-sphere control (Cd 1.149). The test pair shares the
   artifact, so it cannot produce their difference, but neither absolute Cd
   is a clean external-flow measurement.
5. **Outlet proximity (ruled out as the cause).** Lengthening the tunnel at
   x=1.0D raises Cd 1.154→1.175; lengthening at x=7.5D lowers it 0.910→0.863,
   in line with the extra drift from 12k→20k steps, not with reflection.
6. **Subgrid model and resolution (secondary here).** At D=10, Cs=0 vs 0.12
   changes Cd by only ≈0.02 (1.180 vs 1.200 at the pytest-style window) and at
   D=30 by ≈0.003. At common t*=40–50 the Cs=0.12 series falls 1.153 (D=10) →
   1.083 (D=20) → 1.055 (D=30), i.e. refinement lowers Cd toward, and then
   slightly below, the 1.087 reference while the floor artifact and tunnel
   drift remain. Neither Cs nor resolution can account for a 0.25–0.34 gap
   between the two tests.
7. **Re / voxel geometry (ruled out).** Re is exactly 100 in both cases and the
   masks are identical (Section 2).

---

## 10. Impossible / guarded cases

- **D=40, Cs=0 and D=40, Cs=0.12**: requested grid (400, 240, 240) at the same
  proportions. The solver arrays alone need `cells × 45 × 4 bytes` =
  23,040,000 × 45 × 4 = 4.147 GB (3.86 GiB), which exceeds the script's
  90%-of-device-memory guard on the RTX 2050 4 GiB card (3.87 GB usable).
  Both cases are recorded in
  `sweep_summary.csv` with status `impossible: solver arrays exceed 90% of
  device memory` and were **not** silently reduced to a smaller diameter.
  The CUDA context and Warp workspace consume additional memory, so even the
  raw 3.86 GiB estimate leaves no practical headroom.
- **tau condition**: `tau = 0.5 + u·D/(Re·c_s²)` evaluated to 0.51437 (D=9.58),
  0.51497 (D=10), 0.52994 (D=20), 0.54490 (D=30) and 0.55987 (D=40) — all
  > 0.5001, so no case was rejected on that criterion.

---

## Appendix: generated data

- `setup_audit.csv`, `geometry_audit.csv`, `cylinder_summary.csv`
- `sweep_summary.csv`, `sweep_summary.json`, `sweep_<key>.csv`, `sweep_<key>.png`
- `common_time_cd.csv` (mean Cd in common t* = u·t/D windows)
- `empty_tunnel_drift.csv`, `empty_tunnel_drift.png`
- `force_balance.csv`
- `cylinder_<case>.csv`, `cylinder_<case>.png`
