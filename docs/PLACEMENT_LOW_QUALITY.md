# Placement report (--quality low)

Each shape was run through the standard placement pipeline and voxelized at `--quality low` (target ~1,000,000 cells). Tunnel sizing follows the CLI: a temporary placement determines model length, then `TunnelSpec.from_model_length(...)` is resolved to the largest whole grid fitting 1,000,000 cells.

Orientation: **Ahmed body** and **DrivAerML notchback** use vehicle mode (`orient=auto-vehicle`), as requested. Analytic shapes are reported with the orientations shown below.

| Shape | Orientation | Placed extents (x, y, z) [m] | Bbox blockage | Solid cells | Grid (nx, ny, nz) | Total cells | Notes |
|---|---|---:|---:|---:|---:|---:|---|
| Ahmed body (`ahmed_1.stl`) | `auto-vehicle` | 0.960000 × 0.298459 × 0.204768 | 2.829% | 5,262 | 220 × 82 × 55 | 992,200 | Voxelization: mesh not watertight after closing small holes; body treated as solid. |
| DrivAerML notchback (`drivaer_1.stl`) | `auto-vehicle` | 0.960000 × 0.412505 × 0.307559 | 5.874% | 5,748 | 220 × 82 × 55 | 992,200 | Voxelization: mesh not watertight after closing small holes; body treated as solid; cells unreachable from outside filled. |
| Sphere (`sphere.stl`) | `none` | 0.953871 × 0.953871 × 0.956931 | 42.530% | 44,638 | 220 × 82 × 55 | 992,200 | Voxelization: mesh had open edges; small holes were filled. |
| Cylinder (`cylinder.stl`) | `none` | 0.240000 × 0.240000 × 0.480000 | 21.333% | 16,764 | 220 × 82 × 55 | 992,200 | — |
| Ahmed-style box (`ahmed_box.stl`) | `auto-vehicle` | 0.960000 × 0.357701 × 0.264828 | 4.386% | 8,640 | 220 × 82 × 55 | 992,200 | — |

*Measured on 2026-11-16 with `load_surface` (no decimation), `build_model`, and `voxelize` at the computed low-quality grid.*
