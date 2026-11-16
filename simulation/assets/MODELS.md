# Models

This directory contains the standard test shapes used by S-Stream. **Do not download whole datasets**; only the specific files recorded here are used.

## Model inventory

| Model | File | Role | Source URL | License (as stated at source) | Citation | Face count | Bounding box (x_min, x_max, y_min, y_max, z_min, z_max) [m] | Watertight |
|---|---|---|---|---|---|---:|---|:---:|
| Ahmed body (run 1) | `3d-files/ahmed_1.stl` | **Validation** | https://huggingface.co/datasets/neashton/ahmedml/resolve/main/run_1/ahmed_1.stl (dataset: https://huggingface.co/datasets/neashton/ahmedml) | `cc-by-sa-4.0` (CC BY-SA 4.0), per dataset card | Ashton, N., Maddix, D., Gundry, S., Shabestari, P. (2024). *AhmedML: High-Fidelity Computational Fluid Dynamics dataset for incompressible, low-speed bluff body aerodynamics*. arXiv:2407.20801. https://arxiv.org/abs/2407.20801 | 149,948 | (-1.1879923343658447, 1.1920928955078125e-07, -0.18489953875541687, 0.18494421243667603, 0.04999375343322754, 0.30339083075523376) | Yes |
| DrivAerML notchback (run 1) | `3d-files/drivaer_1.stl` | **Demo** | https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_1/drivaer_1.stl (dataset: https://huggingface.co/datasets/neashton/drivaerml) | `cc-by-sa-4.0` (CC BY-SA 4.0), per dataset card | Ashton, N. et al. (2024). *DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics*. arXiv:2408.11969. https://arxiv.org/abs/2408.11969 | 753,234 | (-0.7651960253715515, 3.9384799003601074, -1.0102499723434448, 1.0102499723434448, -0.31964099407196045, 1.1951099634170532) | Yes |
| Sphere (analytic) | `3d-files/sphere.stl` | **Validation** | Generated locally (no download) by `simulation/scripts/generate_shapes.py` | N/A (generated analytic geometry, not a third-party asset) | N/A | 28,320 | (-0.4999564290046692, 0.4999564290046692, -0.4999564290046692, 0.4999564290046692, -0.5, 0.5) | Yes |
| Cylinder (analytic) | `3d-files/cylinder.stl` | **Validation** | Generated locally (no download) by `simulation/scripts/generate_shapes.py` | N/A (generated analytic geometry, not a third-party asset) | N/A | 476 | (-0.25, 0.25, -0.25, 0.25, -0.5, 0.5) | Yes |
| Ahmed-style box (analytic) | `3d-files/ahmed_box.stl` | **Validation** | Generated locally (no download) by `simulation/scripts/generate_shapes.py` (plain rectangular prism using reference Ahmed body dimensions 1.044 × 0.389 × 0.288 m) | N/A (generated analytic geometry, not a third-party asset) | N/A | 12 | (-0.5220000147819519, 0.5220000147819519, -0.19449999928474426, 0.19449999928474426, -0.14399999380111694, 0.14399999380111694) | Yes |

## Notes

- **Validation shapes:** Ahmed body, sphere, cylinder, and Ahmed-style box (standard/reference shapes suitable for validation checks).
- **Demo shapes:** DrivAerML notchback (complex production-like geometry for demonstration).
- Ahmed and DrivAer meshes were downloaded individually (one file each) from the HuggingFace datasets above on 2026-11-16; dataset licenses and citations are recorded from the source dataset cards at that time.
- Face counts, bounding boxes, and watertight status were measured locally with PyVista/VTK after download/generation (boundary edge count = 0 reported as watertight).
