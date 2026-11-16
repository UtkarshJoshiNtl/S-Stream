# Models

This directory contains the standard test shapes used by S-Stream. **Do not download whole datasets**; only the specific files recorded here are used.

## Model inventory

| Model | File | Role | Source URL | License (as stated at source) | Citation | Face count | Bounding box (x_min, x_max, y_min, y_max, z_min, z_max) [m] | Watertight |
|---|---|---|---|---|---|---:|---|:---:|
| Ahmed body (run 1) | `3d-files/ahmed_1.stl` | **Validation** | https://huggingface.co/datasets/neashton/ahmedml/resolve/main/run_1/ahmed_1.stl (dataset: https://huggingface.co/datasets/neashton/ahmedml) | `cc-by-sa-4.0` (CC BY-SA 4.0), per dataset card | Ashton, N., Maddix, D., Gundry, S., Shabestari, P. (2024). *AhmedML: High-Fidelity Computational Fluid Dynamics dataset for incompressible, low-speed bluff body aerodynamics*. arXiv:2407.20801. https://arxiv.org/abs/2407.20801 | 149,175 | (-1.1879923343658447, 1.1920928955078125e-07, -0.18489953875541687, 0.18494421243667603, 0.04999375343322754, 0.30339083075523376) | No (461 open boundary edges) |
| DrivAerML notchback (run 1) | `3d-files/drivaer_1.stl` | **Demo** | https://huggingface.co/datasets/neashton/drivaerml/resolve/main/run_1/drivaer_1.stl (dataset: https://huggingface.co/datasets/neashton/drivaerml) | `cc-by-sa-4.0` (CC BY-SA 4.0), per dataset card | Ashton, N. et al. (2024). *DrivAerML: High-Fidelity Computational Fluid Dynamics Dataset for Road-Car External Aerodynamics*. arXiv:2408.11969. https://arxiv.org/abs/2408.11969 | 747,927 | (-0.7651960253715515, 3.9384799003601074, -1.0102499723434448, 1.0102499723434448, -0.31964099407196045, 1.1951099634170532) | No (817 open boundary edges) |
| Sphere (analytic) | `3d-files/sphere.stl` | **Validation** | Generated locally (no download) by `simulation/scripts/generate_shapes.py` | N/A (generated analytic geometry, not a third-party asset) | N/A | 27,904 | (-0.4999564290046692, 0.4999564290046692, -0.4999564290046692, 0.4999564290046692, -0.5, 0.5) | No (312 open boundary edges) |
| Cylinder (analytic) | `3d-files/cylinder.stl` | **Validation** | Generated locally (no download) by `simulation/scripts/generate_shapes.py` | N/A (generated analytic geometry, not a third-party asset) | N/A | 476 | (-0.25, 0.25, -0.25, 0.25, -0.5, 0.5) | Yes (0 open boundary edges) |
| Ahmed-style box (analytic) | `3d-files/ahmed_box.stl` | **Validation** | Generated locally (no download) by `simulation/scripts/generate_shapes.py` (plain rectangular prism using reference Ahmed body dimensions 1.044 × 0.389 × 0.288 m) | N/A (generated analytic geometry, not a third-party asset) | N/A | 12 | (-0.5220000147819519, 0.5220000147819519, -0.19449999928474426, 0.19449999928474426, -0.14399999380111694, 0.14399999380111694) | Yes (0 open boundary edges) |

## Notes

- **Watertight check (reported):** face counts and watertight status above are measured on the mesh returned by `sstream.mesh.load_surface(...)` (load/cleanup path) using PyVista/VTK `extract_feature_edges(boundary_edges=True, feature_edges=False, manifold_edges=False, non_manifold_edges=False)`; **0 open boundary edges = watertight**. This supersedes earlier asset-only notes. At runtime, `sstream.voxelize.voxelize(...)` may repair small holes and/or fill cells unreachable from outside (see its `watertight` flag and `warnings`), so voxelization-time behavior can differ from the load-time edge check (e.g. sphere may be reported repaired/fillable at voxelization).
- **Validation shapes:** Ahmed body, sphere, cylinder, and Ahmed-style box (standard/reference shapes suitable for validation checks).
- **Demo shapes:** DrivAerML notchback (complex production-like geometry for demonstration). Ahmed and DrivAer are **visual demo/reference geometries** in this context, not drag-validation targets against published high-Re values (see project solver-acceptance notes).
- Ahmed and DrivAer meshes were downloaded individually (one file each) from the HuggingFace datasets above on 2026-11-16; dataset licenses and citations are recorded from the source dataset cards at that time.
- **Geometry note:** AhmedML `ahmed_1.stl` is a specific dataset variant/morph (run 1); its file bounding box/length (~1.188 m streamwise extent in file coordinates) and other details differ from the canonical/textbook Ahmed body, so do not compare it to canonical reference-body numbers without accounting for the exact variant and conditions. The analytic `ahmed_box.stl` is a plain rectangular prism using reference Ahmed dimensions (not an Ahmed body).

## Licensing

- **Downloaded model files (`ahmed_1.stl`, `drivaer_1.stl`) are licensed under CC BY-SA 4.0 as stated by their source datasets** (attribution and ShareAlike obligations apply to those model files and derivatives of them). Keep attribution/citations as recorded above.
- **The repository’s code license (MIT, see root `LICENSE`) does not cover these third-party/downloaded model assets.** Check the source license texts yourself for your use case.

## Large STL files and git history (proposal, not executed)

The committed DrivAer STL (~753k faces) and Ahmed STL (~150k faces) make git history heavy. Decide and apply one of the following **before further history growth** (rewriting already-pushed/committed history is disruptive, so treat this as a planned follow-up, not performed here):

1. **Git LFS (recommended if files stay in-repo):** track `simulation/assets/3d-files/*.stl` (and/or specific large files) with Git LFS, migrate existing blobs to LFS, and document LFS as a prerequisite for clones. Pros: paths stay stable for code/tests; Cons: requires LFS setup and a history rewrite/migration to shrink existing history.
2. **Download script + gitignore (recommended for minimal repo size):** remove large third-party STLs from version control, add them to `.gitignore`, and provide a deterministic download/generate script (e.g. `simulation/scripts/fetch_models.py`) that fetches only the exact recorded files from the URLs above and verifies checksums/metadata, while keeping small analytic shapes generated locally. Pros: small history going forward; Cons: requires network for first setup and a history rewrite to purge already-committed blobs.
3. **Hybrid:** keep small analytic/generated STLs committed; fetch/cache large dataset STLs via script or LFS pointers.

**Not executed:** no `.gitattributes`/LFS migration, no file removals, and no history rewrite were performed as part of this change. Choose an approach and coordinate any history rewrite explicitly.
