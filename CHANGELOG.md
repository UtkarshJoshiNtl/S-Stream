# Changelog

All notable changes to S-Stream will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/).

## [0.3.5] - 2026-07-15

### Added

- **Core engines**
  - D2Q9 CPU engine (`LBM2D`) with BGK collision
  - D3Q19 3D CPU engine (`LBM3D`) with BGK collision and thermal buoyancy
  - D2Q9 GPU engine (`LBM2DGPU`) via CuPy with fused CUDA kernel
  - D2Q9 PyTorch GPU engine (`LBM2DLettuce`) via Lettuce library
  - Shan-Chen single-component liquid engine (`LBM2DLiquid`) with wetting/non-wetting
  - Shan-Chen two-component engine (`LBM2DMultiComponent`) for immiscible fluids
- **Collision operators**
  - BGK (single-relaxation-time)
  - TRT (two-relaxation-time) for better boundary stability
  - MRT (multi-relaxation-time) for maximum accuracy
  - Smagorinsky subgrid-scale turbulence model
  - WALE near-wall turbulence model
- **Non-Newtonian rheology**
  - Power-law (shear-thinning/thickening)
  - Carreau (polymer behavior)
  - Bingham (yield stress fluids)
- **Thermal LBM** — Boussinesq buoyancy for natural convection
- **Geometry**
  - STL mesh import via trimesh
  - Image-to-obstacle (PNG/BMP → obstacle mask)
  - NACA 4-digit airfoil generator
  - Ellipse, channel/nozzle/diffuser, porous lattice primitives
  - Freehand polygon drawing
- **Scene system** — JSON save/load with full serialization
- **Guided Setup Wizard** — 10 templates across 3 categories
- **Visualization**
  - Engine-agnostic colormap system (10 fields, 6 colormaps)
  - Velocity quiver arrows, streamlines, pressure contours
  - Force arrows on obstacles, particle trail rendering
- **Analysis**
  - Reynolds number, drag coefficient, Strouhal number
  - Automatic flow regime classification (7 regimes)
  - Sanity checks and design scorecard
  - AI context builder for LLM integration
  - Parameter sweep runner with pyqtgraph plots
- **Particle tracer** — Lagrangian advection with RK2, trails, 2D/3D
- **Export** — High-res PNG, MP4/GIF video, probe CSV, Markdown reports
- **Jupyter** — `_repr_png_()`, `_repr_html_()`, `plot_*()` methods on all engines
- **REST API** — 11 FastAPI endpoints via `python main.py --serve`
- **Preset library** — 10 guided experiments
- **CI/CD** — GitHub Actions: lint, test (Ubuntu/Windows/macOS), PyInstaller builds, GitHub Releases, PyPI publish

### Changed

- Refactored step kernel into separate streaming, boundary, collision, and smoke phases
- All engines inherit `SmokeMixin` for shared smoke advection/diffusion/decay
- All engines inherit `ThermalMixin` for shared temperature distribution
- Scene panel dynamically adapts controls based on active engine type

## [0.3.0] - 2026-06-01

### Added

- Multi-component Shan-Chen engine with oil-water separation
- Guided Setup Wizard with 10 templates
- Lagrangian particle tracer with RK2 advection
- Jupyter notebook integration (`plot_*()` methods, inline display)
- REST API mode (FastAPI, 11 endpoints)
- Engine-specific scene panel controls (liquid and multi-component parameters)
- Color gradient perturbation for sharp interfaces
- Colormap system with 10 fields and 6 custom LUTs

## [0.2.0] - 2026-05-01

### Added

- 3D engine (D3Q19) with thermal buoyancy
- TRT and MRT collision operators
- Smagorinsky and WALE turbulence models
- Non-Newtonian rheology (Power-law, Carreau, Bingham)
- STL mesh import, image-to-obstacle, NACA airfoil
- Engine-agnostic colormap system
- Pressure contours, force arrows, particle trails
- Export dialog (image, video, data, Markdown reports)
- Flow regime detection, sanity checks, design scorecard
- Parameter sweep runner

## [0.1.0] - 2026-04-01

### Added

- Initial release: 2D LBM (D2Q9 + BGK)
- PySide6 workbench with OpenGL viewport
- Scene system with JSON save/load
- Obstacle drawing (circle, rectangle, polygon)
- Probe placement with live plots
- Basic analysis (Re, drag coefficient)
- 5 preset experiments
