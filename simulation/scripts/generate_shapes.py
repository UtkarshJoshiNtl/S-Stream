#!/usr/bin/env python3
"""Generate analytic test shapes (no download).

Writes STL files to the assets/3d-files directory by default.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pyvista as pv


def make_sphere(radius: float = 0.5) -> pv.PolyData:
    return pv.Sphere(radius=radius, theta_resolution=120, phi_resolution=120).triangulate().clean()


def make_cylinder(radius: float = 0.25, height: float = 1.0) -> pv.PolyData:
    cyl = pv.Cylinder(center=(0.0, 0.0, 0.0), direction=(0.0, 0.0, 1.0),
                      radius=radius, height=height, resolution=120, capping=True)
    return cyl.triangulate().clean()


def make_ahmed_box(length: float = 1.044, width: float = 0.389, height: float = 0.288) -> pv.PolyData:
    """Plain Ahmed-style box (rectangular prism), reference Ahmed body dimensions in metres."""
    box = pv.Cube(center=(0.0, 0.0, 0.0), x_length=length, y_length=width, z_length=height)
    return box.triangulate().clean()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "assets" / "3d-files"))
    args = parser.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    shapes = {
        "sphere.stl": make_sphere(),
        "cylinder.stl": make_cylinder(),
        "ahmed_box.stl": make_ahmed_box(),
    }
    for name, mesh in shapes.items():
        path = out / name
        mesh.save(path)
        print(f"wrote {path}  points={mesh.n_points} faces={mesh.n_cells}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
