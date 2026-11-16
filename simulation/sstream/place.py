"""Cheap placement check: render the model in the tunnel, no simulation.

``sstream-place MODEL [orientation flags] --out place.png`` runs the same
import-and-place contract as the solver front end, then draws the placed
model inside the tunnel box with an arrow showing the wind direction.
It exists so orientation (``--orient/--flip/--yaw/--pitch``) can be
verified in seconds instead of after a long run.

This module owns the only rendering in the placement path; the solver
pipeline never imports it.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pyvista as pv

from .mesh import DEFAULT_MODEL, build_model, load_surface
from .options import add_orientation_args, mesh_spec_from_args
from .schema import TunnelSpec

WIND_COLOR = "#4aa3df"
TUNNEL_COLOR = "#8a99a8"
FLOOR_COLOR = "#20262e"
MODEL_COLOR = "#b8c0c8"


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="sstream-place",
        description="Place a model in the tunnel and render it with a wind "
        "arrow, without running the simulation.",
    )
    parser.add_argument(
        "mesh",
        nargs="?",
        default=str(DEFAULT_MODEL),
        help=f"surface mesh to place (.obj/.stl/.ply; default: bundled Ahmed body)",
    )
    add_orientation_args(parser)
    parser.add_argument("--out", required=True, help="output PNG path")
    parser.add_argument(
        "--decimate",
        type=int,
        default=None,
        metavar="TARGET_FACES",
        help="optionally decimate the input mesh toward TARGET_FACES triangles (default: off)",
    )
    parser.add_argument("--width", type=int, default=1600, help="image width in px")
    parser.add_argument("--height", type=int, default=900, help="image height in px")
    return parser.parse_args(argv)


def render_placement(
    mesh_path: str | Path,
    tunnel: TunnelSpec,
    out_path: str | Path,
    orient_args: argparse.Namespace,
    width: int = 1600,
    height: int = 900,
    decimate: int | None = None,
) -> None:
    """Place ``mesh_path`` in ``tunnel`` and write the scene to ``out_path``."""
    surface = load_surface(mesh_path, decimate=decimate)
    build_temp = build_model(surface, mesh_spec_from_args(orient_args), TunnelSpec())
    model_bounds = np.asarray(build_temp.mesh.bounds, dtype=float).reshape(3, 2)
    model_length = float(model_bounds[0, 1] - model_bounds[0, 0])
    tunnel = TunnelSpec.from_model_length(model_length)
    build = build_model(surface, mesh_spec_from_args(orient_args), tunnel)

    placed = np.asarray(build.mesh.bounds, dtype=float).reshape(3, 2)
    placed_extent = placed[:, 1] - placed[:, 0]
    bbox_blockage = placed_extent[1] * placed_extent[2] / (tunnel.width * tunnel.height)
    print(
        f"Tunnel        {tunnel.length:.2f} x {tunnel.width:.2f} x {tunnel.height:.2f} m "
        f"(4L x 1.5L x 1.0L, L={model_length:.2f} m)"
    )
    print(
        f"Placed        {placed_extent[0]:.2f} x {placed_extent[1]:.2f} x "
        f"{placed_extent[2]:.2f} m, bbox blockage {bbox_blockage:.1%}"
    )

    plotter = pv.Plotter(off_screen=True, window_size=(width, height))
    plotter.set_background((0.03, 0.04, 0.06), top=(0.10, 0.14, 0.20))

    # Tunnel box, wireframe, and floor slab.
    plotter.add_mesh(
        pv.Box(bounds=tunnel.bounds),
        style="wireframe",
        color=TUNNEL_COLOR,
        line_width=1.5,
    )
    plotter.add_mesh(
        pv.Box(
            bounds=(
                tunnel.x_min,
                tunnel.x_max,
                tunnel.y_min,
                tunnel.y_max,
                tunnel.z_min - 0.02,
                tunnel.z_min,
            )
        ),
        color=FLOOR_COLOR,
        opacity=0.85,
        show_edges=False,
    )

    # The model, glossy.
    plotter.add_mesh(
        build.mesh,
        color=MODEL_COLOR,
        specular=0.7,
        specular_power=18.0,
        smooth_shading=True,
    )

    # Wind arrow upstream of the model, pointing +X.
    mid_z = float(placed[2].mean())
    arrow = pv.Arrow(
        start=np.array([tunnel.x_min + 0.35, placed[1].mean(), mid_z]),
        direction=np.array([1.0, 0.0, 0.0]),
        scale=0.7,
    )
    plotter.add_mesh(arrow, color=WIND_COLOR, specular=0.3, smooth_shading=True)
    tip = np.array([tunnel.x_min + 1.05, placed[1].mean(), mid_z])
    plotter.add_point_labels(
        [tip],
        ["wind"],
        point_size=6,
        font_size=14,
        text_color=WIND_COLOR,
        shape_color=None,
        show_points=False,
    )

    # Three-quarter view from upstream and above, framing the whole tunnel.
    centre = np.array([tunnel.length * 0.5, 0.0, tunnel.height * 0.45])
    plotter.camera_position = [
        (tunnel.x_min - 4.6, -4.4, 3.6),
        centre.tolist(),
        (0.0, 0.0, 1.0),
    ]
    plotter.show(screenshot=str(out_path), auto_close=False)
    plotter.close()
    print(f"Wrote {out_path}")


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    out = Path(args.out)
    try:
        render_placement(args.mesh, TunnelSpec(), out, args, args.width, args.height, args.decimate)
        return 0
    except Exception as exc:  # CLI boundary: one clean message, exit code 1
        print(f"sstream-place: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
