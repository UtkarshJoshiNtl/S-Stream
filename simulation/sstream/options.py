"""CLI flags and helpers shared by ``sstream`` and ``sstream-place``.

The two commands accept the same orientation vocabulary
(``--orient/--up/--flip/--yaw/--pitch/--fraction``), so the argument
definitions live here once and both front ends import them.
"""

from __future__ import annotations

import argparse

from .schema import MeshSpec


def add_orientation_args(parser: argparse.ArgumentParser) -> None:
    """Register the orientation and placement flags on ``parser``."""
    parser.add_argument(
        "--orient",
        choices=("auto-vehicle", "upright", "none"),
        default="auto-vehicle",
        help="auto-vehicle: longest extent along the wind, shortest up "
        "(default, for cars/planes); upright: keep the model's up axis "
        "(--up) and put the wind along the longer horizontal extent "
        "(trees, buildings, people, animals); none: use the file as-is",
    )
    parser.add_argument(
        "--up",
        choices=("+Y", "+Z"),
        default="+Y",
        help="model up axis for --orient upright (default +Y; Blender exports Y-up)",
    )
    parser.add_argument(
        "--flip",
        action="store_true",
        help="rotate 180 deg about up, reversing nose/tail",
    )
    parser.add_argument("--yaw", type=float, default=0.0, help="heading, degrees about up")
    parser.add_argument(
        "--pitch", type=float, default=0.0, help="angle of attack, degrees about lateral"
    )
    parser.add_argument(
        "--fraction",
        type=float,
        default=0.2,
        help="longest model extent as a fraction of the tunnel length (default 0.2)",
    )


def mesh_spec_from_args(args: argparse.Namespace) -> MeshSpec:
    """Build the :class:`MeshSpec` an ``args`` namespace describes."""
    return MeshSpec(
        orient=args.orient,
        up=args.up,
        flip=args.flip,
        pitch=args.pitch,
        yaw=args.yaw,
        fraction=args.fraction,
    )
