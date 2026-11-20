#!/usr/bin/env python3
"""2D midplane slice visualization for S-Stream snapshots.

Reads a snapshot directory containing velocity_step_*.npz files, extracts
the midplane slice, and plots vorticity magnitude or speed with solid masking.

Usage:
    python scripts/slice_view.py <snapshot_dir> <output.png> [--field vorticity|speed] [--step N]
"""

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def load_snapshot(snapshot_dir: Path, step: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Load velocity field and solid mask from snapshot directory."""
    meta_path = snapshot_dir / "meta.json"
    solid_path = snapshot_dir / "solid.npy"

    if not meta_path.exists():
        raise FileNotFoundError(f"meta.json not found in {snapshot_dir}")
    if not solid_path.exists():
        raise FileNotFoundError(f"solid.npy not found in {snapshot_dir}")

    meta = json.loads(meta_path.read_text())
    solid = np.load(solid_path).astype(bool)

    # Find velocity file
    if step is not None:
        vel_path = snapshot_dir / f"velocity_step_{step:07d}.npz"
    else:
        vel_files = sorted(snapshot_dir.glob("velocity_step_*.npz"))
        if not vel_files:
            raise FileNotFoundError(f"No velocity_step_*.npz files in {snapshot_dir}")
        vel_path = vel_files[-1]  # Use latest

    if not vel_path.exists():
        raise FileNotFoundError(f"Velocity file not found: {vel_path}")

    data = np.load(vel_path)
    velocity = data["velocity"].astype(np.float32)

    return velocity, solid, meta


def compute_vorticity(velocity: np.ndarray, spacing: tuple[float, float, float] = (1.0, 1.0, 1.0)) -> np.ndarray:
    """Compute vorticity magnitude from velocity field."""
    vx, vy, vz = (velocity[..., i] for i in range(3))
    dvx = np.gradient(vx, *spacing, axis=(0, 1, 2))
    dvy = np.gradient(vy, *spacing, axis=(0, 1, 2))
    dvz = np.gradient(vz, *spacing, axis=(0, 1, 2))
    # Vorticity components
    omega_x = dvz[1] - dvy[2]
    omega_y = dvx[2] - dvz[0]
    omega_z = dvy[0] - dvx[1]
    return np.sqrt(omega_x**2 + omega_y**2 + omega_z**2)


def compute_speed(velocity: np.ndarray) -> np.ndarray:
    """Compute velocity magnitude."""
    return np.sqrt(np.sum(velocity**2, axis=-1))


def plot_slice(field: np.ndarray, solid: np.ndarray, output_path: Path, field_name: str, step: int):
    """Plot midplane slice with solid masking."""
    # Extract midplane (y = ny//2)
    ny = field.shape[1]
    mid_y = ny // 2
    slice_2d = field[:, mid_y, :]
    solid_2d = solid[:, mid_y, :]

    # Mask solid cells
    masked = np.ma.masked_where(solid_2d, slice_2d)

    fig, ax = plt.subplots(figsize=(12, 6))
    im = ax.imshow(masked.T, origin="lower", cmap="RdBu_r", aspect="auto")
    ax.set_xlabel("x (streamwise)")
    ax.set_ylabel("z (vertical)")
    ax.set_title(f"{field_name} at midplane (y={mid_y}), step {step}")
    plt.colorbar(im, ax=ax, label=field_name)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="2D midplane slice visualization")
    parser.add_argument("snapshot_dir", type=Path, help="Directory containing snapshots")
    parser.add_argument("output", type=Path, help="Output PNG path")
    parser.add_argument("--field", choices=["vorticity", "speed"], default="vorticity",
                        help="Field to visualize (default: vorticity)")
    parser.add_argument("--step", type=int, default=None, help="Specific step to visualize")
    args = parser.parse_args()

    velocity, solid, meta = load_snapshot(args.snapshot_dir, args.step)

    # Get spacing from meta
    spacing = tuple(meta["tunnel"]["spacing"])

    if args.field == "vorticity":
        field = compute_vorticity(velocity, spacing)
    else:
        field = compute_speed(velocity)

    # Determine step number
    if args.step is not None:
        step = args.step
    else:
        vel_files = sorted(args.snapshot_dir.glob("velocity_step_*.npz"))
        step = int(vel_files[-1].stem.split("_")[-1]) if vel_files else 0

    plot_slice(field, solid, args.output, args.field, step)


if __name__ == "__main__":
    main()
