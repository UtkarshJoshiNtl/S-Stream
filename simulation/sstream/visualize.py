"""Offline visualization pipeline for LBM results.

Loads velocity snapshots, computes derived fields (Q-criterion, smoke
particles), renders frames with PyVista, and exports video or VTI files
for ParaView.

This module never touches the solver — it only reads snapshot files.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .fields import q_criterion, speed


@dataclass
class SnapshotData:
    """One time step of simulation data."""
    step: int
    velocity: np.ndarray  # (nx, ny, nz, 3) float32
    density: np.ndarray | None = None  # (nx, ny, nz) float32
    solid: np.ndarray | None = None  # (nx, ny, nz) bool


def load_snapshots(snapshot_dir: Path) -> tuple[list[SnapshotData], dict]:
    """Load all velocity snapshots and metadata from a snapshot directory.

    Returns (snapshots, meta) where snapshots is sorted by step and meta
    contains grid info, spacing, and solid mask path.
    """
    snapshot_dir = Path(snapshot_dir)
    meta_path = snapshot_dir / "meta.json"
    if not meta_path.exists():
        raise FileNotFoundError(f"No meta.json in {snapshot_dir}")

    meta = json.loads(meta_path.read_text())

    # Load solid mask
    solid_path = snapshot_dir / meta["solid_mask"]["file"]
    solid = np.load(solid_path).astype(bool) if solid_path.exists() else None

    # Load velocity snapshots
    snapshots = []
    snap_info = meta["snapshots"]
    for step in snap_info["steps"]:
        vel_path = snapshot_dir / f"velocity_step_{step:07d}.npz"
        if not vel_path.exists():
            continue
        data = np.load(vel_path)
        velocity = data["velocity"].astype(np.float32)
        # velocity is stored as (nx, ny, nz, 3) but might be (nx, ny, nz, 3) or flattened
        if velocity.ndim == 4 and velocity.shape[-1] == 3:
            pass  # already correct
        elif velocity.ndim == 3:
            # Might be stored as separate components
            velocity = velocity.reshape(*velocity.shape[:3], 3)
        snapshots.append(SnapshotData(step=step, velocity=velocity, solid=solid))

    return snapshots, meta


def compute_q_field(velocity: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    """Compute Q-criterion from velocity field."""
    return q_criterion(velocity, spacing)


def compute_speed_field(velocity: np.ndarray) -> np.ndarray:
    """Compute velocity magnitude."""
    return speed(velocity)


class SmokeParticles:
    """Particles seeded at inlet and advected through velocity fields.

    Particles are advected using trilinear interpolation of the velocity
    field.  Particles that exit the domain or hit solid are reseeded at
    the inlet with new random positions.
    """

    def __init__(
        self,
        shape: tuple[int, int, int],
        n_particles: int = 10000,
        seed: int = 42,
        inlet_x: int = 0,
    ):
        self.shape = shape
        self.n_particles = n_particles
        self.rng = np.random.default_rng(seed)
        self.inlet_x = inlet_x
        self.positions = self._seed_particles()
        self.ages = np.zeros(n_particles, dtype=np.float32)

    def _seed_particles(self) -> np.ndarray:
        """Seed particles randomly in the inlet plane."""
        nx, ny, nz = self.shape
        positions = np.zeros((self.n_particles, 3), dtype=np.float32)
        positions[:, 0] = self.inlet_x + self.rng.random(self.n_particles) * 2.0
        positions[:, 1] = self.rng.random(self.n_particles) * (ny - 1)
        positions[:, 2] = self.rng.random(self.n_particles) * (nz - 1)
        return positions

    def _trilinear_sample(self, field: np.ndarray, pos: np.ndarray) -> np.ndarray:
        """Sample field at particle positions using trilinear interpolation."""
        nx, ny, nz = field.shape[:3]
        result = np.zeros((len(pos),) + field.shape[3:] if field.ndim == 4 else (len(pos),), dtype=np.float32)

        for i, p in enumerate(pos):
            x, y, z = p
            # Clamp to valid range
            x = np.clip(x, 0, nx - 1.001)
            y = np.clip(y, 0, ny - 1.001)
            z = np.clip(z, 0, nz - 1.001)

            x0, y0, z0 = int(x), int(y), int(z)
            x1, y1, z1 = min(x0 + 1, nx - 1), min(y0 + 1, ny - 1), min(z0 + 1, nz - 1)
            tx, ty, tz = x - x0, y - y0, z - z0

            if field.ndim == 4:  # velocity field (nx, ny, nz, 3)
                c000 = field[x0, y0, z0]
                c100 = field[x1, y0, z0]
                c010 = field[x0, y1, z0]
                c110 = field[x1, y1, z0]
                c001 = field[x0, y0, z1]
                c101 = field[x1, y0, z1]
                c011 = field[x0, y1, z1]
                c111 = field[x1, y1, z1]

                c00 = c000 * (1 - tx) + c100 * tx
                c10 = c010 * (1 - tx) + c110 * tx
                c01 = c001 * (1 - tx) + c101 * tx
                c11 = c011 * (1 - tx) + c111 * tx
                c0 = c00 * (1 - ty) + c10 * ty
                c1 = c01 * (1 - ty) + c11 * ty
                result[i] = c0 * (1 - tz) + c1 * tz
            else:  # scalar field (nx, ny, nz)
                c000 = field[x0, y0, z0]
                c100 = field[x1, y0, z0]
                c010 = field[x0, y1, z0]
                c110 = field[x1, y1, z0]
                c001 = field[x0, y0, z1]
                c101 = field[x1, y0, z1]
                c011 = field[x0, y1, z1]
                c111 = field[x1, y1, z1]

                c00 = c000 * (1 - tx) + c100 * tx
                c10 = c010 * (1 - tx) + c110 * tx
                c01 = c001 * (1 - tx) + c101 * tx
                c11 = c011 * (1 - tx) + c111 * tx
                c0 = c00 * (1 - ty) + c10 * ty
                c1 = c01 * (1 - ty) + c11 * ty
                result[i] = c0 * (1 - tz) + c1 * tz

        return result

    def advect(self, velocity: np.ndarray, dt: float = 1.0, solid: np.ndarray | None = None) -> None:
        """Advect particles through velocity field for one time step."""
        velocities = self._trilinear_sample(velocity, self.positions)
        self.positions += velocities * dt
        self.ages += dt

        # Remove particles that exited or hit solid, reseed at inlet
        nx, ny, nz = self.shape
        out_of_bounds = (
            (self.positions[:, 0] < 0) | (self.positions[:, 0] >= nx) |
            (self.positions[:, 1] < 0) | (self.positions[:, 1] >= ny) |
            (self.positions[:, 2] < 0) | (self.positions[:, 2] >= nz)
        )

        if solid is not None:
            # Check if particles are inside solid
            xi = np.clip(self.positions[:, 0].astype(int), 0, nx - 1)
            yi = np.clip(self.positions[:, 1].astype(int), 0, ny - 1)
            zi = np.clip(self.positions[:, 2].astype(int), 0, nz - 1)
            in_solid = solid[xi, yi, zi]
            out_of_bounds |= in_solid

        n_reseed = out_of_bounds.sum()
        if n_reseed > 0:
            self.positions[out_of_bounds] = self._seed_particles()[:n_reseed]
            self.ages[out_of_bounds] = 0

    def get_positions(self) -> np.ndarray:
        """Get current particle positions."""
        return self.positions.copy()

    def get_ages(self) -> np.ndarray:
        """Get particle ages (time since seeding)."""
        return self.ages.copy()


def render_frame(
    velocity: np.ndarray,
    solid: np.ndarray,
    q_field: np.ndarray | None = None,
    speed_field: np.ndarray | None = None,
    particles: SmokeParticles | None = None,
    spacing: tuple[float, float, float] = (1.0, 1.0, 1.0),
    show_solid: bool = True,
    show_q: bool = True,
    show_particles: bool = True,
    show_volume: bool = True,
    q_threshold: float | None = None,
    camera_position: tuple[float, float, float] | None = None,
) -> np.ndarray:
    """Render a single frame as an image array.

    Returns RGB image as (height, width, 3) uint8.
    """
    import pyvista as pv

    nx, ny, nz = velocity.shape[:3]

    # Create grid for volume rendering
    # RectilinearGrid expects 1D arrays of coordinates
    x = np.arange(nx) * spacing[0]
    y = np.arange(ny) * spacing[1]
    z = np.arange(nz) * spacing[2]
    grid = pv.RectilinearGrid(x, y, z)

    # Add velocity magnitude as scalar
    if speed_field is None:
        speed_field = compute_speed_field(velocity)
    grid["speed"] = speed_field.ravel()

    # Add Q-criterion
    if q_field is not None and show_q:
        grid["q"] = q_field.ravel()

    # Plot
    plotter = pv.Plotter(off_screen=True, window_size=(1920, 1080))
    plotter.set_background("white")

    # Add volume rendering of speed (optional)
    if show_volume:
        plotter.add_volume(
            grid,
            scalars="speed",
            cmap="viridis",
            opacity="sigmoid",
            clim=[0, np.percentile(speed_field, 95)],
        )

    # Add Q-criterion isosurface
    if q_field is not None and show_q:
        # Auto-compute threshold if not specified: use 99th percentile of positive Q
        if q_threshold is None:
            positive_q = q_field[q_field > 0]
            if len(positive_q) > 0:
                q_threshold = np.percentile(positive_q, 95)
            else:
                q_threshold = 0.0

        q_mask = q_field > q_threshold
        if q_mask.any():
            # Extract isosurface using PyVista's contour
            q_grid = pv.RectilinearGrid(x, y, z)
            q_grid["q"] = q_field.ravel()
            contours = q_grid.contour(q_threshold)
            if len(contours.points) > 0:
                plotter.add_mesh(
                    contours,
                    scalars="speed" if "speed" in contours.point_data else None,
                    cmap="hot",
                    opacity=0.7,
                )

    # Add solid body
    if show_solid and solid is not None:
        solid_mask = solid.astype(np.uint8)
        # Create surface mesh from solid mask
        solid_grid = pv.RectilinearGrid(x, y, z)
        solid_grid["solid"] = solid_mask.ravel()
        solid_surface = solid_grid.contour(0.5)
        if len(solid_surface.points) > 0:
            plotter.add_mesh(solid_surface, color="gray", opacity=0.9)

    # Add smoke particles
    if show_particles and particles is not None:
        pos = particles.get_positions()
        # Convert lattice coords to physical
        pos_phys = pos * np.array(spacing)
        particle_cloud = pv.PolyData(pos_phys)
        plotter.add_points(
            particle_cloud,
            color="black",
            point_size=2,
            opacity=0.5,
            render_points_as_spheres=False,
        )

    # Set camera
    if camera_position is None:
        # Default: side view
        center = np.array([nx * spacing[0] / 2, ny * spacing[1] / 2, nz * spacing[2] / 2])
        camera_position = (center[0], -ny * spacing[1] * 2, center[2])
        plotter.camera_position = [camera_position, tuple(center), (0, 0, 1)]
    else:
        center = np.array([nx * spacing[0] / 2, ny * spacing[1] / 2, nz * spacing[2] / 2])
        plotter.camera_position = [camera_position, tuple(center), (0, 0, 1)]

    plotter.add_axes()
    plotter.show(auto_close=False)

    # Capture screenshot
    img = plotter.screenshot(return_img=True)
    plotter.close()

    return img


def render_video(
    snapshots: list[SnapshotData],
    meta: dict,
    output_path: Path,
    n_particles: int = 10000,
    fps: int = 30,
    show_q: bool = True,
    show_particles: bool = True,
    show_volume: bool = True,
    q_threshold: float | None = None,
) -> Path:
    """Render snapshots to video file.

    Args:
        snapshots: List of SnapshotData sorted by step
        meta: Metadata dict from load_snapshots
        output_path: Output video file path (.mp4)
        n_particles: Number of smoke particles
        fps: Frames per second
        show_q: Show Q-criterion isosurfaces
        show_particles: Show smoke particles
        q_threshold: Q-criterion threshold for isosurface

    Returns:
        Path to output video file
    """
    import imageio.v2 as imageio

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Get grid info
    grid_shape = meta["tunnel"]["grid_shape"]
    spacing = tuple(meta["tunnel"]["spacing"])
    solid = snapshots[0].solid if snapshots else None

    # Initialize particles
    particles = SmokeParticles(grid_shape, n_particles=n_particles) if show_particles else None

    # Render frames
    frames = []
    for i, snap in enumerate(snapshots):
        velocity = snap.velocity
        q_field = compute_q_field(velocity, spacing) if show_q else None
        speed_field = compute_speed_field(velocity)

        # Advect particles between snapshots
        if particles is not None and i > 0:
            # Estimate dt from step difference
            dt = (snap.step - snapshots[i-1].step) / 100.0  # Normalize
            particles.advect(velocity, dt=dt, solid=solid)

        frame = render_frame(
            velocity=velocity,
            solid=solid,
            q_field=q_field,
            speed_field=speed_field,
            particles=particles,
            spacing=spacing,
            show_q=show_q,
            show_particles=show_particles,
            show_volume=show_volume,
            q_threshold=q_threshold,
        )
        frames.append(frame)

    # Write video
    imageio.mimsave(str(output_path), frames, fps=fps)
    return output_path


def export_vti(
    velocity: np.ndarray,
    density: np.ndarray | None,
    solid: np.ndarray,
    spacing: tuple[float, float, float],
    output_path: Path,
    q_field: np.ndarray | None = None,
) -> Path:
    """Export fields to VTI format for ParaView.

    VTI (VTK Image Data) is a simple XML format that ParaView can read.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nx, ny, nz = velocity.shape[:3]

    # Compute derived fields if not provided
    if q_field is None:
        q_field = compute_q_field(velocity, spacing)
    speed_field = compute_speed_field(velocity)

    # Helper to format array data as space-separated values
    def format_array(arr: np.ndarray, name: str, dtype: str = "Float32") -> str:
        flat = arr.T.flatten() if arr.ndim == 3 else arr.flatten()
        # Format as space-separated values
        values = ' '.join(f'{v:.6e}' for v in flat)
        return f'        <DataArray type="{dtype}" Name="{name}" format="ascii">\n          {values}\n        </DataArray>'

    # Build VTI content
    lines = []
    lines.append('<?xml version="1.0"?>')
    lines.append('<VTKFile type="ImageData" version="0.1" byte_order="LittleEndian">')
    lines.append(f'  <ImageData WholeExtent="0 {nx-1} 0 {ny-1} 0 {nz-1}" Origin="0 0 0" Spacing="{spacing[0]} {spacing[1]} {spacing[2]}">')
    lines.append(f'    <Piece Extent="0 {nx-1} 0 {ny-1} 0 {nz-1}">')
    lines.append('      <PointData>')
    lines.append(format_array(velocity[..., 0], 'velocity_x'))
    lines.append(format_array(velocity[..., 1], 'velocity_y'))
    lines.append(format_array(velocity[..., 2], 'velocity_z'))
    lines.append(format_array(speed_field, 'speed'))
    lines.append(format_array(q_field, 'q_criterion'))
    if density is not None:
        lines.append(format_array(density, 'density'))
    lines.append(format_array(solid.astype(np.uint8), 'solid', 'UInt8'))
    lines.append('      </PointData>')
    lines.append('    </Piece>')
    lines.append('  </ImageData>')
    lines.append('</VTKFile>')

    output_path.write_text('\n'.join(lines))
    return output_path
