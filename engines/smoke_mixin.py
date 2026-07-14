from __future__ import annotations


class SmokeMixin:
    """Mixin for smoke advection, diffusion, and decay logic.

    Supports both 2D (shape: H, W) and 3D (shape: D, H, W) smoke fields.

    Concrete engines must set the following attributes before calling these methods:
    - self.smoke
    - self.obstacles
    - self.u
    - self.v
    - self.smoke_diffusion
    - self.smoke_decay
    - self.emitters
    - self.width
    - self.height
    - self.depth (3D only)
    - self._x_coords
    - self._y_coords
    - self._z_coords (3D only)
    - self._lap_buffer (optional, for CPU engines)
    - self.xp (numpy or cupy module)
    """

    def apply_emitters(self) -> None:
        xp = self.xp
        is_3d = self.smoke.ndim == 3
        for emitter in self.emitters:
            if is_3d:
                x, y, z, strength = emitter
                if (0 <= z < self.depth and 0 <= y < self.height
                        and 0 <= x < self.width):
                    self.smoke[z, y, x] += strength
            else:
                x, y, strength = emitter
                if 0 <= y < self.height and 0 <= x < self.width:
                    self.smoke[y, x] += strength
        xp.clip(self.smoke, 0, 1, out=self.smoke)

    def advect_smoke(self) -> None:
        xp = self.xp
        is_3d = self.smoke.ndim == 3
        if is_3d:
            self._advect_smoke_3d(xp)
        else:
            self._advect_smoke_2d(xp)

    def _advect_smoke_2d(self, xp) -> None:
        u_adv = xp.where(self.obstacles, 0.0, self.u)
        v_adv = xp.where(self.obstacles, 0.0, self.v)

        x_orig = self._x_coords[xp.newaxis, :] - u_adv
        y_orig = self._y_coords[:, xp.newaxis] - v_adv
        x_orig = xp.clip(x_orig, 0, self.width - 1)
        y_orig = xp.clip(y_orig, 0, self.height - 1)

        x0 = xp.floor(x_orig).astype(xp.int32)
        y0 = xp.floor(y_orig).astype(xp.int32)
        x1 = xp.minimum(x0 + 1, self.width - 1)
        y1 = xp.minimum(y0 + 1, self.height - 1)

        fx = x_orig - x0
        fy = y_orig - y0

        c00 = self.smoke[y0, x0]
        c10 = self.smoke[y0, x1]
        c01 = self.smoke[y1, x0]
        c11 = self.smoke[y1, x1]

        self.smoke = (
            c00 * (1 - fx) * (1 - fy)
            + c10 * fx * (1 - fy)
            + c01 * (1 - fx) * fy
            + c11 * fx * fy
        )

    def _advect_smoke_3d(self, xp) -> None:
        u_adv = xp.where(self.obstacles, 0.0, self.u)
        v_adv = xp.where(self.obstacles, 0.0, self.v)
        w_adv = xp.where(self.obstacles, 0.0, self.w_vel)

        x_orig = self._x_coords[xp.newaxis, xp.newaxis, :] - u_adv
        y_orig = self._y_coords[xp.newaxis, :, xp.newaxis] - v_adv
        z_orig = self._z_coords[:, xp.newaxis, xp.newaxis] - w_adv

        x_orig = xp.clip(x_orig, 0, self.width - 1)
        y_orig = xp.clip(y_orig, 0, self.height - 1)
        z_orig = xp.clip(z_orig, 0, self.depth - 1)

        x0 = xp.floor(x_orig).astype(xp.int32)
        y0 = xp.floor(y_orig).astype(xp.int32)
        z0 = xp.floor(z_orig).astype(xp.int32)
        x1 = xp.minimum(x0 + 1, self.width - 1)
        y1 = xp.minimum(y0 + 1, self.height - 1)
        z1 = xp.minimum(z0 + 1, self.depth - 1)

        fx = x_orig - x0
        fy = y_orig - y0
        fz = z_orig - z0

        c000 = self.smoke[z0, y0, x0]
        c100 = self.smoke[z0, y0, x1]
        c010 = self.smoke[z0, y1, x0]
        c110 = self.smoke[z0, y1, x1]
        c001 = self.smoke[z1, y0, x0]
        c101 = self.smoke[z1, y0, x1]
        c011 = self.smoke[z1, y1, x0]
        c111 = self.smoke[z1, y1, x1]

        self.smoke = (
            c000 * (1 - fx) * (1 - fy) * (1 - fz)
            + c100 * fx * (1 - fy) * (1 - fz)
            + c010 * (1 - fx) * fy * (1 - fz)
            + c110 * fx * fy * (1 - fz)
            + c001 * (1 - fx) * (1 - fy) * fz
            + c101 * fx * (1 - fy) * fz
            + c011 * (1 - fx) * fy * fz
            + c111 * fx * fy * fz
        )

    def diffuse_smoke(self) -> None:
        xp = self.xp
        is_3d = self.smoke.ndim == 3
        if is_3d:
            self._diffuse_smoke_3d(xp)
        else:
            self._diffuse_smoke_2d(xp)

    def _diffuse_smoke_2d(self, xp) -> None:
        s = self.smoke
        d = self.smoke_diffusion
        if hasattr(self, "_lap_buffer"):
            lap = self._lap_buffer
        else:
            lap = xp.zeros_like(s)
        lap[:] = 0.0
        lap[1:] += s[:-1] - s[1:]
        lap[:-1] += s[1:] - s[:-1]
        lap[:, 1:] += s[:, :-1] - s[:, 1:]
        lap[:, :-1] += s[:, 1:] - s[:, :-1]
        s += d * lap

    def _diffuse_smoke_3d(self, xp) -> None:
        s = self.smoke
        d = self.smoke_diffusion
        if hasattr(self, "_lap_buffer"):
            lap = self._lap_buffer
        else:
            lap = xp.zeros_like(s)
        lap[:] = 0.0
        # Z-axis
        lap[1:] += s[:-1] - s[1:]
        lap[:-1] += s[1:] - s[:-1]
        # Y-axis
        lap[:, 1:] += s[:, :-1] - s[:, 1:]
        lap[:, :-1] += s[:, 1:] - s[:, :-1]
        # X-axis
        lap[:, :, 1:] += s[:, :, :-1] - s[:, :, 1:]
        lap[:, :, :-1] += s[:, :, 1:] - s[:, :, :-1]
        s += d * lap

    def decay_smoke(self) -> None:
        self.smoke *= self.smoke_decay
