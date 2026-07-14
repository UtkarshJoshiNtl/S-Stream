from __future__ import annotations


class SmokeMixin:
    """Mixin for smoke advection, diffusion, and decay logic.

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
    - self._x_coords
    - self._y_coords
    - self._lap_buffer (optional, for CPU engines)
    - self.xp (numpy or cupy module)
    """

    def apply_emitters(self) -> None:
        xp = self.xp
        for x, y, strength in self.emitters:
            if 0 <= y < self.height and 0 <= x < self.width:
                self.smoke[y, x] += strength
        xp.clip(self.smoke, 0, 1, out=self.smoke)

    def advect_smoke(self) -> None:
        xp = self.xp
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

    def diffuse_smoke(self) -> None:
        xp = self.xp
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

    def decay_smoke(self) -> None:
        self.smoke *= self.smoke_decay
