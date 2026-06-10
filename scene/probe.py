from __future__ import annotations

import numpy as np

from engines.base import SimEngine
from scene.scene import ProbeSpec

MAX_HISTORY = 512


class Probe:
    def __init__(self, spec: ProbeSpec) -> None:
        self.spec = spec
        self.history: dict[str, list[float]] = {f: [] for f in spec.fields}
        self._step = 0

    def record(self, sim: SimEngine) -> None:
        y, x = self.spec.y, self.spec.x
        h, w = sim.grid_shape
        y = int(np.clip(y, 0, h - 1))
        x = int(np.clip(x, 0, w - 1))

        vel = sim.get_velocity()
        u_val = float(vel[y, x, 0])
        v_val = float(vel[y, x, 1])

        for f in self.spec.fields:
            if f == "u":
                val = u_val
            elif f == "v":
                val = v_val
            elif f == "speed":
                val = np.sqrt(u_val**2 + v_val**2)
            elif f == "pressure":
                rho = float(sim.get_density()[y, x])
                val = rho - 1.0
            else:
                continue
            self.history[f].append(val)
            if len(self.history[f]) > MAX_HISTORY:
                self.history[f].pop(0)
        self._step += 1

    @property
    def step(self) -> int:
        return self._step

    def clear(self) -> None:
        for f in self.spec.fields:
            self.history[f].clear()
        self._step = 0
