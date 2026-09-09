from __future__ import annotations

import logging
from collections import deque

import numpy as np

from engines.base import SimEngine
from scene.scene import ProbeSpec

logger = logging.getLogger(__name__)

MAX_HISTORY = 512

# Quantities a probe computes locally rather than reading from the engine.
#
# `speed` is deliberately here even though the engine exposes a "speed" field.
# get_field("speed") returns a display value normalised to 0..1 against a running
# max, whereas the probe series is the physical magnitude. They are different
# quantities, so swapping one for the other would silently rescale every
# existing speed trace in every saved scene. Anything else the engine declares is
# recorded straight from the field.
LOCAL_FIELDS = ("u", "v", "speed")


def recordable_fields(sim: SimEngine) -> list[str]:
    """Every field name a probe on this engine can record.

    The local quantities come first, then whatever the engine declares. This is
    the same engine-derived shape as the field catalog (see C18) -- a probe can
    only ask for something the engine can actually produce.
    """
    names = list(LOCAL_FIELDS)
    for name in sim.get_field_names():
        if name not in names:
            names.append(name)
    return names


class Probe:
    def __init__(self, spec: ProbeSpec) -> None:
        self.spec = spec
        self.history: dict[str, deque[float]] = {
            f: deque(maxlen=MAX_HISTORY) for f in spec.fields
        }
        self._step = 0
        self._unavailable: set[str] = set()

    def _read_local(self, name: str, u_val: float, v_val: float) -> float:
        if name == "u":
            return u_val
        if name == "v":
            return v_val
        return float(np.hypot(u_val, v_val))

    def record(
        self, sim: SimEngine, cache: dict[str, np.ndarray] | None = None
    ) -> None:
        """Append one sample per requested field.

        `cache` is an optional per-step scratch dict shared across probes. A
        field is a whole-domain array, so without it N probes on the same field
        would each recompute it -- and get_field normalises against a running
        max, so recomputing is not merely wasteful, it perturbs the value the
        next probe sees.
        """
        y, x = self.spec.y, self.spec.x
        h, w = sim.grid_shape
        y = int(np.clip(y, 0, h - 1))
        x = int(np.clip(x, 0, w - 1))

        u_val, v_val = sim.get_velocity_at(x, y)

        for f in self.spec.fields:
            if f in LOCAL_FIELDS:
                val = self._read_local(f, u_val, v_val)
            else:
                if f in self._unavailable:
                    continue
                try:
                    if cache is not None and f in cache:
                        field = cache[f]
                    else:
                        field = sim.get_field(f)
                        if cache is not None:
                            cache[f] = field
                except ValueError as exc:
                    # Normally caught earlier, at placement or preset load. If a
                    # field disappears while running (thermal toggled off), say
                    # so once rather than every single step.
                    self._unavailable.add(f)
                    logger.error(
                        "Probe at (%d, %d) cannot record %r and will skip it: %s",
                        x,
                        y,
                        f,
                        exc,
                    )
                    continue
                val = float(field[y, x])
            self.history[f].append(val)
        self._step += 1

    @property
    def step(self) -> int:
        return self._step

    @property
    def unavailable_fields(self) -> set[str]:
        """Fields this probe asked for but the engine cannot produce."""
        return set(self._unavailable)

    def clear(self) -> None:
        for f in self.spec.fields:
            self.history[f].clear()
        self._step = 0
        self._unavailable.clear()
