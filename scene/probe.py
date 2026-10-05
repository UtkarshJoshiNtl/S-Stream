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


def default_probe_fields(sim: SimEngine) -> list[str]:
    """Fields to give a newly placed probe.

    Derived from the engine rather than hardcoded, for the same reason the field
    combo is: a default that happens to be unavailable on some engine is a
    silently dead probe. LOCAL_FIELDS are always safe, so they come first and
    `pressure` is added when the engine offers it.
    """
    names = list(LOCAL_FIELDS)
    available = sim.get_field_names()
    if "pressure" in available:
        names.append("pressure")
    return names


def validate_probe_fields(sim: SimEngine, spec: ProbeSpec) -> list[str]:
    """Requested fields this engine cannot record. Empty means the probe is sound.

    Probe fields reach a scene two ways that can go stale: hand-written or
    edited JSON, and a scene carrying fields from an engine the user has since
    swapped out. Both are invisible until the trace just sits flat.
    """
    available = set(recordable_fields(sim))
    return [f for f in spec.fields if f not in available]


def validate_scene_probes(sim: SimEngine, scene) -> list[str]:
    """Human-readable problems for every probe in a scene. Empty means fine."""
    problems = []
    for index, spec in enumerate(getattr(scene, "probes", []) or []):
        for field in validate_probe_fields(sim, spec):
            label = spec.name or f"probe {index + 1}"
            problems.append(
                f"{label}: field {field!r} is not available on "
                f"{type(sim).__name__} and will not be recorded"
            )
    return problems


class Probe:
    def __init__(self, spec: ProbeSpec) -> None:
        self.spec = spec
        self.history: dict[str, deque[float]] = {
            f: deque(maxlen=MAX_HISTORY) for f in spec.fields
        }
        # The simulation step each sample was taken at, parallel to `history`.
        #
        # A probe does not sample every step: the analysis panel records on every
        # second tick, so consecutive samples are two simulation steps apart. The
        # plots labelled their x-axis "step" while plotting the *sample index*, and
        # `strouhal_number` was told dt=1.0, so every sample was treated as one step
        # apart. Both read a cadence of 1 where the truth is 2, which puts every
        # displayed time axis and every reported frequency a factor of two out.
        #
        # Keeping the step index beside the data is what lets a consumer recover the
        # real spacing instead of assuming it. Recording it is the only way to know.
        self.steps: deque[int] = deque(maxlen=MAX_HISTORY)
        self._step = 0
        self._unavailable: set[str] = set()

    def _read_local(self, name: str, u_val: float, v_val: float) -> float:
        if name == "u":
            return u_val
        if name == "v":
            return v_val
        return float(np.hypot(u_val, v_val))

    def record(
        self,
        sim: SimEngine,
        cache: dict[str, np.ndarray] | None = None,
        step: int | None = None,
    ) -> None:
        """Append one sample per requested field.

        `cache` is an optional per-step scratch dict shared across probes. A
        field is a whole-domain array, so without it N probes on the same field
        would each recompute it -- and get_field normalises against a running
        max, so recomputing is not merely wasteful, it perturbs the value the
        next probe sees.

        `step` is the simulation step this sample belongs to. Pass it whenever the
        caller knows it; the fallback keeps the sample counter, which is only
        right when the probe happens to sample every step.
        """
        y, x = self.spec.y, self.spec.x
        h, w = sim.grid_shape
        y = int(np.clip(y, 0, h - 1))
        x = int(np.clip(x, 0, w - 1))

        u_val, v_val = sim.get_velocity_at(x, y)
        recorded = False

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
            recorded = True

        if recorded:
            self.steps.append(self._step if step is None else int(step))
        self._step += 1

    @property
    def step(self) -> int:
        return self._step

    def sample_interval(self) -> int | None:
        """Simulation steps between consecutive samples, or None if unknowable.

        The median of the observed gaps, because a probe that missed a few samples
        (a paused window, a slow frame) would otherwise report an average cadence
        that matches no actual interval.
        """
        if len(self.steps) < 2:
            return None
        gaps = np.diff(np.asarray(self.steps, dtype=float))
        interval = float(np.median(gaps))
        if interval <= 0:
            return None
        return int(interval)

    def times(self) -> list[float]:
        """Recorded step indices, for consumers that need the sample cadence."""
        return list(self.steps)

    @property
    def unavailable_fields(self) -> set[str]:
        """Fields this probe asked for but the engine cannot produce."""
        return set(self._unavailable)

    def clear(self) -> None:
        for f in self.spec.fields:
            self.history[f].clear()
        self.steps.clear()
        self._step = 0
        self._unavailable.clear()
