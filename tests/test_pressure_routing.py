"""Every engine renders pressure through the one canonical definition.

PLAN.md C36 (P1-2), following C35. C35 defined the quantity; C36 routes the
per-engine display paths through it.

Six backends computed their own version of the same thing inline:

    LBM2D                get_pressure_gauge()          (rho - 1) / 3
    LBM3D                self.rho - 1.0                          x3 too big
    LBM2DGPU             get_density() - 1.0                     x3 too big
    LBM2DLettuce         get_density() - 1.0                     x3 too big
    LBM2DLiquid          self.rho - 1.0                           x3 too big
    LBM2DMultiComponent  self.rho1 + self.rho2 - 1.0              x3 too big
    LBM3DGPU             get_density() - 1.0                      x3 too big

A user switching from the CPU engine to the GPU engine saw the same pressure field
appear to triple, with no indication that anything had changed: both renders land in
[0, 1] after the symmetric EMA normalization, so the image is identical in extent and
only the implied scale differs. Nothing in the UI compares them, so nothing reported
it.

`/pressure` had the mirror-image problem: it returned the absolute `rho / 3`, carrying
the reference state's 1/3 offset with nothing in the response to say so.

The GPU engines cannot be instantiated without CUDA, so their source is asserted
rather than executed -- which is why the check below is a source assertion and not a
behavioural one.
"""

from __future__ import annotations

import ast
import inspect
import textwrap
from pathlib import Path

import numpy as np
import pytest

from engines.lbm2d import LBM2D
from engines.lbm2d_liquid import LBM2DLiquid
from engines.lbm2d_multicomponent import LBM2DMultiComponent
from engines.lbm3d import LBM3D

_ENGINES_DIR = Path(__file__).resolve().parents[1] / "engines"

#: Engines whose `get_field("pressure")` source can be executed here.
_CPU_ENGINES = (LBM2D, LBM3D, LBM2DLiquid, LBM2DMultiComponent)

#: Engines that need CUDA, asserted at the source level.
_GPU_ENGINES = ("lbm2d_gpu.py", "lbm3d_gpu.py", "lbm2d_lettuce.py")


def _instantiate(cls):
    if cls is LBM3D:
        return cls(width=12, height=12, depth=6)
    return cls(width=24, height=24)


def _pressure_branch(tree: ast.AST) -> ast.If | None:
    """The `if name == "pressure":` branch inside `get_field`."""
    for node in ast.walk(tree):
        if not isinstance(node, ast.If):
            continue
        test = ast.unparse(node.test)
        if test in ('name == "pressure"', "name == 'pressure'"):
            return node
    return None


def _density_gradient(sim, lo: float = 0.99, hi: float = 1.01) -> np.ndarray:
    """A density field that varies only along the first axis, over `sim.grid_shape`.

    A gradient rather than a uniform value, because the signed normalization is only
    observable where the field has both signs.
    """
    shape = sim.grid_shape
    rows = np.linspace(lo, hi, shape[0]).reshape(-1, *([1] * (len(shape) - 1)))
    return np.ascontiguousarray(np.broadcast_to(rows, shape).astype(np.float64))


def _set_total_density(sim, value) -> None:
    """Write a total density, whatever the engine's density storage is."""
    value = np.asarray(value, dtype=np.float64)
    if isinstance(sim, LBM2DMultiComponent):
        sim.rho1[:] = value / 2.0
        sim.rho2[:] = value / 2.0
    else:
        sim.rho[:] = value


def _settle(sim, field: str = "pressure", iters: int = 250) -> np.ndarray:
    """Render until the EMA smoothing has converged, then return the field.

    The render divides by an exponentially smoothed running maximum, so a single
    call is a function of the *history* of inputs rather than of the current one.
    Comparing raw first calls between engines would therefore measure the shared
    initial `_ema_pres_max = 0.001` rather than the field.
    """
    out = None
    for _ in range(iters):
        out = sim.get_field(field)
    assert out is not None
    return out


class TestEveryEngineCallsTheCanonicalMethod:
    @pytest.mark.parametrize("filename", sorted(_GPU_ENGINES))
    def test_gpu_engines_source(self, filename):
        """Source-level, because these cannot be instantiated without CUDA."""
        tree = ast.parse((_ENGINES_DIR / filename).read_text())
        branch = _pressure_branch(tree)
        assert branch is not None, f"{filename} has no pressure branch"
        calls = {
            node.func.attr
            for node in ast.walk(branch)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
        }
        assert "get_pressure_gauge" in calls, (
            f"{filename} still derives the pressure display inline instead of "
            f"calling get_pressure_gauge(); it was {sorted(calls)}"
        )

    @pytest.mark.parametrize("cls", _CPU_ENGINES, ids=lambda c: c.__name__)
    def test_no_inline_density_deviation_remains(self, cls):
        """The specific expression that was three times too large, gone everywhere.

        Asserted over the whole `get_field` body rather than only the pressure
        branch: `rho - 1` is legitimate as a *density* rendering in some models, and
        the only reliable way to be sure none survived is to look for the arithmetic
        anywhere in the method.
        """
        tree = ast.parse(textwrap.dedent(inspect.getsource(cls.get_field)))
        for node in ast.walk(tree):
            if not isinstance(node, ast.BinOp) or not isinstance(node.op, ast.Sub):
                continue
            # `self.rho - lo` is the legitimate min-max density normalization;
            # `self.rho - 1` is the deviation that started all this.
            if not (
                isinstance(node.right, ast.Constant) and node.right.value in (1, 1.0)
            ):
                continue
            left = ast.unparse(node.left)
            if left in ("self.rho", "self.get_density()", "total"):
                raise AssertionError(
                    f"{cls.__name__}.get_field still subtracts 1 from {left} "
                    "inline: that is the density deviation, three times the "
                    "gauge pressure"
                )

    def test_the_only_density_minus_one_left_is_the_multicomponent_total(self):
        """`rho1 + rho2 - 1` *is* the total density deviation, and the base class
        turns it into the gauge pressure for free. Kept as an explicit note rather
        than deleted, because deleting it would mean duplicating the equation of
        state in a display function."""
        assert "(self.rho1 + self.rho2 - 1.0)" not in inspect.getsource(
            LBM2DMultiComponent.get_field
        )
        sim = _instantiate(LBM2DMultiComponent)
        sim.rho1[:] = 0.4
        sim.rho2[:] = 0.7
        assert np.allclose(sim.get_pressure_gauge(), (1.1 - 1.0) / 3)


class TestTheRenderedFieldIsAFunctionOfTheGaugeOnly:
    """With the EMA settled, the render must be `p' / max|p'| * 0.5 + 0.5` of the
    *gauge* pressure — the definition, written down where a change to it fails.

    Worth pinning even though a later test shows the mapping is scale-invariant:
    this one fails if the sign convention or the mid-grey anchor moves.
    """

    @pytest.mark.parametrize("cls", _CPU_ENGINES, ids=lambda c: c.__name__)
    def test_render_is_the_signed_normalization_of_the_gauge(self, cls):
        sim = _instantiate(cls)
        _set_total_density(sim, _density_gradient(sim, 0.98, 1.02))

        gauge = sim.get_pressure_gauge()
        expected = np.clip(
            gauge / max(float(np.max(np.abs(gauge))), 1e-6) * 0.5 + 0.5, 0, 1
        )
        assert np.allclose(_settle(sim), expected, atol=2e-3)

    @pytest.mark.parametrize("cls", _CPU_ENGINES, ids=lambda c: c.__name__)
    def test_the_field_straddles_mid_grey(self, cls):
        """A signed quantity rendered on a diverging colour map must straddle 0.5.
        A one-sided render means the sign is being lost somewhere."""
        sim = _instantiate(cls)
        _set_total_density(sim, _density_gradient(sim))
        field = _settle(sim)
        assert float(field.min()) < 0.45
        assert float(field.max()) > 0.55

    def test_the_render_cannot_detect_the_factor_of_three(self):
        """Why the source assertions above are load-bearing rather than a fallback.

        The signed normalization divides by `max|p'|`, so multiplying the input by
        three leaves the rendered image *identical*, to the last bit. The rendered
        field is scale-invariant, which means the factor-of-3 divergence was never
        visible on screen and could never have been spotted by looking at it. Only
        the numeric path -- probes, statistics, contours, the colorbar, the REST
        endpoint -- can observe it, and only the numeric assertions below can catch
        it. This test pins that blindness so the source checks do not look paranoid.
        """
        sim = LBM2D(width=32, height=32)
        _set_total_density(sim, _density_gradient(sim))
        correct = _settle(sim).copy()

        sim.rho[:] = (sim.rho - 1.0) * 3.0 + 1.0  # the old, three-times-larger bug
        wrong = _settle(sim).copy()

        assert np.allclose(correct, wrong, atol=1e-6)
        # ...while the underlying quantity differs by exactly the factor.
        assert np.allclose(sim.get_pressure_gauge(), (sim.rho - 1.0) / 3.0)

    def test_the_same_density_gives_the_same_numeric_pressure_on_every_engine(
        self,
    ):
        """The property the whole item exists to restore, checked where it is
        actually observable: identical physics, identical numbers, whichever backend
        produced them."""
        values = {}
        for cls in _CPU_ENGINES:
            sim = _instantiate(cls)
            _set_total_density(sim, _density_gradient(sim))
            gauge = sim.get_pressure_gauge()
            values[cls.__name__] = (
                float(np.max(gauge)),
                float(np.min(gauge)),
            )

        reference = values["LBM2D"]
        for name, pair in values.items():
            assert pair == pytest.approx(
                reference, rel=1e-5
            ), f"{name} reports a different pressure for the same density field"

    def test_the_render_is_still_shared_across_engines(self):
        """Pinned because it is the property a user sees when switching backends.

        Note what it does *not* prove: the render is scale-invariant, so it is
        satisfied by the buggy code too. See the test above.
        """
        renders = {}
        for cls in _CPU_ENGINES:
            sim = _instantiate(cls)
            _set_total_density(sim, _density_gradient(sim))
            field = _settle(sim)
            # Flatten to the first axis so a 2D and a 3D engine are comparable.
            renders[cls.__name__] = np.asarray(field).reshape(-1)[:24]

        reference = renders["LBM2D"]
        for name, field in renders.items():
            assert np.allclose(
                field, reference, atol=2e-3
            ), f"{name} renders the same density field differently from LBM2D"

    def test_a_resting_field_is_uniform_mid_grey(self):
        """The sign the EMA divides by is the sign of the gauge pressure, so at rest
        the field must be flat at 0.5 -- not dark blue at -1 or bright red."""
        sim = LBM2D(width=32, height=32)
        sim.rho[:] = 1.0
        assert np.allclose(sim.get_field("pressure"), 0.5, atol=1e-6)


class TestRestEndpointsAgree:
    """`/pressure` returned absolute pressure; every other pressure surface in the
    product returns the gauge."""

    def test_the_endpoint_returns_the_gauge(self):
        pytest.importorskip("fastapi")

        from engines.api import create_app

        source = inspect.getsource(create_app)
        assert "get_pressure_gauge()" in source
        assert "sim.get_pressure()" not in source

    def test_the_response_declares_which_quantity_it_is(self):
        from engines.api import create_app

        tree = ast.parse(textwrap.dedent(inspect.getsource(create_app)))
        branch = next(
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.FunctionDef) and node.name == "get_pressure"
        )
        keys = {
            ast.literal_eval(node)
            for node in ast.walk(branch)
            if isinstance(node, ast.Constant) and isinstance(node.value, str)
        }
        assert {"quantity", "unit", "reference_pressure"} <= keys, (
            "an endpoint returning a gauge pressure must say so; a client cannot "
            "infer the convention from the numbers"
        )
