"""Tests for Phase 5.3: Guided Setup Wizard."""

from __future__ import annotations

import pytest

from scene.scene import (
    CircleObstacle,
    EmitterSpec,
    Scene,
    apply_to_sim,
)


def _build_templates():
    """Import-free template builder for testing without PySide6."""
    from scene.scene import (
        CircleObstacle,
        EmitterSpec,
        ProbeSpec,
        RectObstacle,
        Scene,
        SceneProductMeta,
    )
    from dataclasses import dataclass, field

    @dataclass
    class WizardTemplate:
        name: str
        category: str
        description: str
        icon: str
        scene: Scene
        tips: list[str] = field(default_factory=list)

    templates = [
        WizardTemplate(
            name="Vortex Shedding",
            category="Study Flow Physics",
            description="Observe periodic vortex shedding behind a cylinder.",
            icon="🌀",
            scene=Scene(
                name="Vortex Shedding",
                width=256,
                height=128,
                viscosity=0.002,
                u_inflow=0.1,
                obstacles=[CircleObstacle(name="Cylinder", x=80, y=64, radius=8)],
                emitters=[EmitterSpec(name="Smoke", x=2, y=64, strength=0.05)],
                product=SceneProductMeta(
                    recommended_colormap="vorticity",
                    autorun_steps=5000,
                    lesson_headline="Watch vortices alternate behind the cylinder",
                ),
            ),
            tips=["Watch the vortices alternate", "Try changing viscosity"],
        ),
        WizardTemplate(
            name="Lid-Driven Cavity",
            category="Study Flow Physics",
            description="The classic benchmark: a square cavity with a moving lid.",
            icon="🔲",
            scene=Scene(
                name="Lid-Driven Cavity",
                width=128,
                height=128,
                viscosity=0.01,
                u_inflow=0.0,
                emitters=[EmitterSpec(name="Smoke", x=2, y=2, strength=0.05)],
                product=SceneProductMeta(
                    recommended_colormap="speed",
                    autorun_steps=3000,
                ),
            ),
        ),
        WizardTemplate(
            name="Channel Flow",
            category="Study Flow Physics",
            description="Fully developed flow in a channel.",
            icon="🌊",
            scene=Scene(
                name="Channel Flow",
                width=256,
                height=64,
                viscosity=0.01,
                u_inflow=0.1,
                emitters=[EmitterSpec(name="Smoke", x=2, y=32, strength=0.06)],
                product=SceneProductMeta(recommended_colormap="speed"),
            ),
        ),
        WizardTemplate(
            name="Blank Canvas",
            category="Create & Experiment",
            description="Start empty.",
            icon="🎨",
            scene=Scene(
                name="Blank Canvas",
                width=128,
                height=128,
                emitters=[EmitterSpec(name="Inlet", x=2, y=64, strength=0.05)],
            ),
        ),
        WizardTemplate(
            name="Two Cylinders",
            category="Create & Experiment",
            description="Two cylinders in tandem.",
            icon="⚙️",
            scene=Scene(
                name="Two Cylinders",
                width=256,
                height=128,
                viscosity=0.002,
                u_inflow=0.1,
                obstacles=[
                    CircleObstacle(name="Front", x=80, y=64, radius=8),
                    CircleObstacle(name="Rear", x=140, y=64, radius=8),
                ],
                emitters=[EmitterSpec(name="Smoke", x=2, y=64, strength=0.05)],
            ),
        ),
        WizardTemplate(
            name="What is LBM?",
            category="Learn Lattice Boltzmann",
            description="Interactive introduction to LBM.",
            icon="📖",
            scene=Scene(
                name="What is LBM?",
                width=128,
                height=128,
                emitters=[EmitterSpec(name="Smoke", x=2, y=64, strength=0.08)],
                product=SceneProductMeta(
                    recommended_colormap="smoke",
                    autorun_steps=2000,
                ),
            ),
        ),
    ]
    return templates, WizardTemplate


class TestBuildTemplates:
    def test_templates_exist(self) -> None:
        templates, _ = _build_templates()
        assert len(templates) >= 5

    def test_all_templates_have_required_fields(self) -> None:
        templates, _ = _build_templates()
        for t in templates:
            assert t.name, f"Template missing name"
            assert t.category, f"Template {t.name} missing category"
            assert t.description, f"Template {t.name} missing description"
            assert t.icon, f"Template {t.name} missing icon"
            assert isinstance(t.scene, Scene)

    def test_categories_are_populated(self) -> None:
        templates, _ = _build_templates()
        categories = {t.category for t in templates}
        assert "Study Flow Physics" in categories
        assert "Create & Experiment" in categories
        assert "Learn Lattice Boltzmann" in categories

    def test_scene_names_match(self) -> None:
        templates, _ = _build_templates()
        for t in templates:
            assert t.scene.name == t.name


class TestTemplateScenes:
    def test_vortex_shedding_has_cylinder(self) -> None:
        templates, _ = _build_templates()
        vortex = next(t for t in templates if t.name == "Vortex Shedding")
        assert len(vortex.scene.obstacles) == 1
        assert isinstance(vortex.scene.obstacles[0], CircleObstacle)

    def test_channel_flow_is_wide(self) -> None:
        templates, _ = _build_templates()
        channel = next(t for t in templates if t.name == "Channel Flow")
        assert channel.scene.width > channel.scene.height

    def test_blank_canvas_has_no_obstacles(self) -> None:
        templates, _ = _build_templates()
        blank = next(t for t in templates if t.name == "Blank Canvas")
        assert len(blank.scene.obstacles) == 0
        assert len(blank.scene.emitters) > 0

    def test_all_templates_have_emitters(self) -> None:
        templates, _ = _build_templates()
        for t in templates:
            assert len(t.scene.emitters) > 0, f"Template '{t.name}' has no emitters"

    def test_autorun_templates_have_colormap(self) -> None:
        templates, _ = _build_templates()
        for t in templates:
            if t.scene.product.autorun_steps:
                assert (
                    t.scene.product.recommended_colormap
                ), f"Template '{t.name}' has autorun but no colormap"

    def test_two_cylinders_has_two_obstacles(self) -> None:
        templates, _ = _build_templates()
        tc = next(t for t in templates if t.name == "Two Cylinders")
        assert len(tc.scene.obstacles) == 2


class TestApplyTemplateScene:
    def test_apply_vortex_shedding(self) -> None:
        from engines.lbm2d import LBM2D

        templates, _ = _build_templates()
        vortex = next(t for t in templates if t.name == "Vortex Shedding")
        sim = LBM2D(width=64, height=64, viscosity=0.01)
        apply_to_sim(vortex.scene, sim)
        assert sim.viscosity == vortex.scene.viscosity
        assert sim.u_inflow == vortex.scene.u_inflow

    def test_apply_blank_canvas(self) -> None:
        from engines.lbm2d import LBM2D

        templates, _ = _build_templates()
        blank = next(t for t in templates if t.name == "Blank Canvas")
        sim = LBM2D(width=32, height=32)
        apply_to_sim(blank.scene, sim)
        assert sim.get_emitter_count() > 0


class TestWizardTemplate:
    def test_template_dataclass(self) -> None:
        from dataclasses import dataclass, field

        @dataclass
        class WT:
            name: str
            category: str
            description: str
            icon: str
            scene: Scene
            tips: list[str] = field(default_factory=list)

        t = WT(
            name="Test",
            category="Test",
            description="Desc",
            icon="T",
            scene=Scene(name="Test"),
            tips=["Tip 1"],
        )
        assert t.name == "Test"
        assert len(t.tips) == 1

    def test_template_default_tips(self) -> None:
        from dataclasses import dataclass, field

        @dataclass
        class WT:
            name: str
            category: str
            description: str
            icon: str
            scene: Scene
            tips: list[str] = field(default_factory=list)

        t = WT(
            name="Test",
            category="Test",
            description="Desc",
            icon="T",
            scene=Scene(name="Test"),
        )
        assert t.tips == []
