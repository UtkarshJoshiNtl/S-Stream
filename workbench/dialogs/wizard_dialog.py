"""Guided Setup Wizard — template selector for first-time users.

Shows categorized flow templates so new users never see a blank screen.
Each template auto-populates the scene with geometry, parameters, probes,
and an optional autorun demo.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from scene.scene import (
    CircleObstacle,
    EmitterSpec,
    RectObstacle,
    Scene,
    SceneProductMeta,
)


@dataclass
class WizardTemplate:
    """A self-contained scene template for the wizard."""

    name: str
    category: str
    description: str
    icon: str
    scene: Scene
    tips: list[str] = field(default_factory=list)


def _build_templates() -> list[WizardTemplate]:
    """Build all wizard templates."""
    return [
        # --- Study Flow Physics ---
        WizardTemplate(
            name="Vortex Shedding",
            category="Study Flow Physics",
            description="Observe periodic vortex shedding behind a cylinder. "
            "Classic Karman vortex street at Re~200.",
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
            tips=[
                "Watch the vortices alternate side-to-side",
                "Try changing viscosity to see laminar vs turbulent shedding",
                "Switch to vorticity colormap to see the pattern clearly",
            ],
        ),
        WizardTemplate(
            name="Lid-Driven Cavity",
            category="Study Flow Physics",
            description="The classic benchmark: a square cavity with a moving lid. "
            "Produces a primary vortex and corner eddies.",
            icon="🔲",
            scene=Scene(
                name="Lid-Driven Cavity",
                width=128,
                height=128,
                viscosity=0.01,
                u_inflow=0.0,
                obstacles=[],
                emitters=[EmitterSpec(name="Smoke", x=2, y=2, strength=0.05)],
                product=SceneProductMeta(
                    recommended_colormap="speed",
                    autorun_steps=3000,
                    lesson_headline=(
                        "Classic benchmark: primary vortex" " and corner eddies"
                    ),
                ),
            ),
            tips=[
                "Speed colormap shows the primary vortex clearly",
                "Try Re=100 (viscosity=0.01) vs Re=1000 (viscosity=0.001)",
                "Compare with Ghia et al. (1982) benchmark data",
            ],
        ),
        WizardTemplate(
            name="Backward-Facing Step",
            category="Study Flow Physics",
            description="Flow separation and reattachment behind a step. "
            "Classic test for turbulence and separation models.",
            icon="📐",
            scene=Scene(
                name="Backward-Facing Step",
                width=256,
                height=128,
                viscosity=0.002,
                u_inflow=0.1,
                obstacles=[
                    RectObstacle(name="Step", x=0, y=0, w=64, h=64),
                ],
                emitters=[EmitterSpec(name="Smoke", x=2, y=96, strength=0.05)],
                product=SceneProductMeta(
                    recommended_colormap="vorticity",
                    autorun_steps=5000,
                    lesson_headline="Separation and reattachment at the step",
                ),
            ),
            tips=[
                "Watch the recirculation zone form behind the step",
                "Try different viscosities to see the reattachment length change",
                "Vorticity colormap shows the shear layer clearly",
            ],
        ),
        WizardTemplate(
            name="Bluff Body Drag",
            category="Study Flow Physics",
            description="Flow around a circular obstacle. Study drag, wake "
            "structure, and pressure distribution.",
            icon="💨",
            scene=Scene(
                name="Bluff Body Drag",
                width=192,
                height=128,
                viscosity=0.003,
                u_inflow=0.12,
                obstacles=[CircleObstacle(name="Cylinder", x=64, y=64, radius=10)],
                emitters=[EmitterSpec(name="Smoke", x=2, y=64, strength=0.04)],
                product=SceneProductMeta(
                    recommended_colormap="pressure",
                    autorun_steps=4000,
                    lesson_headline=(
                        "Pressure distribution and wake" " behind a bluff body"
                    ),
                ),
            ),
            tips=[
                "Pressure colormap shows high pressure upstream, low downstream",
                "Force arrows show drag direction and magnitude",
                "Try different obstacle sizes to study Re dependence",
            ],
        ),
        WizardTemplate(
            name="Channel Flow",
            category="Study Flow Physics",
            description="Fully developed flow in a channel. Poiseuille flow "
            "with parabolic velocity profile.",
            icon="🌊",
            scene=Scene(
                name="Channel Flow",
                width=256,
                height=64,
                viscosity=0.01,
                u_inflow=0.1,
                obstacles=[],
                emitters=[EmitterSpec(name="Smoke", x=2, y=32, strength=0.06)],
                product=SceneProductMeta(
                    recommended_colormap="speed",
                    autorun_steps=3000,
                    lesson_headline="Parabolic velocity profile in channel flow",
                ),
            ),
            tips=[
                "Speed shows the parabolic profile developing",
                "Add a probe to measure velocity at different heights",
                "Compare with analytical Poiseuille solution",
            ],
        ),
        WizardTemplate(
            name="Nozzle & Diffuser",
            category="Study Flow Physics",
            description="Converging-diverging channel. Study flow acceleration "
            "and deceleration, pressure recovery.",
            icon="🔔",
            scene=Scene(
                name="Nozzle & Diffuser",
                width=256,
                height=128,
                viscosity=0.005,
                u_inflow=0.1,
                obstacles=[
                    RectObstacle(name="Top Wall", x=80, y=0, w=96, h=32),
                    RectObstacle(name="Bottom Wall", x=80, y=96, w=96, h=32),
                ],
                emitters=[EmitterSpec(name="Smoke", x=2, y=64, strength=0.05)],
                product=SceneProductMeta(
                    recommended_colormap="speed",
                    autorun_steps=4000,
                    lesson_headline=(
                        "Flow acceleration in nozzle," " deceleration in diffuser"
                    ),
                ),
            ),
            tips=[
                "Speed shows acceleration in the narrow section",
                "Pressure drops in the nozzle, recovers in the diffuser",
                "Try different constriction widths",
            ],
        ),
        WizardTemplate(
            name="Porous Screen",
            category="Study Flow Physics",
            description="Flow through a periodic lattice structure. Study "
            "pressure drop and flow resistance in porous media.",
            icon="🧱",
            scene=Scene(
                name="Porous Screen",
                width=192,
                height=128,
                viscosity=0.005,
                u_inflow=0.1,
                obstacles=[],
                emitters=[EmitterSpec(name="Smoke", x=2, y=64, strength=0.05)],
                product=SceneProductMeta(
                    recommended_colormap="speed",
                    autorun_steps=4000,
                    lesson_headline="Pressure drop and flow through porous media",
                ),
            ),
            tips=[
                "Speed shows flow acceleration through pores",
                "Pressure drops across the lattice",
                "Try different lattice cell sizes",
            ],
        ),
        # --- Create & Experiment ---
        WizardTemplate(
            name="Blank Canvas",
            category="Create & Experiment",
            description="Start empty. Add obstacles, emitters, and probes "
            "using the toolbar. Full creative control.",
            icon="🎨",
            scene=Scene(
                name="Blank Canvas",
                width=128,
                height=128,
                viscosity=0.02,
                u_inflow=0.15,
                emitters=[EmitterSpec(name="Inlet", x=2, y=64, strength=0.05)],
            ),
            tips=[
                "Use toolbar to draw circles, rectangles, or polygons",
                "Add emitters to inject smoke",
                "Place probes to measure local flow quantities",
            ],
        ),
        WizardTemplate(
            name="Two Cylinders",
            category="Create & Experiment",
            description="Two cylinders in tandem. Study interference effects "
            "and wake interaction between bodies.",
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
                product=SceneProductMeta(
                    recommended_colormap="vorticity",
                    autorun_steps=5000,
                    lesson_headline="Wake interaction between tandem cylinders",
                ),
            ),
            tips=[
                "Watch how the front cylinder's wake affects the rear one",
                "Try different spacing between cylinders",
                "Force arrows show drag on each cylinder",
            ],
        ),
        # --- Learn LBM ---
        WizardTemplate(
            name="What is LBM?",
            category="Learn Lattice Boltzmann",
            description="Interactive introduction to how Lattice Boltzmann "
            "Method works. See particles stream and collide.",
            icon="📖",
            scene=Scene(
                name="What is LBM?",
                width=128,
                height=128,
                viscosity=0.02,
                u_inflow=0.1,
                emitters=[EmitterSpec(name="Smoke", x=2, y=64, strength=0.08)],
                product=SceneProductMeta(
                    recommended_colormap="smoke",
                    autorun_steps=2000,
                    lesson_headline="How LBM simulates fluid flow",
                ),
            ),
            tips=[
                "LBM tracks probability distributions of虚拟 particles",
                "Streaming moves particles to neighbors",
                "Collision relaxes toward equilibrium (BGK model)",
            ],
        ),
    ]


class _TemplateCard(QFrame):
    """Clickable card representing a single wizard template."""

    clicked = Signal(object)  # emits WizardTemplate

    def __init__(self, template: WizardTemplate, parent=None):
        super().__init__(parent)
        self._template = template
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(230, 200)
        self.setStyleSheet(
            "QFrame { background: #111827; border: 2px solid #334155; "
            "border-radius: 12px; } "
            "QFrame:hover { border: 2px solid #38bdf8; background: #172033; }"
        )

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 12, 14, 10)
        layout.setSpacing(4)

        icon_label = QLabel(template.icon)
        icon_label.setStyleSheet("font-size: 28px;")
        layout.addWidget(icon_label)

        name_label = QLabel(f"<b>{template.name}</b>")
        name_label.setStyleSheet("color: #f8fafc; font-size: 13px;")
        layout.addWidget(name_label)

        desc_label = QLabel(template.description)
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(desc_label, 1)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._template)
        super().mousePressEvent(event)


class WizardDialog(QDialog):
    """Guided setup wizard shown on first launch.

    Emits template_selected(WizardTemplate) when user picks a template,
    or finished() when user skips.
    """

    template_selected = Signal(object)  # WizardTemplate

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Welcome to S-Stream")
        self.setMinimumSize(860, 600)
        self.setModal(True)

        self._templates = _build_templates()

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Title
        title = QLabel("<h1>What would you like to study?</h1>")
        title.setStyleSheet("color: #f8fafc;")
        layout.addWidget(title)

        subtitle = QLabel(
            "Pick a flow scenario to get started instantly. Each template "
            "sets up the geometry, parameters, and probes for you."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #cbd5e1;")
        layout.addWidget(subtitle)

        # Scrollable template grid
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(16)

        # Group by category
        categories: dict[str, list[WizardTemplate]] = {}
        for t in self._templates:
            categories.setdefault(t.category, []).append(t)

        for cat_name, cat_templates in categories.items():
            cat_label = QLabel(f"<h3>{cat_name}</h3>")
            cat_label.setStyleSheet("color: #7dd3fc; margin-top: 8px;")
            main_layout.addWidget(cat_label)

            row_layout = QHBoxLayout()
            row_layout.setSpacing(12)
            for tmpl in cat_templates:
                card = _TemplateCard(tmpl)
                card.clicked.connect(self._on_template_clicked)
                row_layout.addWidget(card)
            row_layout.addStretch()
            main_layout.addLayout(row_layout)

        main_layout.addStretch()
        scroll.setWidget(container)
        layout.addWidget(scroll, 1)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.addStretch()
        skip_btn = QPushButton("Skip — Start Empty")
        skip_btn.clicked.connect(self.reject)
        btn_row.addWidget(skip_btn)
        layout.addLayout(btn_row)

        self.setStyleSheet(
            "QDialog { background: #020617; } "
            "QLabel { color: #e5e7eb; } "
            "QPushButton { color: #f8fafc; background: #1e293b; "
            "border: 1px solid #475569; padding: 8px 20px; border-radius: 6px; "
            "font-size: 13px; } "
            "QPushButton:hover { background: #334155; }"
        )

    def _on_template_clicked(self, template: WizardTemplate) -> None:
        self.template_selected.emit(template)
        self.accept()
