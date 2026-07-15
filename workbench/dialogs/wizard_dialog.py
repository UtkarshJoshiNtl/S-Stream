"""Unified Start Dialog — templates, presets, and recipes in one place.

Replaces the separate WizardDialog, PresetsDialog, and RecipesDialog
with a single tabbed dialog. Shown on first launch and via the Start
toolbar button.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QPixmap
from PySide6.QtWidgets import (
    QDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QPushButton,
    QScrollArea,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from presets.loader import list_presets
from scene.scene import (
    CircleObstacle,
    EmitterSpec,
    LatticeObstacle,
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


RECIPES: dict[str, str] = {
    "Show vortex shedding": (
        "Open Cylinder Wake, run the demo, switch to vorticity, "
        "and watch alternating wake structures."
    ),
    "Compare drag of two shapes": (
        "Draw or load two obstacle scenes, run each to a settled wake, "
        "then compare Cd and wake strength."
    ),
    "Generate Cd vs Re": (
        "Use Sweep Re to vary inlet/viscosity and export the plotted drag trend."
    ),
    "Explain Reynolds number": (
        "Start with Channel Flow, change viscosity, "
        "and watch the Re readout and sanity notes."
    ),
    "Create a lab-report figure": (
        "Run a flagship preset, wait for Demo ready, "
        "then export a report PNG and Markdown summary."
    ),
}


def _build_templates() -> list[WizardTemplate]:
    """Build all wizard templates."""
    return [
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
                viscosity=0.128,
                u_inflow=0.0,
                obstacles=[
                    RectObstacle(name="Left Wall", x=0, y=1, w=2, h=126),
                    RectObstacle(name="Right Wall", x=126, y=1, w=2, h=126),
                ],
                emitters=[EmitterSpec(name="Smoke", x=64, y=64, strength=0.05)],
                description=(
                    "Cavity mode: MovingWall lid is applied when "
                    "domain_mode=cavity (lid_velocity≈0.1)."
                ),
                product=SceneProductMeta(
                    recommended_colormap="speed",
                    autorun_steps=3000,
                    lesson_headline=(
                        "Classic benchmark: primary vortex" " and corner eddies"
                    ),
                ),
            ),
            tips=[
                "Engine switches to cavity mode with MovingWall lid (lid_velocity=0.1)",
                "Re≈100 at viscosity=0.128; try Re≈1000 with viscosity≈0.0128",
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
                obstacles=[
                    LatticeObstacle(
                        name="Screen",
                        x=72,
                        y=16,
                        w=48,
                        h=96,
                        cell_size=8,
                        wall_thickness=1,
                    ),
                ],
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
                "LBM tracks probability distributions of virtual particles",
                "Streaming moves particles to neighbors",
                "Collision relaxes toward equilibrium (BGK model)",
            ],
        ),
    ]


# --- Card widgets ---


class _TemplateCard(QFrame):
    """Clickable card representing a single wizard template."""

    clicked = Signal(object)

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


class _PresetCard(QWidget):
    """Clickable card for a preset scene file."""

    clicked = Signal(str)

    def __init__(self, preset: dict, parent=None):
        super().__init__(parent)
        self._file = preset["file"]
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(6)

        self.setMinimumSize(220, 190)
        self.setMaximumSize(260, 230)
        self.setStyleSheet(
            "QWidget { background: #111827; border: 1px solid #334155; "
            "border-radius: 10px; } "
            "QWidget:hover { border: 1px solid #38bdf8; background: #172033; }"
        )

        thumb_label = QLabel()
        thumb_path = preset.get("thumbnail", "")
        if thumb_path and Path(thumb_path).exists():
            pix = QPixmap(thumb_path).scaled(
                230,
                86,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation,
            )
            thumb_label.setPixmap(pix)
        else:
            thumb_label.setText("flow")
            thumb_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            thumb_label.setStyleSheet("font-size: 24px; color: #38bdf8;")
        thumb_label.setMinimumHeight(72)
        layout.addWidget(thumb_label)

        name_label = QLabel(f"<b>{preset['name']}</b>")
        name_label.setStyleSheet("color: #f8fafc; font-size: 14px;")
        layout.addWidget(name_label)

        headline = (
            preset.get("headline") or preset.get("recipe") or "Ready-made flow story"
        )
        headline_label = QLabel(headline)
        headline_label.setWordWrap(True)
        headline_label.setStyleSheet("color: #7dd3fc; font-size: 11px;")
        layout.addWidget(headline_label)

        desc = preset["description"]
        desc_label = QLabel(desc[:92] + ("..." if len(desc) > 92 else ""))
        desc_label.setWordWrap(True)
        desc_label.setStyleSheet("color: #94a3b8; font-size: 11px;")
        layout.addWidget(desc_label)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.MouseButton.LeftButton:
            self.clicked.emit(self._file)


# --- Unified dialog ---


class StartDialog(QDialog):
    """Unified start dialog with three tabs: Flow Stories, Presets, Recipes."""

    template_selected = Signal(object)
    preset_selected = Signal(str)
    recipe_selected = Signal(str)

    def __init__(self, parent=None, tab: int = 0):
        super().__init__(parent)
        self.setWindowTitle("Start — S-Stream")
        self.setMinimumSize(860, 620)
        self.setModal(True)

        self._templates = _build_templates()

        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        title = QLabel("<h1>Start</h1>")
        title.setStyleSheet("color: #f8fafc;")
        layout.addWidget(title)

        tabs = QTabWidget()
        tabs.addTab(self._build_stories_tab(), "Flow Stories")
        tabs.addTab(self._build_presets_tab(), "Preset Gallery")
        tabs.addTab(self._build_recipes_tab(), "Recipes")
        tabs.setCurrentIndex(max(0, min(tab, 2)))
        layout.addWidget(tabs, 1)

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
            "QTabWidget::pane { border: 1px solid #334155; background: #020617; } "
            "QTabBar::tab { background: #1e293b; color: #94a3b8; "
            "padding: 8px 16px; border: 1px solid #334155; "
            "border-bottom: none; border-radius: 6px 6px 0 0; } "
            "QTabBar::tab:selected { background: #020617; color: #f8fafc; }"
        )

    def set_active_tab(self, index: int) -> None:
        """Programmatically select the active tab (0=Stories, 1=Presets, 2=Recipes)."""
        parent = self.parent()
        if parent is not None:
            for child in self.findChildren(QTabWidget):
                child.setCurrentIndex(max(0, min(index, 2)))
                break

    def _build_stories_tab(self) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        main_layout = QVBoxLayout(container)
        main_layout.setSpacing(16)

        subtitle = QLabel(
            "Pick a scenario to begin. Each template sets up geometry, "
            "parameters, and a short guided demo for you."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #cbd5e1;")
        main_layout.addWidget(subtitle)

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
        return scroll

    def _build_presets_tab(self) -> QWidget:
        presets = list_presets()

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setStyleSheet("QScrollArea { background: transparent; }")

        container = QWidget()
        container.setStyleSheet("background: transparent;")
        outer = QVBoxLayout(container)
        outer.setSpacing(10)

        subtitle = QLabel(
            "Pick a saved scene: S-Stream will choose the view, run the flow, "
            "explain what is happening, and help you export a figure."
        )
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #cbd5e1;")
        outer.addWidget(subtitle)

        if not presets:
            no_presets = QLabel("No presets found.")
            no_presets.setStyleSheet("color: #64748b;")
            outer.addWidget(no_presets)
        else:
            grid_container = QWidget()
            grid_container.setStyleSheet("background: transparent;")
            grid = QGridLayout(grid_container)
            grid.setSpacing(14)
            for i, preset in enumerate(presets):
                card = _PresetCard(preset)
                card.clicked.connect(self._on_preset_clicked)
                row, col = divmod(i, 3)
                grid.addWidget(card, row, col)
            outer.addWidget(grid_container)

        outer.addStretch()
        scroll.setWidget(container)
        return scroll

    def _build_recipes_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setSpacing(8)

        subtitle = QLabel("Choose a guided workflow to follow step-by-step.")
        subtitle.setWordWrap(True)
        subtitle.setStyleSheet("color: #cbd5e1;")
        layout.addWidget(subtitle)

        self._recipe_list = QListWidget()
        for name in RECIPES:
            self._recipe_list.addItem(name)
        self._recipe_list.currentTextChanged.connect(self._show_recipe)
        layout.addWidget(self._recipe_list)

        self._recipe_details = QLabel("Select a recipe to see the workflow.")
        self._recipe_details.setWordWrap(True)
        layout.addWidget(self._recipe_details)

        use_btn = QPushButton("Use This Recipe")
        use_btn.clicked.connect(self._accept_recipe)
        layout.addWidget(use_btn)

        widget.setStyleSheet(
            "QWidget { background: transparent; } "
            "QListWidget { background: #0b1020; color: #e5e7eb; "
            "border: 1px solid #374151; border-radius: 4px; } "
            "QPushButton { background: #2563eb; color: white; "
            "padding: 8px; border-radius: 5px; }"
        )
        return widget

    def _on_template_clicked(self, template: WizardTemplate) -> None:
        self.template_selected.emit(template)
        self.accept()

    def _on_preset_clicked(self, file_path: str) -> None:
        self.preset_selected.emit(file_path)
        self.accept()

    def _show_recipe(self, name: str) -> None:
        self._recipe_details.setText(RECIPES.get(name, ""))

    def _accept_recipe(self) -> None:
        item = self._recipe_list.currentItem()
        if item:
            self.recipe_selected.emit(item.text())
        self.accept()
