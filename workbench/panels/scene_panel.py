from __future__ import annotations

import copy
from dataclasses import fields
from pathlib import Path

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QDoubleSpinBox,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMenu,
    QPushButton,
    QSpinBox,
    QTreeWidget,
    QTreeWidgetItem,
    QVBoxLayout,
    QWidget,
)

from engines.base import SimEngine
from scene.scene import (
    AirfoilObstacle,
    ChannelObstacle,
    CircleObstacle,
    EmitterSpec,
    EllipseObstacle,
    ImageObstacle,
    LatticeObstacle,
    ObstacleSpec,
    ProbeSpec,
    RectObstacle,
    Scene,
    STLObstacle,
    apply_to_sim,
)

_CATEGORY_KEY = {0: "obstacles", 1: "emitters", 2: "probes"}

_OBSTACLE_DEFAULTS: dict[str, ObstacleSpec] = {
    "Circle": CircleObstacle(name="Circle", x=32, y=32, radius=8),
    "Rectangle": RectObstacle(name="Rect", x=20, y=20, w=16, h=16),
    "Ellipse": EllipseObstacle(name="Ellipse", x=32, y=32, rx=10, ry=6),
    "Airfoil": AirfoilObstacle(
        name="Airfoil", x=40, y=32, chord=24, angle_of_attack=5.0, naca_code="0012"
    ),
    "Channel": ChannelObstacle(
        name="Channel", x=20, y=40, w=60, h=40, inlet_ratio=0.6, outlet_ratio=1.0
    ),
    "Lattice": LatticeObstacle(
        name="Lattice", x=20, y=20, w=40, h=40, cell_size=8, wall_thickness=1
    ),
}


class _PropEditor(QWidget):
    changed = Signal()

    def __init__(self, obj: object, parent=None):
        super().__init__(parent)
        self._obj = obj
        layout = QFormLayout(self)
        self._widgets: dict[str, QWidget] = {}
        for f in fields(obj):
            val = getattr(obj, f.name)
            if f.name == "name":
                w = QLineEdit(val)
                w.textChanged.connect(lambda t, n=f.name: self._set(n, t))
            elif f.name in ("fields", "points", "path"):
                if f.name == "path":
                    w = QLineEdit(str(val))
                    w.setReadOnly(True)
                    layout.addRow(f.name, w)
                    self._widgets[f.name] = w
                    continue
                continue
            elif isinstance(val, bool):
                from PySide6.QtWidgets import QCheckBox

                w = QCheckBox()
                w.setChecked(val)
                w.toggled.connect(lambda v, n=f.name: self._set(n, v))
            elif isinstance(val, int):
                w = QSpinBox()
                w.setRange(0, 1023)
                w.setValue(val)
                w.valueChanged.connect(lambda v, n=f.name: self._set(n, v))
            elif isinstance(val, float):
                w = QDoubleSpinBox()
                w.setRange(-10.0, 10.0)
                w.setSingleStep(0.001)
                w.setDecimals(4)
                w.setValue(val)
                w.valueChanged.connect(lambda v, n=f.name: self._set(n, v))
            else:
                continue
            layout.addRow(f.name, w)
            self._widgets[f.name] = w

    def _set(self, name: str, value) -> None:
        setattr(self._obj, name, value)
        self.changed.emit()


class ScenePanel(QWidget):
    scene_changed = Signal()
    parameters_changed = Signal()

    def __init__(self, sim: SimEngine, scene: Scene, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.scene = scene
        self._current_editor: _PropEditor | None = None
        self._current_item: QTreeWidgetItem | None = None
        self._expert_mode = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)

        # --- Simulation parameters ---
        self._build_parameter_group(layout)

        self.tree = QTreeWidget()
        self.tree.setHeaderLabel("Scene Objects")
        self.tree.currentItemChanged.connect(self._on_item_selected)
        layout.addWidget(self.tree, 1)

        self.props_group = QGroupBox("Properties")
        self.props_layout = QVBoxLayout()
        self._props_placeholder = QLabel("Select an object")
        self.props_layout.addWidget(self._props_placeholder)
        self.props_group.setLayout(self.props_layout)
        layout.addWidget(self.props_group)

        btn_row = QHBoxLayout()
        self.add_btn = QPushButton("Add")
        add_menu = QMenu(self)
        add_menu.addAction("Circle Obstacle", lambda: self._add_obstacle("Circle"))
        add_menu.addAction(
            "Rectangle Obstacle", lambda: self._add_obstacle("Rectangle")
        )
        add_menu.addAction("Ellipse Obstacle", lambda: self._add_obstacle("Ellipse"))
        add_menu.addAction("Airfoil", lambda: self._add_obstacle("Airfoil"))
        add_menu.addAction("Channel", lambda: self._add_obstacle("Channel"))
        add_menu.addAction("Lattice/Porous", lambda: self._add_obstacle("Lattice"))
        add_menu.addSeparator()
        add_menu.addAction("Import STL...", lambda: self._import_stl())
        add_menu.addAction("Import Image...", lambda: self._import_image())
        add_menu.addSeparator()
        add_menu.addAction("Emitter", lambda: self._add_emitter())
        add_menu.addAction("Probe", lambda: self._add_probe())
        self.add_btn.setMenu(add_menu)
        btn_row.addWidget(self.add_btn)

        self.remove_btn = QPushButton("Remove")
        self.remove_btn.clicked.connect(self._remove_selected)
        btn_row.addWidget(self.remove_btn)
        layout.addLayout(btn_row)

        self._rebuild_tree()

    def _build_parameter_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Simulation Parameters")
        form = QFormLayout()
        form.setContentsMargins(4, 4, 4, 4)

        self._param_visc = QDoubleSpinBox()
        self._param_visc.setRange(0.0005, 0.1)
        self._param_visc.setSingleStep(0.001)
        self._param_visc.setDecimals(4)
        self._param_visc.setValue(self.sim.viscosity)
        self._param_visc.valueChanged.connect(self._on_param_visc)
        self._param_visc.setToolTip(
            "Kinematic viscosity (nu). Controls fluid thickness.\n"
            "Lower = thinner fluid, higher Reynolds number.\n"
            "Typical: water ~0.001, air ~0.0001\n"
            "Range: 0.0005 (nearly inviscid) to 0.1 (very viscous)"
        )
        form.addRow("Viscosity", self._param_visc)

        self._param_u_inflow = QDoubleSpinBox()
        self._param_u_inflow.setRange(0.0, 0.5)
        self._param_u_inflow.setSingleStep(0.005)
        self._param_u_inflow.setDecimals(4)
        self._param_u_inflow.setValue(self.sim.u_inflow)
        self._param_u_inflow.valueChanged.connect(self._on_param_u_inflow)
        self._param_u_inflow.setToolTip(
            "Inflow velocity (lattice units). Drives the flow from left to right.\n"
            "Re = u_inflow * characteristic_length / viscosity.\n"
            "Higher values produce faster, more turbulent flow.\n"
            "Range: 0.0 (static) to 0.5 (max stable)"
        )
        form.addRow("Inflow", self._param_u_inflow)

        self._param_diff = QDoubleSpinBox()
        self._param_diff.setRange(0.0, 0.5)
        self._param_diff.setSingleStep(0.001)
        self._param_diff.setDecimals(4)
        self._param_diff.setValue(self.sim.smoke_diffusion)
        self._param_diff.valueChanged.connect(self._on_param_diff)
        self._param_diff.setToolTip(
            "Smoke diffusion coefficient. Controls how fast the passive scalar spreads.\n"
            "Higher values = more diffuse, blurry smoke.\n"
            "Lower values = sharper, more detailed smoke patterns.\n"
            "Range: 0.0 (none) to 0.5 (highly diffuse)"
        )
        form.addRow("Smoke Diffusion", self._param_diff)

        self._param_decay = QDoubleSpinBox()
        self._param_decay.setRange(0.9, 1.0)
        self._param_decay.setSingleStep(0.0005)
        self._param_decay.setDecimals(4)
        self._param_decay.setValue(self.sim.smoke_decay)
        self._param_decay.valueChanged.connect(self._on_param_decay)
        self._param_decay.setToolTip(
            "Smoke decay rate. Controls how fast the passive scalar fades.\n"
            "1.0 = no decay (smoke persists forever).\n"
            "0.999 = slow fade, good for steady-state visualization.\n"
            "0.99 = fast fade, good for transient flow studies.\n"
            "Range: 0.9 (fast) to 1.0 (none)"
        )
        form.addRow("Smoke Decay", self._param_decay)

        group.setLayout(form)
        layout.addWidget(group)

        self._build_particle_group(layout)
        self._build_engine_specific_group(layout)

    def _build_particle_group(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Particles")
        form = QFormLayout()
        form.setContentsMargins(4, 4, 4, 4)

        self._trail_spin = QSpinBox()
        self._trail_spin.setRange(1, 100)
        self._trail_spin.setValue(20)
        self._trail_spin.valueChanged.connect(self._on_trail_length)
        self._trail_spin.setToolTip(
            "Number of past positions kept per particle for trail rendering.\n"
            "Higher = longer trails, more visual memory of the flow path.\n"
            "Range: 1 (dots only) to 100 (very long trails)"
        )
        form.addRow("Trail Length", self._trail_spin)

        btn_row = QHBoxLayout()
        add_random_btn = QPushButton("Add Random")
        add_random_btn.setToolTip("Add 50 particles at random positions in the domain")
        add_random_btn.clicked.connect(self._add_particles_random)
        btn_row.addWidget(add_random_btn)

        clear_btn = QPushButton("Clear All")
        clear_btn.setToolTip("Remove all particles")
        clear_btn.clicked.connect(self._clear_particles)
        btn_row.addWidget(clear_btn)
        form.addRow(btn_row)

        self._particle_count_label = QLabel("0 particles")
        form.addRow(self._particle_count_label)

        group.setLayout(form)
        layout.addWidget(group)

    def _build_engine_specific_group(self, layout: QVBoxLayout) -> None:
        engine_name = type(self.sim).__name__
        if engine_name == "LBM2DLiquid":
            self._build_liquid_params(layout)
        elif engine_name == "LBM2DMultiComponent":
            self._build_multicomponent_params(layout)

    def _build_liquid_params(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Liquid Parameters (Shan-Chen)")
        form = QFormLayout()
        form.setContentsMargins(4, 4, 4, 4)

        self._param_g = QDoubleSpinBox()
        self._param_g.setRange(-15.0, 15.0)
        self._param_g.setSingleStep(0.5)
        self._param_g.setDecimals(2)
        self._param_g.setValue(self.sim.g)
        self._param_g.valueChanged.connect(self._on_param_g)
        self._param_g.setToolTip(
            "Shan-Chen cohesion strength (g).\n"
            "g < 0 gives cohesion → spontaneous liquid/vapor separation.\n"
            "Magnitude controls surface tension and density contrast.\n"
            "Typical: -3 to -8"
        )
        form.addRow("Cohesion (g)", self._param_g)

        self._param_g_adhesion_liquid = QDoubleSpinBox()
        self._param_g_adhesion_liquid.setRange(-15.0, 15.0)
        self._param_g_adhesion_liquid.setSingleStep(0.5)
        self._param_g_adhesion_liquid.setDecimals(2)
        self._param_g_adhesion_liquid.setValue(self.sim.g_adhesion)
        self._param_g_adhesion_liquid.valueChanged.connect(
            self._on_param_g_adhesion_liquid
        )
        self._param_g_adhesion_liquid.setToolTip(
            "Wall adhesion strength (g_adhesion).\n"
            "g < 0 = wetting (liquid clings to walls).\n"
            "g > 0 = non-wetting (liquid repelled from walls).\n"
            "g = 0 = neutral"
        )
        form.addRow("Wall Adhesion", self._param_g_adhesion_liquid)

        self._param_droplet = QDoubleSpinBox()
        self._param_droplet.setRange(0.0, 100.0)
        self._param_droplet.setSingleStep(1.0)
        self._param_droplet.setDecimals(0)
        self._param_droplet.setSpecialValueText("auto (1/4 height)")
        droplet_val = self.sim.droplet_radius if self.sim.droplet_radius else 0.0
        self._param_droplet.setValue(droplet_val)
        self._param_droplet.valueChanged.connect(self._on_param_droplet)
        self._param_droplet.setToolTip(
            "Initial high-density droplet radius (lattice units).\n"
            "0 = auto (1/4 of grid height).\n"
            "The droplet is placed at the domain center and "
            "separates into liquid/vapor over time."
        )
        form.addRow("Droplet Radius", self._param_droplet)

        group.setLayout(form)
        layout.addWidget(group)

    def _build_multicomponent_params(self, layout: QVBoxLayout) -> None:
        group = QGroupBox("Multi-Component Parameters")
        form = QFormLayout()
        form.setContentsMargins(4, 4, 4, 4)

        self._param_g11 = QDoubleSpinBox()
        self._param_g11.setRange(-15.0, 15.0)
        self._param_g11.setSingleStep(0.5)
        self._param_g11.setDecimals(2)
        self._param_g11.setValue(self.sim.g11)
        self._param_g11.valueChanged.connect(self._on_param_g11)
        self._param_g11.setToolTip(
            "Intra-component cohesion for component 1 (e.g. oil).\n"
            "g < 0 = cohesive (forms droplets).\n"
            "Typical: -3 to -8"
        )
        form.addRow("g11 (cohesion 1)", self._param_g11)

        self._param_g22 = QDoubleSpinBox()
        self._param_g22.setRange(-15.0, 15.0)
        self._param_g22.setSingleStep(0.5)
        self._param_g22.setDecimals(2)
        self._param_g22.setValue(self.sim.g22)
        self._param_g22.valueChanged.connect(self._on_param_g22)
        self._param_g22.setToolTip(
            "Intra-component cohesion for component 2 (e.g. water).\n"
            "g < 0 = cohesive (forms droplets).\n"
            "Typical: -3 to -8"
        )
        form.addRow("g22 (cohesion 2)", self._param_g22)

        self._param_g12 = QDoubleSpinBox()
        self._param_g12.setRange(-15.0, 15.0)
        self._param_g12.setSingleStep(0.5)
        self._param_g12.setDecimals(2)
        self._param_g12.setValue(self.sim.g12)
        self._param_g12.valueChanged.connect(self._on_param_g12)
        self._param_g12.setToolTip(
            "Inter-component repulsion strength.\n"
            "g > 0 = repel (immiscible, forms separate phases).\n"
            "g = 0 = miscible (mix freely).\n"
            "g < 0 = attract (unusual).\n"
            "Typical: +3 to +8"
        )
        form.addRow("g12 (repulsion)", self._param_g12)

        self._param_sigma = QDoubleSpinBox()
        self._param_sigma.setRange(0.0, 0.5)
        self._param_sigma.setSingleStep(0.01)
        self._param_sigma.setDecimals(3)
        self._param_sigma.setValue(self.sim.sigma)
        self._param_sigma.valueChanged.connect(self._on_param_sigma)
        self._param_sigma.setToolTip(
            "Color gradient perturbation strength.\n"
            "Sharpens the interface between components.\n"
            "0 = off, 0.05 = moderate, 0.1+ = strong.\n"
            "Too high causes numerical instability."
        )
        form.addRow("Interface (sigma)", self._param_sigma)

        self._param_mc_adhesion = QDoubleSpinBox()
        self._param_mc_adhesion.setRange(-15.0, 15.0)
        self._param_mc_adhesion.setSingleStep(0.5)
        self._param_mc_adhesion.setDecimals(2)
        self._param_mc_adhesion.setValue(self.sim.g_adhesion)
        self._param_mc_adhesion.valueChanged.connect(self._on_param_mc_adhesion)
        self._param_mc_adhesion.setToolTip(
            "Wall adhesion for both components.\n"
            "g < 0 = wetting, g > 0 = non-wetting.\n"
            "Affects both components equally."
        )
        form.addRow("Wall Adhesion", self._param_mc_adhesion)

        group.setLayout(form)
        layout.addWidget(group)

    def _on_param_g(self, val: float) -> None:
        self.sim.g = val
        self.parameters_changed.emit()

    def _on_param_g_adhesion_liquid(self, val: float) -> None:
        self.sim.g_adhesion = val
        self.parameters_changed.emit()

    def _on_param_droplet(self, val: float) -> None:
        self.sim.droplet_radius = int(val) if val > 0 else None
        self.parameters_changed.emit()

    def _on_param_g11(self, val: float) -> None:
        self.sim.g11 = val
        self.parameters_changed.emit()

    def _on_param_g22(self, val: float) -> None:
        self.sim.g22 = val
        self.parameters_changed.emit()

    def _on_param_g12(self, val: float) -> None:
        self.sim.g12 = val
        self.parameters_changed.emit()

    def _on_param_sigma(self, val: float) -> None:
        self.sim.sigma = val
        self.parameters_changed.emit()

    def _on_param_mc_adhesion(self, val: float) -> None:
        self.sim.g_adhesion = val
        self.parameters_changed.emit()

    def _on_param_visc(self, val: float) -> None:
        self.scene.viscosity = val
        self.sim.viscosity = val
        self.parameters_changed.emit()

    def _on_param_u_inflow(self, val: float) -> None:
        self.scene.u_inflow = val
        self.sim.u_inflow = val
        self.parameters_changed.emit()

    def _on_param_diff(self, val: float) -> None:
        self.scene.smoke_diffusion = val
        self.sim.smoke_diffusion = val
        self.parameters_changed.emit()

    def _on_param_decay(self, val: float) -> None:
        self.scene.smoke_decay = val
        self.sim.smoke_decay = val
        self.parameters_changed.emit()

    def sync_params_from_scene(self) -> None:
        """Update spinner values to match the current scene (after load)."""
        self._param_visc.setValue(self.scene.viscosity)
        self._param_u_inflow.setValue(self.scene.u_inflow)
        self._param_diff.setValue(self.scene.smoke_diffusion)
        self._param_decay.setValue(self.scene.smoke_decay)

    def _on_trail_length(self, val: int) -> None:
        tracer = self.sim.get_particle_tracer()
        if tracer is not None:
            tracer.set_trail_length(val)

    def _add_particles_random(self) -> None:
        tracer = self.sim.get_particle_tracer()
        if tracer is not None:
            tracer.add_particles_random(50)
            self._particle_count_label.setText(f"{tracer.count} particles")

    def _clear_particles(self) -> None:
        tracer = self.sim.get_particle_tracer()
        if tracer is not None:
            tracer.clear()
            self._particle_count_label.setText("0 particles")

    def update_particle_count(self) -> None:
        """Update the particle count label (call from timer)."""
        tracer = self.sim.get_particle_tracer()
        if tracer is not None:
            self._particle_count_label.setText(f"{tracer.count} particles")

    def set_expert_mode(self, expert: bool) -> None:
        """Toggle between beginner and expert mode."""
        self._expert_mode = expert
        # In beginner mode, hide smoke diffusion and decay parameters
        self._param_diff.setVisible(expert)
        self._param_decay.setVisible(expert)
        # Update the form layout labels visibility
        form = self._param_visc.parent().layout()
        if form:
            form.labelForField(self._param_diff).setVisible(expert)
            form.labelForField(self._param_decay).setVisible(expert)

    def _rebuild_tree(self) -> None:
        self.tree.blockSignals(True)
        self.tree.clear()
        top_names = ["Obstacles", "Emitters", "Probes"]
        cat_lists = [self.scene.obstacles, self.scene.emitters, self.scene.probes]
        self._top_items = {}
        self._item_to_data: dict[QTreeWidgetItem, tuple[str, int]] = {}
        for idx, (name, items) in enumerate(zip(top_names, cat_lists)):
            top = QTreeWidgetItem([f"{name} ({len(items)})"])
            top.setFlags(top.flags() & ~Qt.ItemFlag.ItemIsSelectable)
            self.tree.addTopLevelItem(top)
            self._top_items[idx] = top
            for i, item in enumerate(items):
                child = QTreeWidgetItem([self._item_label(item)])
                child.setData(0, Qt.ItemDataRole.UserRole, (idx, i))
                top.addChild(child)
                self._item_to_data[child] = (idx, i)
        self.tree.blockSignals(False)

    @staticmethod
    def _item_label(item) -> str:
        name = getattr(item, "name", str(item))
        cls = type(item).__name__.replace("Spec", "")
        if isinstance(item, STLObstacle):
            return f"{name} [STL @ {Path(item.path).name}]"
        if isinstance(item, ImageObstacle):
            return f"{name} [Image @ {Path(item.path).name}]"
        x = getattr(item, "x", "?")
        y = getattr(item, "y", "?")
        return f"{name} [{cls} @ {x},{y}]"

    def _on_item_selected(self, current: QTreeWidgetItem, _prev) -> None:
        self._clear_props()
        if current is None:
            return
        data = current.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        cat_idx, item_idx = data
        cat_name = _CATEGORY_KEY[cat_idx]
        obj = getattr(self.scene, cat_name)[item_idx]
        self._current_item = current
        self._current_editor = _PropEditor(obj, self)
        self._current_editor.changed.connect(self._on_prop_changed)
        self.props_layout.insertWidget(0, self._current_editor)
        self._props_placeholder.hide()

    def _clear_props(self) -> None:
        if self._current_editor:
            self.props_layout.removeWidget(self._current_editor)
            self._current_editor.deleteLater()
            self._current_editor = None
        self._current_item = None
        self._props_placeholder.show()

    def _on_prop_changed(self) -> None:
        self._refresh_current_label()
        self._reapply()
        self.scene_changed.emit()

    def _refresh_current_label(self) -> None:
        if self._current_item is None:
            return
        data = self._current_item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        cat_idx, item_idx = data
        cat_name = _CATEGORY_KEY[cat_idx]
        obj = getattr(self.scene, cat_name)[item_idx]
        self._current_item.setText(0, self._item_label(obj))

    def _add_obstacle(self, kind: str) -> None:
        obs = copy.deepcopy(_OBSTACLE_DEFAULTS[kind])
        self.scene.obstacles.append(obs)
        self._reapply()
        self._rebuild_tree()
        self.scene_changed.emit()

    def _import_stl(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self, "Import STL Mesh", "", "STL Files (*.stl);;All Files (*)"
        )
        if not path:
            return
        obs = STLObstacle(
            name=Path(path).stem,
            path=path,
            scale=1.0,
            offset_x=0,
            offset_y=0,
        )
        self.scene.obstacles.append(obs)
        self._reapply()
        self._rebuild_tree()
        self.scene_changed.emit()

    def _import_image(self) -> None:
        from PySide6.QtWidgets import QFileDialog

        path, _ = QFileDialog.getOpenFileName(
            self,
            "Import Image as Obstacle",
            "",
            "Images (*.png *.bmp *.jpg *.jpeg);;All Files (*)",
        )
        if not path:
            return
        obs = ImageObstacle(
            name=Path(path).stem,
            path=path,
            threshold=128,
            invert=False,
        )
        self.scene.obstacles.append(obs)
        self._reapply()
        self._rebuild_tree()
        self.scene_changed.emit()

    def _add_emitter(self) -> None:
        emit = EmitterSpec(
            name="Emitter",
            x=self.scene.width // 2,
            y=self.scene.height // 2,
        )
        self.scene.emitters.append(emit)
        self._reapply()
        self._rebuild_tree()
        self.scene_changed.emit()

    def _add_probe(self) -> None:
        cx, cy = self.scene.width // 2, self.scene.height // 2
        probe = ProbeSpec(name="Probe", x=cx, y=cy)
        self.scene.probes.append(probe)
        self._rebuild_tree()
        self.scene_changed.emit()

    def _remove_selected(self) -> None:
        item = self.tree.currentItem()
        if item is None:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if data is None:
            return
        cat_idx, item_idx = data
        cat_name = _CATEGORY_KEY[cat_idx]
        lst = getattr(self.scene, cat_name)
        if 0 <= item_idx < len(lst):
            del lst[item_idx]
        self._clear_props()
        self._reapply()
        self._rebuild_tree()
        self.scene_changed.emit()

    def add_obstacle_from_viewport(self, obs: ObstacleSpec) -> None:
        self.scene.obstacles.append(obs)
        self._reapply()
        self._rebuild_tree()
        self.scene_changed.emit()

    def _reapply(self) -> None:
        apply_to_sim(self.scene, self.sim)

    def refresh(self) -> None:
        self._rebuild_tree()
        self._reapply()
