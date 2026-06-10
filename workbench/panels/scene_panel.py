from __future__ import annotations

import copy
from dataclasses import fields

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
    CircleObstacle,
    EmitterSpec,
    ObstacleSpec,
    ProbeSpec,
    RectObstacle,
    Scene,
    apply_to_sim,
)

_CATEGORY_KEY = {0: "obstacles", 1: "emitters", 2: "probes"}

_OBSTACLE_DEFAULTS: dict[str, ObstacleSpec] = {
    "Circle": CircleObstacle(name="Circle", x=32, y=32, radius=8),
    "Rectangle": RectObstacle(name="Rect", x=20, y=20, w=16, h=16),
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
            elif f.name in ("fields", "points"):
                continue
            elif isinstance(val, int):
                w = QSpinBox()
                w.setRange(0, 1023)
                w.setValue(val)
                w.valueChanged.connect(lambda v, n=f.name: self._set(n, v))
            elif isinstance(val, float):
                w = QDoubleSpinBox()
                w.setRange(0.0, 10.0)
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
        add_menu.addAction(
            "Circle Obstacle", lambda: self._add_obstacle("Circle")
        )
        add_menu.addAction(
            "Rectangle Obstacle", lambda: self._add_obstacle("Rectangle")
        )
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
        form.addRow("Viscosity", self._param_visc)

        self._param_u_inflow = QDoubleSpinBox()
        self._param_u_inflow.setRange(0.0, 0.5)
        self._param_u_inflow.setSingleStep(0.005)
        self._param_u_inflow.setDecimals(4)
        self._param_u_inflow.setValue(self.sim.u_inflow)
        self._param_u_inflow.valueChanged.connect(self._on_param_u_inflow)
        form.addRow("Inflow", self._param_u_inflow)

        self._param_diff = QDoubleSpinBox()
        self._param_diff.setRange(0.0, 0.5)
        self._param_diff.setSingleStep(0.001)
        self._param_diff.setDecimals(4)
        self._param_diff.setValue(self.sim.smoke_diffusion)
        self._param_diff.valueChanged.connect(self._on_param_diff)
        form.addRow("Smoke Diffusion", self._param_diff)

        self._param_decay = QDoubleSpinBox()
        self._param_decay.setRange(0.9, 1.0)
        self._param_decay.setSingleStep(0.0005)
        self._param_decay.setDecimals(4)
        self._param_decay.setValue(self.sim.smoke_decay)
        self._param_decay.valueChanged.connect(self._on_param_decay)
        form.addRow("Smoke Decay", self._param_decay)

        group.setLayout(form)
        layout.addWidget(group)

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
