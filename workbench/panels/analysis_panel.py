from __future__ import annotations

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QLabel,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from engines.base import SimEngine
from scene.probe import Probe

_COLORS = ["#4fc3f7", "#ff7043", "#66bb6a", "#ffca28", "#ab47bc"]


class _ProbePlot(QWidget):
    def __init__(self, probe: Probe, index: int, parent=None):
        super().__init__(parent)
        self.probe = probe
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        header = QWidget()
        hrow = QVBoxLayout(header)
        hrow.setContentsMargins(0, 0, 0, 0)

        name_row = QWidget()
        nr = QVBoxLayout(name_row)
        nr.setContentsMargins(0, 0, 0, 0)
        loc = f"({probe.spec.x}, {probe.spec.y})"
        self.name_label = QLabel(f"<b>{probe.spec.name}</b>  @ {loc}")
        nr.addWidget(self.name_label)

        self.field_combo = QComboBox()
        for f in probe.spec.fields:
            self.field_combo.addItem(f)
        nr.addWidget(self.field_combo)

        hrow.addWidget(name_row)
        layout.addWidget(header)

        self.plot = pg.PlotWidget()
        self.plot.setMinimumHeight(120)
        self.plot.setMaximumHeight(200)
        self.plot.showGrid(True, True, 0.3)
        self.plot.setLabel("left", probe.spec.fields[0] if probe.spec.fields else "")
        self.plot.setLabel("bottom", "step")
        color = _COLORS[index % len(_COLORS)]
        self.curve = self.plot.plot(pen=color)
        layout.addWidget(self.plot)

    def update_plot(self) -> None:
        field = self.field_combo.currentText()
        data = self.probe.history.get(field, [])
        if len(data) < 2:
            return
        self.curve.setData(data)
        self.plot.setLabel("left", field)


class AnalysisPanel(QWidget):
    def __init__(self, sim: SimEngine, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.probes: list[Probe] = []
        self._probe_widgets: list[_ProbePlot] = []
        self._tick_counter = 0

        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(6)

        # --- Physics readouts ---
        self.physics_group = QGroupBox("Physics Readouts")
        pf = QFormLayout()
        self.re_label = QLabel("—")
        self.st_label = QLabel("—")
        self.cd_label = QLabel("—")
        pf.addRow("Re", self.re_label)
        pf.addRow("St", self.st_label)
        pf.addRow("Cd", self.cd_label)
        self.physics_group.setLayout(pf)
        layout.addWidget(self.physics_group)

        # --- Probe plots (scrollable) ---
        self.probes_group = QGroupBox("Probe Plots")
        self.probes_scroll = QScrollArea()
        self.probes_scroll.setWidgetResizable(True)
        self.probes_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self.probes_container = QWidget()
        self.probes_layout = QVBoxLayout(self.probes_container)
        self.probes_layout.setContentsMargins(0, 0, 0, 0)
        self.probes_layout.setSpacing(4)
        self.probes_scroll.setWidget(self.probes_container)

        self._no_probes_label = QLabel(
            "No probes placed.\nClick Probe tool then click viewport to add one."
        )
        self._no_probes_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._no_probes_label.setStyleSheet("color: #64748b; padding: 20px;")
        self._no_probes_label.setVisible(True)

        pg = QVBoxLayout()
        pg.addWidget(self._no_probes_label)
        pg.addWidget(self.probes_scroll)
        self.probes_group.setLayout(pg)
        layout.addWidget(self.probes_group, 1)

        # --- Field statistics ---
        self.stats_group = QGroupBox("Field Statistics")
        sf = QFormLayout()
        self.min_label = QLabel("—")
        self.max_label = QLabel("—")
        self.mean_label = QLabel("—")
        sf.addRow("Min", self.min_label)
        sf.addRow("Max", self.max_label)
        sf.addRow("Mean", self.mean_label)
        self.stats_group.setLayout(sf)
        layout.addWidget(self.stats_group)

    def set_probes(self, probes: list[Probe]) -> None:
        self.probes = probes
        self._rebuild_probe_widgets()

    def _rebuild_probe_widgets(self) -> None:
        for w in self._probe_widgets:
            self.probes_layout.removeWidget(w)
            w.deleteLater()
        self._probe_widgets.clear()
        has_probes = len(self.probes) > 0
        self._no_probes_label.setVisible(not has_probes)
        self.probes_scroll.setVisible(has_probes)
        for i, probe in enumerate(self.probes):
            pw = _ProbePlot(probe, i)
            self.probes_layout.addWidget(pw)
            self._probe_widgets.append(pw)
        if has_probes:
            self.probes_layout.addStretch()

    def tick(self, dt: float = 1.0) -> None:
        self._tick_counter += 1

        if self._tick_counter % 2 == 0:
            for probe in self.probes:
                probe.record(self.sim)

        if self._tick_counter % 5 == 0:
            self._update_physics(dt)
            self._update_field_stats()

        if self._tick_counter % 3 == 0:
            for pw in self._probe_widgets:
                pw.update_plot()

    def _update_physics(self, dt: float) -> None:
        from analysis.physics import drag_coefficient, reynolds_number, strouhal_number

        Re = reynolds_number(self.sim)
        self.re_label.setText(f"{Re:.1f}")

        Cd = drag_coefficient(self.sim)
        self.cd_label.setText(f"{Cd:.3f}")

        St = None
        if self.probes and self.sim.u_inflow > 0:
            v_data = self.probes[0].history.get("v", [])
            diam = 1.0
            St = strouhal_number(v_data, dt, diameter=diam, velocity=self.sim.u_inflow)
        self.st_label.setText(f"{St:.3f}" if St is not None else "—")

    def set_colormap(self, cmap: str) -> None:
        self._colormap = cmap

    def _update_field_stats(self) -> None:
        cmap = getattr(self, "_colormap", "smoke")
        if cmap == "smoke":
            field = self.sim.get_smoke()
        elif cmap in ("density", "phase", "pressure"):
            rho = self.sim.get_density()
            field = rho if cmap == "density" else rho - 1.0
        else:
            vel = self.sim.get_velocity()
            field = np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2)
        self.min_label.setText(f"{field.min():.4f}")
        self.max_label.setText(f"{field.max():.4f}")
        self.mean_label.setText(f"{field.mean():.4f}")
