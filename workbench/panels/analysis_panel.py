from __future__ import annotations

from collections import deque

import numpy as np
import pyqtgraph as pg
from PySide6.QtCore import Qt
from PySide6.QtGui import QFont
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
from scene.scene import Scene

_COLORS = ["#4fc3f7", "#ff7043", "#66bb6a", "#ffca28", "#ab47bc"]


class _ProbePlot(QWidget):
    def __init__(self, probe: Probe, index: int, parent=None):
        super().__init__(parent)
        self.probe = probe
        self._y_range = None  # Track stable y-range
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
        self.plot.setLabel(
            "left", probe.spec.fields[0] if probe.spec.fields else "", units=""
        )
        self.plot.setLabel("bottom", "step")
        self.plot.getAxis("left").setStyle(tickFont=QFont("Arial", 9))
        color = _COLORS[index % len(_COLORS)]
        self.curve = self.plot.plot(pen=color)
        layout.addWidget(self.plot)

    def update_plot(self) -> None:
        field = self.field_combo.currentText()
        data = self.probe.history.get(field, [])
        if len(data) < 2:
            return
        self.curve.setData(data)
        self.plot.setLabel("left", field, units="")

        # Stabilize y-axis range to prevent wild jumps
        data_arr = np.array(data)
        data_min, data_max = data_arr.min(), data_arr.max()
        data_range = data_max - data_min

        if self._y_range is None:
            # Initialize with first data range + 10% padding
            padding = max(0.1 * data_range, 1e-6)
            self._y_range = (data_min - padding, data_max + padding)
        else:
            # Gradually expand range if needed, never shrink
            current_min, current_max = self._y_range
            new_min = min(current_min, data_min)
            new_max = max(current_max, data_max)
            # Only expand if data significantly exceeds current range
            if data_min < current_min or data_max > current_max:
                padding = max(0.1 * (new_max - new_min), 1e-6)
                self._y_range = (new_min - padding, new_max + padding)

        self.plot.setYRange(self._y_range[0], self._y_range[1])


class AnalysisPanel(QWidget):
    def __init__(self, sim: SimEngine, parent=None):
        super().__init__(parent)
        self.sim = sim
        self.scene: Scene | None = None
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
        self.re_label.setToolTip("Reynolds number: ratio of inertial to viscous forces")
        self.st_label = QLabel("—")
        self.st_label.setToolTip(
            "Strouhal number: vortex shedding frequency (requires periodic shedding)"
        )
        self.cd_label = QLabel("—")
        self.cd_label.setToolTip(
            "Drag coefficient: flow resistance (may fluctuate while developing)"
        )
        pf.addRow("Re", self.re_label)
        pf.addRow("St", self.st_label)
        pf.addRow("Cd", self.cd_label)
        self.physics_group.setLayout(pf)
        layout.addWidget(self.physics_group)

        # --- Stability strip chart (live max |u|) ---
        self.stability_group = QGroupBox("Stability (max |u|)")
        st_layout = QVBoxLayout()
        st_layout.setContentsMargins(4, 4, 4, 4)
        self.stability_plot = pg.PlotWidget()
        self.stability_plot.setMinimumHeight(56)
        self.stability_plot.setMaximumHeight(72)
        self.stability_plot.hideAxis("left")
        self.stability_plot.hideAxis("bottom")
        self.stability_plot.setMenuEnabled(False)
        self.stability_plot.setMouseEnabled(False, False)
        self.stability_plot.getViewBox().setDefaultPadding(0.0)
        self.stability_curve = self.stability_plot.plot(pen="#3fb950")
        self._stability_data: deque[float] = deque(maxlen=256)
        st_layout.addWidget(self.stability_plot)
        self.stability_group.setLayout(st_layout)
        layout.addWidget(self.stability_group)

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

        probes_outer = QVBoxLayout()
        probes_outer.addWidget(self._no_probes_label)
        probes_outer.addWidget(self.probes_scroll)
        self.probes_group.setLayout(probes_outer)
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

    def reset_runtime(self) -> None:
        """Drop everything derived from a previous run.

        Called by the app whenever simulation state is reset, so a fresh run does
        not inherit the previous run's measurements. Without this the stability
        trace keeps drawing the old convergence curve beside the new one, and
        `_tick_counter` stays offset so probe sampling no longer lines up with
        the step count it is plotted against.
        """
        self._stability_data.clear()
        self._tick_counter = 0
        for probe in self.probes:
            probe.clear()
        if self.stability_curve is not None:
            self.stability_curve.setData([], [])

    def set_probes(self, probes: list[Probe]) -> None:
        self.probes = probes
        self._rebuild_probe_widgets()

    def _rebuild_probe_widgets(self) -> None:
        for w in self._probe_widgets:
            self.probes_layout.removeWidget(w)
            w.deleteLater()
        self._probe_widgets.clear()
        while self.probes_layout.count():
            item = self.probes_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
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

        self._record_stability()

        if self._tick_counter % 2 == 0:
            # One shared per-step field cache: a field is a whole-domain array,
            # so without it every probe would recompute each field it watches.
            cache: dict[str, np.ndarray] = {}
            for probe in self.probes:
                probe.record(self.sim, cache)

        if self._tick_counter % 5 == 0:
            self._update_physics(dt)
            self._update_field_stats()

        if self._tick_counter % 3 == 0:
            for pw in self._probe_widgets:
                pw.update_plot()

    def _record_stability(self) -> None:
        try:
            vel = self.sim.get_velocity()
            max_u = float(np.sqrt(vel[:, :, 0] ** 2 + vel[:, :, 1] ** 2).max())
        except Exception:
            return
        self._stability_data.append(max_u)
        if self._tick_counter % 3 == 0 and len(self._stability_data) > 1:
            data = list(self._stability_data)
            self.stability_curve.setData(data)
            lo = min(data)
            hi = max(data)
            pad = max((hi - lo) * 0.1, 1e-4)
            self.stability_plot.setYRange(lo - pad, hi + pad, padding=0)
            self.stability_plot.setXRange(0, self._stability_data.maxlen, padding=0)

    def set_scene(self, scene: Scene) -> None:
        self.scene = scene

    def _update_physics(self, dt: float) -> None:
        from analysis.physics import (
            characteristic_length,
            drag_coefficient,
            reynolds_number,
            strouhal_number,
        )

        diam = characteristic_length(self.scene) if self.scene is not None else 1.0
        Re = reynolds_number(self.sim, diam)
        self.re_label.setText(f"{Re:.1f}")

        Cd = drag_coefficient(self.sim)
        self.cd_label.setText(f"{Cd:.3f}")

        St = None
        if self.probes and self.sim.u_inflow > 0:
            v_data = self.probes[0].history.get("v", [])
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
